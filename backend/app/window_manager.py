"""Timestamp-aligned sensor windows and asynchronous prediction processing."""

from __future__ import annotations

import logging
import math
import threading
import time
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from .feature_extractor import extract_window_features
from .predictor import StressPredictor

LOGGER = logging.getLogger(__name__)
WINDOW_SECONDS = 60
MODALITIES = ("bvp", "eda", "temperature", "accelerometer")
EXPECTED_RATES = {"bvp": 64, "eda": 4, "temperature": 4, "accelerometer": 32}
EXPECTED_COUNTS = {name: rate * WINDOW_SECONDS for name, rate in EXPECTED_RATES.items()}
MAX_GAP_MULTIPLIER = 1.5
TIMING_TOLERANCE = 0.25


def parse_timestamp(value: str) -> datetime:
    """Parse an ISO-8601 timestamp and normalise it to UTC."""
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO-8601 string")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("timestamp is not valid ISO-8601") from error
    if timestamp.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return timestamp.astimezone(timezone.utc)


@dataclass(frozen=True)
class _Sample:
    sequence: int
    timestamp: datetime
    value: float | tuple[float, float, float]


@dataclass
class _Window:
    device_id: str
    window_start: int
    window_id: str
    first_received_monotonic: float
    last_received_monotonic: float
    samples: dict[str, list[_Sample]] = field(default_factory=lambda: {name: [] for name in MODALITIES})
    last_progress: int = -1

    @property
    def counts(self) -> dict[str, int]:
        return {modality: len(samples) for modality, samples in self.samples.items()}


class WindowManager:
    """Buffer timestamp-aligned sensor windows independently for each device.

    Windows are keyed by their UTC epoch-aligned start and accept only the
    canonical schema validated by ``mqtt_handler.validate_sensor_message``.
    A window is handed off once only when every modality has its exact expected
    count and continuous timestamp coverage. Expired or invalid windows are
    reported as incomplete; missing samples are never synthesized.
    """

    def __init__(self, predictor: StressPredictor | None = None, status_callback: Callable[[dict], None] | None = None, prediction_callback: Callable[[dict], None] | None = None, window_callback: Callable[[dict], None] | None = None, window_seconds: int = WINDOW_SECONDS, timeout_seconds: float = 90, clock: Callable[[], float] = time.monotonic) -> None:
        if isinstance(window_seconds, bool) or not isinstance(window_seconds, int) or window_seconds <= 0:
            raise ValueError("window_seconds must be a positive integer")
        self.predictor = predictor or StressPredictor()
        self.status_callback = status_callback
        self.prediction_callback = prediction_callback
        self.window_callback = window_callback
        self.window_seconds = window_seconds
        self.expected_counts = {
            modality: rate * self.window_seconds
            for modality, rate in EXPECTED_RATES.items()
        }
        self.timeout_seconds = timeout_seconds
        self.clock = clock
        self._windows: dict[str, dict[int, _Window]] = {}
        self._retired_windows: set[tuple[str, int]] = set()
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="stress-window")

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    def expire_stalled_windows(self) -> int:
        now = self.clock()
        expired: list[_Window] = []
        with self._lock:
            for device_id, device_windows in list(self._windows.items()):
                for start, window in list(device_windows.items()):
                    if now - window.last_received_monotonic >= self.timeout_seconds:
                        del device_windows[start]
                        self._retired_windows.add((device_id, start))
                        expired.append(window)
                if not device_windows:
                    self._windows.pop(device_id, None)
        for window in expired:
            self._emit_status(window, "WINDOW_INCOMPLETE", "Window timed out before all samples arrived")
        return len(expired)

    def ingest(self, message: dict) -> bool:
        """Accept one already-validated MQTT message, returning whether it was buffered."""
        modality = message.get("modality")
        if modality not in MODALITIES:
            return False
        try:
            timestamp = parse_timestamp(message["timestamp"])
            device_id = message["device_id"]
            sequence = message["sequence"]
            if not isinstance(device_id, str) or not device_id or isinstance(sequence, bool) or not isinstance(sequence, int):
                return False
            value: float | tuple[float, float, float]
            if modality == "accelerometer":
                if any(isinstance(message[name], bool) for name in ("acc_x", "acc_y", "acc_z")):
                    return False
                value = (float(message["acc_x"]), float(message["acc_y"]), float(message["acc_z"]))
                if not all(math.isfinite(component) for component in value):
                    return False
            else:
                if isinstance(message["value"], bool):
                    return False
                value = float(message["value"])
                if not math.isfinite(value):
                    return False
        except (KeyError, TypeError, ValueError):
            return False

        window_start = int(timestamp.timestamp()) // self.window_seconds * self.window_seconds
        sample = _Sample(sequence=sequence, timestamp=timestamp, value=value)
        now = self.clock()
        expired: list[_Window] = []
        process: _Window | None = None
        started = False
        with self._lock:
            device_windows = self._windows.setdefault(device_id, {})
            for start, candidate in list(device_windows.items()):
                if now - candidate.last_received_monotonic >= self.timeout_seconds:
                    del device_windows[start]
                    self._retired_windows.add((device_id, start))
                    expired.append(candidate)
            if (device_id, window_start) in self._retired_windows:
                return False
            window = device_windows.get(window_start)
            if window is None:
                window = _Window(device_id, window_start, f"{device_id}-{window_start}", now, now)
                device_windows[window_start] = window
                started = True
            samples = window.samples[modality]
            insertion_index = bisect_left([item.timestamp for item in samples], sample.timestamp)
            if not self._is_valid_insertion(window, modality, sample, insertion_index):
                return False
            samples.insert(insertion_index, sample)
            window.last_received_monotonic = now
            if all(len(window.samples[name]) == self.expected_counts[name] for name in MODALITIES) and self._ready_for_processing(window, now):
                del device_windows[window_start]
                if not device_windows:
                    self._windows.pop(device_id, None)
                self._retired_windows.add((device_id, window_start))
                process = window
        for candidate in expired:
            self._emit_status(candidate, "WINDOW_INCOMPLETE", "Window timed out before all samples arrived")
        if started:
            self._emit_status(window, "WINDOW_STARTED", "Started collecting a 60-second sensor window")
        self._emit_progress(window)
        if process is not None:
            self._emit_status(process, "WINDOW_COMPLETED", "All required samples received")
            if self.window_callback:
                self.window_callback(self._completed_window(process))
            self._executor.submit(self._process_window, process)
        return True

    def _is_valid_insertion(self, window: _Window, modality: str, sample: _Sample, insertion_index: int) -> bool:
        existing = window.samples[modality]
        if len(existing) >= self.expected_counts[modality] or any(item.sequence == sample.sequence for item in existing):
            return False
        neighbors: list[_Sample] = []
        if insertion_index:
            neighbors.append(existing[insertion_index - 1])
        if insertion_index < len(existing):
            neighbors.append(existing[insertion_index])
        expected_interval = 1.0 / EXPECTED_RATES[modality]
        for item in neighbors:
            sequence_delta = abs(sample.sequence - item.sequence)
            timestamp_delta = abs((sample.timestamp - item.timestamp).total_seconds())
            tolerance = max(expected_interval * TIMING_TOLERANCE, 0.002)
            if sequence_delta == 0 or abs(timestamp_delta - sequence_delta * expected_interval) > tolerance:
                return False
        return True

    def _ready_for_processing(self, window: _Window, arrival_time: float) -> bool:
        min_duration = self.window_seconds - 1
        if arrival_time - window.first_received_monotonic < min_duration:
            return False
        for modality, samples in window.samples.items():
            ordered = sorted(samples, key=lambda item: item.timestamp)
            coverage = (ordered[-1].timestamp - ordered[0].timestamp).total_seconds()
            interval = 1.0 / EXPECTED_RATES[modality]
            if coverage < min_duration:
                return False
            if any((current.timestamp - previous.timestamp).total_seconds() > interval * MAX_GAP_MULTIPLIER + 0.002 for previous, current in zip(ordered, ordered[1:])):
                return False
        return True

    def _emit_progress(self, window: _Window) -> None:
        ratio = min(len(window.samples[name]) / expected for name, expected in self.expected_counts.items())
        milestone = min(100, int(ratio * 10) * 10)
        if milestone > window.last_progress:
            window.last_progress = milestone
            self._emit_status(window, "COLLECTING_SAMPLES", f"Collected approximately {milestone}% of the required samples")

    def _emit_status(self, window: _Window, stage: str, message: str) -> None:
        if self.status_callback:
            self.status_callback({"device_id": window.device_id, "window_id": window.window_id, "timestamp": datetime.now(timezone.utc).isoformat(), "stage": stage, "message": message, "received_sample_counts": window.counts, "expected_sample_counts": self.expected_counts.copy()})

    def _completed_window(self, window: _Window) -> dict:
        """Return the single feature-extractor-compatible handoff payload."""
        ordered = {
            modality: sorted(window.samples[modality], key=lambda item: item.timestamp)
            for modality in MODALITIES
        }
        return {
            "device_id": window.device_id,
            "window_id": window.window_id,
            "window_start": datetime.fromtimestamp(window.window_start, timezone.utc).isoformat(),
            "window_duration_seconds": self.window_seconds,
            "sample_counts": window.counts,
            "reading": {
                "bvp": [item.value for item in ordered["bvp"]],
                "eda": [item.value for item in ordered["eda"]],
                "skin_temperature": [item.value for item in ordered["temperature"]],
                "acc_x": [item.value[0] for item in ordered["accelerometer"]],
                "acc_y": [item.value[1] for item in ordered["accelerometer"]],
                "acc_z": [item.value[2] for item in ordered["accelerometer"]],
            },
        }

    def _process_window(self, window: _Window) -> None:
        try:
            self._emit_status(window, "EXTRACTING_FEATURES", "Extracting model features")
            completed_window = self._completed_window(window)
            self._emit_status(window, "PREDICTING", "Running stress prediction")
            predict_window = getattr(self.predictor, "predict_window", None)
            if callable(predict_window):
                result = predict_window(completed_window)
            else:
                # Preserve compatibility with lightweight predictor doubles.
                features = extract_window_features(completed_window)
                result = self.predictor.predict(features)
            self._emit_status(window, "FEATURES_READY", "27 model features extracted")
            self._emit_status(window, "PREDICTION_READY", "Stress prediction completed")
            prediction = {"device_id": window.device_id, "window_id": window.window_id, "timestamp": datetime.now(timezone.utc).isoformat(), "window_duration_seconds": self.window_seconds, **result, "model_version": self.predictor.metadata.get("model_version", "unknown")}
            if self.prediction_callback:
                self.prediction_callback(prediction)
        except Exception as error:
            LOGGER.exception("Window processing failed for %s", window.window_id)
            self._emit_status(window, "WINDOW_ERROR", f"Window processing failed: {error}")
        finally:
            self._emit_status(window, "WINDOW_RESET", "Window buffer retired")
