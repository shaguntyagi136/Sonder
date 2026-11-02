"""Active ingestion pipeline: validate -> window -> features -> inference -> storage."""
from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from pydantic import ValidationError

from app.core.errors import InferenceError, ModelUnavailableError, PreprocessingError, SensorValidationError, StorageError
from app.core.timeutil import utcnow
from app.database.base import Repository
from app.ml.service import InferenceService
from app.mqtt.parsing import decode_message, device_id_from_topic
from app.preprocessing.preprocessor import Preprocessor
from app.schemas.records import LiveTelemetry, StoredPrediction, StoredReading
from app.schemas.sensor import SensorPayload
from app.services.metrics import Metrics

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionOutcome:
    status: str
    reading: StoredReading
    prediction: StoredPrediction | None = None
    error: str | None = None


_EMPTY_COUNTS = {"bvp": 0, "eda": 0, "skin_temperature": 0, "acc_x": 0, "acc_y": 0, "acc_z": 0}
LIVE_STALE_AFTER_SECONDS = 5.0
RAW_WINDOW_SECONDS = 60.0


@dataclass
class _SampleWindow:
    device_id: str
    subject_id: str | None
    start: datetime
    sample_rates_hz: dict[str, float]
    samples: dict[str, list[float]] = field(default_factory=lambda: {"bvp": [], "eda": [], "skin_temperature": [], "acc_x": [], "acc_y": [], "acc_z": []})
    last_timestamps: dict[str, datetime] = field(default_factory=dict)
    last_logged_sample_total: int = 0


@dataclass
class _LiveSample:
    device_id: str
    subject_id: str | None
    timestamp: datetime
    received_at: datetime
    last_modality: str
    bvp: float | None = None
    eda: float | None = None
    skin_temperature: float | None = None
    acc_x: float | None = None
    acc_y: float | None = None
    acc_z: float | None = None
    sample_counts: dict[str, int] = field(default_factory=lambda: dict(_EMPTY_COUNTS))


def _validation_errors(exc: ValidationError) -> list[dict[str, str]]:
    return [{"field": ".".join(str(part) for part in error["loc"]) or "payload", "message": error["msg"]} for error in exc.errors()]


class IngestionPipeline:
    """Repository/inference pipeline for full HTTP windows and progressive MQTT samples."""

    def __init__(self, repo: Repository, inference: InferenceService, preprocessor: Preprocessor | None, *, subscription_patterns: list[str], max_payload_bytes: int):
        self.repo, self.inference, self.preprocessor = repo, inference, preprocessor
        self.subscription_patterns, self.max_payload_bytes = subscription_patterns, max_payload_bytes
        self.stats = Metrics()
        self._sample_windows: dict[str, _SampleWindow] = {}
        self._live_lock = threading.Lock()
        self._live: dict[str, _LiveSample] = {}
        self._live_latest_device_id: str | None = None

    def ingest(self, payload: Mapping[str, Any], *, transport: str = "http") -> IngestionOutcome:
        try:
            sensor_payload = SensorPayload.model_validate(payload)
        except ValidationError as exc:
            self.stats.incr("received"); self.stats.incr("rejected")
            raise SensorValidationError(_validation_errors(exc)) from None
        if not sensor_payload.device_id:
            self.stats.incr("received"); self.stats.incr("rejected")
            raise SensorValidationError([{"field": "device_id", "message": "device_id is required for persistence"}])
        return self._process(sensor_payload, transport)

    def ingest_mqtt_message(self, topic: str, raw_payload: bytes) -> IngestionOutcome | None:
        """Never raise on paho's network thread; emit an outcome only for a full window."""
        self.stats.incr("received")
        self.stats.incr("raw_mqtt_received")
        try:
            payload = decode_message(raw_payload, self.max_payload_bytes)
            topic_device_id = device_id_from_topic(topic, self.subscription_patterns)
            if topic_device_id is None:
                return self._reject("topic does not match a configured MQTT subscription")
            if payload.get("device_id") not in (None, topic_device_id):
                return self._reject("device_id does not match MQTT topic")
            payload["device_id"] = topic_device_id
            if "modality" in payload:
                return self._ingest_sample(payload)
            try:
                sensor_payload = SensorPayload.model_validate(payload)
            except ValidationError as exc:
                return self._reject("payload validation failed", _validation_errors(exc))
            return self._process(sensor_payload, "mqtt", already_counted=True)
        except Exception:
            self.stats.incr("unexpected_errors")
            log.exception("Unexpected MQTT ingestion error for topic %s", topic)
            return None

    def _process(self, payload: SensorPayload, transport: str, *, already_counted: bool = False) -> IngestionOutcome:
        if not already_counted:
            self.stats.incr("received")
        reading, features, prediction = payload.to_reading(transport=transport), None, None
        status, error = "model_unavailable", self.inference.load_error or "no model loaded"
        if self.inference.ready and self.preprocessor is not None:
            try:
                self.stats.incr("inference_started")
                log.info(
                    "Inference started: device_id=%s window_start=%s window_end=%s counts=%s",
                    reading.device_id,
                    reading.timestamp.isoformat(),
                    (reading.timestamp + timedelta(seconds=reading.window_seconds)).isoformat(),
                    self._reading_sample_counts(reading),
                )
                features = self.preprocessor.transform(reading)
                expected_names = tuple(self.inference.spec.feature_order) if self.inference.spec else ()
                if features.names != expected_names:
                    raise PreprocessingError("extracted feature names do not match model metadata")
                if not all(math.isfinite(value) for value in features.values):
                    raise PreprocessingError("extracted feature values must be finite")
                prediction = self.inference.predict(features.values)
                self.stats.incr("inference_succeeded")
                log.info(
                    "Inference succeeded: device_id=%s window_start=%s label=%s confidence=%s",
                    reading.device_id,
                    reading.timestamp.isoformat(),
                    prediction.label,
                    prediction.confidence,
                )
                status, error = "ok", None
            except PreprocessingError as exc:
                self.stats.incr("inference_failed")
                log.exception(
                    "Window preprocessing failed: device_id=%s window_start=%s window_end=%s "
                    "counts=%s error=%s",
                    reading.device_id,
                    reading.timestamp.isoformat(),
                    (reading.timestamp + timedelta(seconds=reading.window_seconds)).isoformat(),
                    self._reading_sample_counts(reading),
                    exc,
                )
                status, error = "preprocessing_failed", str(exc)
            except ModelUnavailableError as exc:
                self.stats.incr("inference_failed")
                log.exception(
                    "Window inference unavailable: device_id=%s window_start=%s window_end=%s "
                    "counts=%s error=%s",
                    reading.device_id,
                    reading.timestamp.isoformat(),
                    (reading.timestamp + timedelta(seconds=reading.window_seconds)).isoformat(),
                    self._reading_sample_counts(reading),
                    exc,
                )
                status, error = "model_unavailable", str(exc)
            except InferenceError as exc:
                self.stats.incr("inference_errors")
                self.stats.incr("inference_failed")
                log.exception(
                    "Window inference failed: device_id=%s window_start=%s window_end=%s "
                    "counts=%s error=%s",
                    reading.device_id,
                    reading.timestamp.isoformat(),
                    (reading.timestamp + timedelta(seconds=reading.window_seconds)).isoformat(),
                    self._reading_sample_counts(reading),
                    exc,
                )
                status, error = "inference_failed", str(exc)
        try:
            stored_reading, stored_prediction = self.repo.save(reading, features, prediction, status, error)
        except StorageError:
            self.stats.incr("storage_errors")
            self.stats.incr("persistence_failed")
            log.exception(
                "Prediction persistence failed: device_id=%s window_start=%s status=%s error=%s",
                reading.device_id,
                reading.timestamp.isoformat(),
                status,
                error,
            )
            raise
        self.stats.incr("persistence_succeeded")
        if stored_prediction is not None:
            log.info(
                "Prediction succeeded: device_id=%s window_start=%s label=%s confidence=%s "
                "prediction_id=%s reading_id=%s",
                reading.device_id,
                reading.timestamp.isoformat(),
                stored_prediction.label,
                stored_prediction.confidence,
                stored_prediction.id,
                stored_prediction.reading_id,
            )
        else:
            log.warning(
                "Prediction unavailable: device_id=%s window_start=%s status=%s error=%s "
                "reading_id=%s",
                reading.device_id,
                reading.timestamp.isoformat(),
                status,
                error,
                stored_reading.id,
            )
        self.stats.incr("predicted" if stored_prediction else "stored_without_prediction")
        return IngestionOutcome(status, stored_reading, stored_prediction, error)

    def _ingest_sample(self, payload: Mapping[str, Any]) -> IngestionOutcome | None:
        modality = payload.get("modality")
        if modality not in {"bvp", "eda", "skin_temperature", "accelerometer"}:
            return self._reject("unknown raw-sample modality")
        try:
            timestamp = datetime.fromisoformat(str(payload["timestamp"]).replace("Z", "+00:00")).astimezone(timezone.utc)
            rate, device_id = float(payload["sample_rate_hz"]), str(payload["device_id"])
            if rate <= 0:
                raise ValueError("sample_rate_hz must be positive")
            subject_id = str(payload["subject_id"]) if payload.get("subject_id") is not None else None
            values = self._sample_values(modality, payload)
        except (KeyError, TypeError, ValueError) as exc:
            return self._reject(f"invalid raw-sample payload: {exc}")
        window = self._sample_windows.get(device_id)
        if window is None or timestamp >= self._window_end(window):
            if window is not None:
                complete = self._window_complete(window)
                self.stats.incr("raw_windows_rolled_over")
                if not complete:
                    self.stats.incr("raw_windows_incomplete")
                log.warning(
                    "Raw MQTT window rollover: device_id=%s reset=%s reason=%s start=%s end=%s elapsed=%.3fs "
                    "last_timestamp=%s counts=%s expected=%s",
                    device_id,
                    not complete,
                    "complete window ended" if complete else self._incomplete_reason(window),
                    window.start.isoformat(),
                    self._window_end(window).isoformat(),
                    (timestamp - window.start).total_seconds(),
                    timestamp.isoformat(),
                    self._sample_counts(window),
                    self._expected_counts(window),
                )
                if complete:
                    self._process_completed_window(window, timestamp, rate)
            window = _SampleWindow(device_id, subject_id, timestamp, self._initial_rates(modality, rate))
            self._sample_windows[device_id] = window
            self.stats.incr("raw_windows_started")
            log.info(
                "Raw MQTT window started: device_id=%s modality=%s timestamp=%s sample_rate_hz=%.3f "
                "start=%s end=%s",
                device_id,
                modality,
                timestamp.isoformat(),
                rate,
                window.start.isoformat(),
                self._window_end(window).isoformat(),
            )
        if timestamp < window.start or (window.subject_id is not None and subject_id not in (None, window.subject_id)):
            return self._reject("raw-sample timestamp or subject conflicts with active window")
        try:
            self._append_sample(window, modality, rate, timestamp, values)
        except ValueError as exc:
            self.stats.incr("raw_samples_rejected")
            log.warning(
                "Raw MQTT sample rejected: device_id=%s modality=%s timestamp=%s "
                "window_start=%s window_end=%s counts=%s reason=%s",
                device_id,
                modality,
                timestamp.isoformat(),
                window.start.isoformat(),
                self._window_end(window).isoformat(),
                self._sample_counts(window),
                exc,
            )
            return self._reject(
                f"raw-sample rejected for device_id={device_id}, modality={modality}: {exc}"
            )
        self._update_live_telemetry(window, modality, timestamp, values)
        self.stats.incr("raw_samples_accepted")
        self.stats.incr(f"raw_samples_accepted_{modality}")
        log.debug(
            "Raw MQTT sample accepted: device_id=%s modality=%s timestamp=%s "
            "window_start=%s window_end=%s elapsed=%.6fs counts=%s expected=%s incomplete=%s",
            device_id,
            modality,
            timestamp.isoformat(),
            window.start.isoformat(),
            self._window_end(window).isoformat(),
            (timestamp - window.start).total_seconds(),
            self._sample_counts(window),
            self._expected_counts(window),
            self._incomplete_reason(window),
        )
        self._log_window_progress(window, modality, timestamp)
        complete = self._window_complete(window)
        if not complete:
            return None
        return self._process_completed_window(window, timestamp, rate)

    @staticmethod
    def _sample_values(modality: str, payload: Mapping[str, Any]) -> tuple[float, ...]:
        if modality == "accelerometer":
            return float(payload["acc_x"]), float(payload["acc_y"]), float(payload["acc_z"])
        return (float(payload["value"]),)

    @staticmethod
    def _initial_rates(modality: str, rate: float) -> dict[str, float]:
        rates = {"bvp": 0.0, "eda": 0.0, "skin_temperature": 0.0, "acc_x": 0.0, "acc_y": 0.0, "acc_z": 0.0}
        if modality == "accelerometer":
            rates.update({"acc_x": rate, "acc_y": rate, "acc_z": rate})
        else:
            rates[modality] = rate
        return rates

    @staticmethod
    def _append_sample(
        window: _SampleWindow,
        modality: str,
        rate: float,
        timestamp: datetime,
        values: tuple[float, ...],
    ) -> None:
        names = ("acc_x", "acc_y", "acc_z") if modality == "accelerometer" else (modality,)
        for name, value in zip(names, values):
            if window.sample_rates_hz[name] not in (0.0, rate):
                raise ValueError(f"sample rate changed for {name}")
            if name in window.last_timestamps and timestamp <= window.last_timestamps[name]:
                raise ValueError(f"duplicate or out-of-order timestamp for {name}")
        for name, value in zip(names, values):
            window.sample_rates_hz[name] = rate
            window.samples[name].append(value)
            window.last_timestamps[name] = timestamp

    @staticmethod
    def _reading_sample_counts(reading: StoredReading) -> dict[str, int]:
        return {
            "bvp": len(reading.bvp),
            "eda": len(reading.eda),
            "skin_temperature": len(reading.skin_temperature),
            "acc_x": len(reading.acc_x),
            "acc_y": len(reading.acc_y),
            "acc_z": len(reading.acc_z),
        }

    def _process_completed_window(
        self,
        window: _SampleWindow,
        timestamp: datetime,
        rate: float,
    ) -> IngestionOutcome | None:
        counts = self._sample_counts(window)
        expected = self._expected_counts(window)
        self.stats.incr("raw_windows_completed")
        log.info(
            "Raw MQTT window completed: device_id=%s modality=%s timestamp=%s sample_rate_hz=%.3f "
            "start=%s end=%s elapsed=%.3fs counts=%s expected=%s completion_condition=True; passing payload to ingest",
            window.device_id,
            window.subject_id,
            timestamp.isoformat(),
            rate,
            window.start.isoformat(),
            self._window_end(window).isoformat(),
            (timestamp - window.start).total_seconds(),
            counts,
            expected,
        )
        self._sample_windows.pop(window.device_id, None)
        complete_payload = {
            "device_id": window.device_id,
            "subject_id": window.subject_id,
            "timestamp": window.start,
            "source": "mqtt_raw_wearable",
            "window_seconds": RAW_WINDOW_SECONDS,
            "sample_rates_hz": window.sample_rates_hz,
            **window.samples,
        }
        try:
            return self.ingest(complete_payload, transport="mqtt")
        except SensorValidationError as exc:
            self._reject("assembled raw window failed validation", exc.errors)
        except Exception:
            log.exception(
                "Raw MQTT completed window processing failed: device_id=%s start=%s end=%s "
                "counts=%s expected=%s",
                window.device_id,
                window.start.isoformat(),
                self._window_end(window).isoformat(),
                counts,
                expected,
            )
        return None

    @staticmethod
    def _window_complete(window: _SampleWindow) -> bool:
        return not IngestionPipeline._incomplete_modalities(window)

    @staticmethod
    def _incomplete_modalities(window: _SampleWindow) -> dict[str, str]:
        reasons = {}
        for name, rate in window.sample_rates_hz.items():
            expected = round(rate * RAW_WINDOW_SECONDS)
            actual = len(window.samples[name])
            if rate <= 0:
                reasons[name] = "no samples/rate observed"
            elif actual < expected:
                reasons[name] = f"{actual}/{expected} samples"
        return reasons

    @classmethod
    def _incomplete_reason(cls, window: _SampleWindow) -> str:
        return ", ".join(f"{name}: {reason}" for name, reason in cls._incomplete_modalities(window).items()) or "none"

    @staticmethod
    def _window_end(window: _SampleWindow) -> datetime:
        return window.start + timedelta(seconds=RAW_WINDOW_SECONDS)

    @staticmethod
    def _sample_counts(window: _SampleWindow) -> dict[str, int]:
        return {name: len(values) for name, values in window.samples.items()}

    @staticmethod
    def _expected_counts(window: _SampleWindow) -> dict[str, int]:
        return {name: round(rate * RAW_WINDOW_SECONDS) for name, rate in window.sample_rates_hz.items()}

    def _log_window_progress(self, window: _SampleWindow, modality: str, timestamp: datetime) -> None:
        """Emit bounded progress diagnostics for high-rate raw MQTT streams."""
        counts = self._sample_counts(window)
        expected = self._expected_counts(window)
        total = sum(counts.values())
        complete = self._window_complete(window)
        reached_threshold = any(expected[name] > 0 and counts[name] >= expected[name] for name in counts)
        if total - window.last_logged_sample_total < 500 and not reached_threshold:
            return
        window.last_logged_sample_total = total
        log.info(
            "Raw MQTT progress: device_id=%s modality=%s timestamp_window=%s..%s elapsed=%.3fs "
            "sample_rate_hz=%s counts=%s expected=%s completion_condition=%s incomplete=%s",
            window.device_id,
            modality,
            window.start.isoformat(),
            self._window_end(window).isoformat(),
            (timestamp - window.start).total_seconds(),
            window.sample_rates_hz,
            counts,
            expected,
            complete,
            self._incomplete_reason(window),
        )

    def live_telemetry(self, device_id: str | None = None) -> LiveTelemetry:
        """Latest accepted MQTT sample, independent of completed-window storage."""
        with self._live_lock:
            key = device_id or self._live_latest_device_id
            sample = self._live.get(key) if key else None
            snapshot = None if sample is None else _LiveSample(
                device_id=sample.device_id,
                subject_id=sample.subject_id,
                timestamp=sample.timestamp,
                received_at=sample.received_at,
                last_modality=sample.last_modality,
                bvp=sample.bvp,
                eda=sample.eda,
                skin_temperature=sample.skin_temperature,
                acc_x=sample.acc_x,
                acc_y=sample.acc_y,
                acc_z=sample.acc_z,
                sample_counts=dict(sample.sample_counts),
            )
        if snapshot is None:
            return LiveTelemetry(
                status="no_data",
                has_data=False,
                stale_after_seconds=LIVE_STALE_AFTER_SECONDS,
                sample_counts=dict(_EMPTY_COUNTS),
            )
        age_seconds = round((utcnow() - snapshot.received_at).total_seconds(), 3)
        return LiveTelemetry(
            status="stale" if age_seconds >= LIVE_STALE_AFTER_SECONDS else "receiving",
            has_data=True,
            device_id=snapshot.device_id,
            subject_id=snapshot.subject_id,
            timestamp=snapshot.timestamp,
            received_at=snapshot.received_at,
            age_seconds=age_seconds,
            stale_after_seconds=LIVE_STALE_AFTER_SECONDS,
            last_modality=snapshot.last_modality,
            bvp=snapshot.bvp,
            eda=snapshot.eda,
            skin_temperature=snapshot.skin_temperature,
            acc_x=snapshot.acc_x,
            acc_y=snapshot.acc_y,
            acc_z=snapshot.acc_z,
            sample_counts=snapshot.sample_counts,
        )

    def diagnostics(self) -> dict[str, Any]:
        """Return counters plus active raw-window state for operational diagnostics."""
        windows = {}
        for device_id, window in self._sample_windows.items():
            windows[device_id] = {
                "start": window.start.isoformat(),
                "end": self._window_end(window).isoformat(),
                "elapsed_seconds": round(
                    (max(window.last_timestamps.values()) - window.start).total_seconds(), 6
                ) if window.last_timestamps else 0.0,
                "counts": self._sample_counts(window),
                "expected": self._expected_counts(window),
                "incomplete": self._incomplete_modalities(window),
                "sample_rates_hz": dict(window.sample_rates_hz),
            }
        counters = self.stats.snapshot()
        return {**counters, "counters": counters, "active_windows": windows}

    def _update_live_telemetry(
        self,
        window: _SampleWindow,
        modality: str,
        timestamp: datetime,
        values: tuple[float, ...],
    ) -> None:
        with self._live_lock:
            current = self._live.get(window.device_id)
            if current is None:
                current = _LiveSample(
                    device_id=window.device_id,
                    subject_id=window.subject_id,
                    timestamp=timestamp,
                    received_at=utcnow(),
                    last_modality=modality,
                )
            current.subject_id = window.subject_id
            current.timestamp = timestamp
            current.received_at = utcnow()
            current.last_modality = modality
            current.sample_counts = self._sample_counts(window)
            if modality == "accelerometer":
                current.acc_x, current.acc_y, current.acc_z = values
            elif modality == "bvp":
                current.bvp = values[0]
            elif modality == "eda":
                current.eda = values[0]
            else:
                current.skin_temperature = values[0]
            self._live[window.device_id] = current
            self._live_latest_device_id = window.device_id

    def _reject(self, reason: str, errors: list[dict[str, str]] | None = None) -> None:
        self.stats.incr("rejected")
        log.warning("MQTT message rejected: %s%s", reason, f" ({errors})" if errors else "")
        return None
