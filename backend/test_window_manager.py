import json
import sys
import threading
from types import SimpleNamespace
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.mqtt_handler import MQTTConfig, MQTTHandler, validate_sensor_message
from app.window_manager import EXPECTED_COUNTS, WindowManager

BASE_TIMESTAMP = 1_798_000_020


class NoopPredictor:
    metadata = {}

    def predict(self, features):
        raise AssertionError("Noop predictor should not be called")


class AdvancingClock:
    def __init__(self, step=0.01):
        self.value = 0.0
        self.step = step

    def __call__(self):
        value = self.value
        self.value += self.step
        return value


def _sensor(device, modality, sequence, second=None):
    rate = {"bvp": 64, "eda": 4, "temperature": 4, "accelerometer": 32}[modality]
    sample_second = sequence / rate if second is None else second
    payload = {
        "device_id": device,
        "timestamp": datetime.fromtimestamp(
            BASE_TIMESTAMP + sample_second, timezone.utc
        ).isoformat().replace("+00:00", "Z"),
        "source": "test",
        "modality": modality,
        "sequence": sequence,
        "sample_rate_hz": {"bvp": 64, "eda": 4, "temperature": 4, "accelerometer": 32}[modality],
    }
    if modality == "accelerometer":
        payload.update(acc_x=0.1, acc_y=0.2, acc_z=0.9)
    else:
        payload["value"] = 1.0
    return payload


def _topic(device, modality):
    return f"stresssense/v1/devices/{device}/sensors/{modality}"


def test_sensor_validation_and_topic_matching():
    message = _sensor("device-a", "bvp", 0)
    assert validate_sensor_message(_topic("device-a", "bvp"), json.dumps(message)) == message
    with pytest.raises(ValueError, match="does not match"):
        validate_sensor_message(_topic("device-b", "bvp"), json.dumps(message))


def test_invalid_sensor_messages_are_rejected():
    message = _sensor("device-a", "eda", 0)
    message["value"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_sensor_message(_topic("device-a", "eda"), json.dumps(message))


def test_negative_sequence_is_rejected():
    message = _sensor("device-a", "eda", -1)
    with pytest.raises(ValueError, match="non-negative"):
        validate_sensor_message(_topic("device-a", "eda"), json.dumps(message))


def test_window_counts_duplicates_and_device_separation():
    statuses = []
    manager = WindowManager(
        predictor=NoopPredictor(),
        status_callback=statuses.append,
    )
    first = _sensor("device-a", "bvp", 0)
    assert manager.ingest(first)
    assert not manager.ingest(first)
    assert manager.ingest(_sensor("device-b", "bvp", 0))
    collecting = [event for event in statuses if event["stage"] == "COLLECTING_SAMPLES"]
    assert collecting[0]["received_sample_counts"]["bvp"] == 1
    assert {event["device_id"] for event in collecting} == {"device-a", "device-b"}


def test_incomplete_window_is_reported_when_next_window_arrives():
    statuses = []
    manager = WindowManager(
        predictor=NoopPredictor(),
        status_callback=statuses.append,
        timeout_seconds=0,
    )
    manager.ingest(_sensor("device-a", "eda", 0))
    manager.ingest(_sensor("device-a", "eda", 1, second=60))
    incomplete = [event for event in statuses if event["stage"] == "WINDOW_INCOMPLETE"]
    assert len(incomplete) == 1
    assert incomplete[0]["received_sample_counts"]["eda"] == 1


def test_complete_window_uses_integrated_predictor_pipeline():
    statuses = []
    predictions = []
    seen = {}

    class FakePredictor:
        metadata = {"model_version": "test"}

        def predict_window(self, completed_window):
            seen["completed_window"] = completed_window
            return {
                "prediction": 0,
                "label": "BASELINE",
                "stress_probability": 0.1,
                "baseline_probability": 0.9,
            }

    manager = WindowManager(
        predictor=FakePredictor(),
        status_callback=statuses.append,
        prediction_callback=predictions.append,
        clock=AdvancingClock(),
    )
    for modality, count in EXPECTED_COUNTS.items():
        for sequence in range(count):
            manager.ingest(_sensor("device-a", modality, sequence))

    manager.close()
    assert len(seen["completed_window"]["reading"]["bvp"]) == 3840
    assert len(seen["completed_window"]["reading"]["acc_x"]) == 1920
    assert predictions[0]["model_version"] == "test"
    assert [event["stage"] for event in statuses][-2:] == ["PREDICTION_READY", "WINDOW_RESET"]


def test_completed_window_prediction_failure_is_reported():
    statuses = []

    class FailingPredictor:
        metadata = {"model_version": "test"}

        def predict_window(self, completed_window):
            raise RuntimeError("prediction unavailable")

    manager = WindowManager(
        predictor=FailingPredictor(),
        status_callback=statuses.append,
        clock=AdvancingClock(),
    )
    for modality, count in EXPECTED_COUNTS.items():
        for sequence in range(count):
            manager.ingest(_sensor("device-a", modality, sequence))

    manager.close()
    error_events = [event for event in statuses if event["stage"] == "WINDOW_ERROR"]
    assert len(error_events) == 1
    assert "prediction unavailable" in error_events[0]["message"]


def test_duplicate_after_completed_window_is_rejected(monkeypatch):
    extracted = threading.Event()

    class FakePredictor:
        metadata = {"model_version": "test"}

        def predict(self, features):
            return {
                "prediction": 0,
                "label": "BASELINE",
                "stress_probability": 0.1,
                "baseline_probability": 0.9,
            }

    monkeypatch.setattr(
        "app.window_manager.extract_window_features",
        lambda reading: (extracted.set() or {f"feature_{i}": 0.0 for i in range(27)}),
    )
    manager = WindowManager(
        predictor=FakePredictor(),
        clock=AdvancingClock(step=0.02),
        timeout_seconds=10000,
    )
    for modality, count in EXPECTED_COUNTS.items():
        for sequence in range(count):
            manager.ingest(_sensor("device-a", modality, sequence))
    assert extracted.wait(2)
    assert not manager.ingest(_sensor("device-a", "bvp", 0))
    manager.close()


def test_out_of_order_samples_are_sorted_before_extraction(monkeypatch):
    manager = WindowManager(
        predictor=NoopPredictor(),
        timeout_seconds=1000,
    )
    first = _sensor("device-a", "eda", 1)
    second = _sensor("device-a", "eda", 0)
    assert manager.ingest(first)
    assert manager.ingest(second)
    window = next(iter(manager._windows["device-a"].values()))
    assert [sample.sequence for sample in sorted(window.samples["eda"], key=lambda item: item.timestamp)] == [0, 1]
    manager.close()


def test_instant_historical_window_is_not_processed(monkeypatch):
    extracted = []
    monkeypatch.setattr(
        "app.window_manager.extract_window_features",
        lambda reading: extracted.append(reading) or {},
    )
    manager = WindowManager(predictor=NoopPredictor(), clock=lambda: 1.0)
    for modality, count in EXPECTED_COUNTS.items():
        for sequence in range(count):
            assert manager.ingest(_sensor("device-a", modality, sequence))
    manager.close()
    assert not extracted


def test_genuine_collection_duration_is_processed(monkeypatch):
    extracted = threading.Event()
    monkeypatch.setattr(
        "app.window_manager.extract_window_features",
        lambda reading: extracted.set() or {f"feature_{i}": 0.0 for i in range(27)},
    )
    manager = WindowManager(
        predictor=NoopPredictor(),
        clock=AdvancingClock(step=0.01),
        timeout_seconds=10000,
    )
    for modality, count in EXPECTED_COUNTS.items():
        for sequence in range(count):
            manager.ingest(_sensor("device-a", modality, sequence))
    assert extracted.wait(2)
    manager.close()


def test_new_window_does_not_expire_late_active_window():
    statuses = []
    manager = WindowManager(
        predictor=NoopPredictor(),
        status_callback=statuses.append,
        timeout_seconds=1000,
    )
    old = _sensor("device-a", "eda", 0)
    assert manager.ingest(old)
    assert manager.ingest(_sensor("device-a", "eda", 0, second=60))
    assert manager.ingest(_sensor("device-a", "eda", 1))
    assert not [
        event for event in statuses if event["stage"] == "WINDOW_INCOMPLETE"
    ]
    manager.close()


def test_rejected_sensor_is_not_broadcast():
    forwarded = []

    class FakeManager:
        def ingest(self, message):
            return False

    handler = MQTTHandler(
        MQTTConfig("localhost", 8883, "user", "password", 60),
        sensor_callback=forwarded.append,
        window_manager=FakeManager(),
    )
    message = _sensor("device-a", "eda", 0)
    handler._on_message(
        None,
        None,
        SimpleNamespace(topic=_topic("device-a", "eda"), payload=json.dumps(message).encode()),
    )
    assert forwarded == []
    handler.window_manager.close = lambda: None


def test_duplicate_timestamps_and_invalid_spacing_are_rejected():
    manager = WindowManager(predictor=NoopPredictor())
    assert manager.ingest(_sensor("device-a", "eda", 0))
    duplicate_timestamp = _sensor("device-a", "eda", 1, second=0)
    assert not manager.ingest(duplicate_timestamp)
    invalid_spacing = _sensor("device-a", "eda", 2, second=10)
    assert not manager.ingest(invalid_spacing)
    manager.close()


def test_window_expires_without_a_new_message():
    current_time = [100.0]
    statuses = []
    manager = WindowManager(
        predictor=NoopPredictor(),
        status_callback=statuses.append,
        timeout_seconds=5,
        clock=lambda: current_time[0],
    )
    manager.ingest(_sensor("device-a", "eda", 0))
    current_time[0] = 106.0
    assert manager.expire_stalled_windows() == 1
    assert [event["stage"] for event in statuses if event["stage"] == "WINDOW_INCOMPLETE"]
    manager.close()


def test_two_devices_can_process_concurrently(monkeypatch):
    started = threading.Barrier(2)
    finished = threading.Event()
    processed = []

    class FakePredictor:
        metadata = {"model_version": "test"}

        def predict(self, features):
            started.wait(timeout=10)
            processed.append(True)
            if len(processed) == 2:
                finished.set()
            return {
                "prediction": 0,
                "label": "BASELINE",
                "stress_probability": 0.1,
                "baseline_probability": 0.9,
            }

    monkeypatch.setattr(
        "app.window_manager.extract_window_features",
        lambda reading: {f"feature_{i}": 0.0 for i in range(27)},
    )
    manager = WindowManager(
        predictor=FakePredictor(),
        clock=AdvancingClock(step=0.02),
    )
    for device in ("device-a", "device-b"):
        for modality, count in EXPECTED_COUNTS.items():
            for sequence in range(count):
                manager.ingest(_sensor(device, modality, sequence))
    assert finished.wait(10)
    manager.close()


def test_processing_exception_does_not_block_next_window(monkeypatch):
    calls = []

    class FakePredictor:
        metadata = {"model_version": "test"}

        def predict(self, features):
            calls.append(True)
            if len(calls) == 1:
                raise RuntimeError("expected test failure")
            return {
                "prediction": 0,
                "label": "BASELINE",
                "stress_probability": 0.1,
                "baseline_probability": 0.9,
            }

    monkeypatch.setattr(
        "app.window_manager.extract_window_features",
        lambda reading: {f"feature_{i}": 0.0 for i in range(27)},
    )
    manager = WindowManager(predictor=FakePredictor(), clock=AdvancingClock())
    for second in (0, 60):
        for modality, count in EXPECTED_COUNTS.items():
            for sequence in range(count):
                manager.ingest(_sensor("device-a", modality, sequence, second=second + sequence / {"bvp": 64, "eda": 4, "temperature": 4, "accelerometer": 32}[modality]))
    manager.close()
    assert len(calls) == 2
