"""Focused credential-independent tests for the Phase 2 sensor pipeline."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.feature_extractor import extract_window_features
from app.mqtt_handler import validate_sensor_message
from app.window_manager import EXPECTED_COUNTS, WindowManager, parse_timestamp


BASE_TIMESTAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)
RATES = {"bvp": 64, "eda": 4, "temperature": 4, "accelerometer": 32}


class AdvancingClock:
    def __init__(self, step: float = 0.01) -> None:
        self.value = 0.0
        self.step = step

    def __call__(self) -> float:
        value = self.value
        self.value += self.step
        return value


class FakePredictor:
    metadata = {"model_version": "pipeline-test"}

    def predict(self, features: dict) -> dict:
        assert len(features) == 27
        return {
            "prediction": 0,
            "label": "BASELINE",
            "stress_probability": 0.1,
            "baseline_probability": 0.9,
        }


def sensor(device_id: str, modality: str, sequence: int, *, timestamp: datetime | None = None) -> dict:
    timestamp = timestamp or BASE_TIMESTAMP + timedelta(seconds=sequence / RATES[modality])
    payload = {
        "device_id": device_id,
        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
        "modality": modality,
        "sequence": sequence,
        "sample_rate_hz": RATES[modality],
    }
    if modality == "accelerometer":
        payload.update(acc_x=0.1, acc_y=-0.2, acc_z=0.9)
    else:
        payload["value"] = {"bvp": 0.2, "eda": 0.5, "temperature": 33.0}[modality]
    return payload


def topic(message: dict) -> str:
    return f"stresssense/v1/devices/{message['device_id']}/sensors/{message['modality']}"


def test_documented_schema_accepts_all_modalities_and_rejects_bad_identifiers():
    for modality in RATES:
        message = sensor("device_01.a", modality, 0)
        assert validate_sensor_message(topic(message), json.dumps(message)) == message

    invalid_device = sensor("device with spaces", "eda", 0)
    with pytest.raises(ValueError, match="device_id"):
        validate_sensor_message(topic(invalid_device), json.dumps(invalid_device))
    naive_timestamp = sensor("device-01", "eda", 0)
    naive_timestamp["timestamp"] = "2026-01-01T00:00:00"
    with pytest.raises(ValueError, match="timezone"):
        validate_sensor_message(topic(naive_timestamp), json.dumps(naive_timestamp))
    boolean_value = sensor("device-01", "bvp", 0)
    boolean_value["value"] = True
    with pytest.raises(ValueError, match="finite"):
        validate_sensor_message(topic(boolean_value), json.dumps(boolean_value))


def test_timestamp_normalisation_ordering_and_modality_isolation():
    assert parse_timestamp("2026-01-01T05:30:00+05:30") == BASE_TIMESTAMP
    manager = WindowManager(predictor=FakePredictor(), timeout_seconds=1_000)
    assert manager.ingest(sensor("device-a", "eda", 1))
    assert manager.ingest(sensor("device-a", "eda", 0))
    assert not manager.ingest(sensor("device-a", "eda", 0))
    assert manager.ingest(sensor("device-a", "bvp", 0))
    assert manager.ingest(sensor("device-b", "eda", 0))
    a_window = next(iter(manager._windows["device-a"].values()))
    assert [sample.sequence for sample in a_window.samples["eda"]] == [0, 1]
    assert len(a_window.samples["bvp"]) == 1
    assert len(manager._windows["device-b"]) == 1
    manager.close()


def test_window_duration_is_configurable_and_counts_follow_sensor_rates():
    manager = WindowManager(predictor=FakePredictor(), window_seconds=30)
    assert manager.expected_counts == {
        "bvp": 1920,
        "eda": 120,
        "temperature": 120,
        "accelerometer": 960,
    }
    manager.close()
    with pytest.raises(ValueError, match="window_seconds"):
        WindowManager(predictor=FakePredictor(), window_seconds=0)


def test_missing_sample_expires_as_incomplete_without_fabrication():
    clock = [0.0]
    statuses: list[dict] = []
    manager = WindowManager(
        predictor=FakePredictor(),
        status_callback=statuses.append,
        timeout_seconds=5,
        clock=lambda: clock[0],
    )
    for modality, count in EXPECTED_COUNTS.items():
        limit = count - 1 if modality == "eda" else count
        for sequence in range(limit):
            assert manager.ingest(sensor("device-01", modality, sequence))
    clock[0] = 6.0
    assert manager.expire_stalled_windows() == 1
    incomplete = [event for event in statuses if event["stage"] == "WINDOW_INCOMPLETE"]
    assert incomplete[0]["received_sample_counts"]["eda"] == EXPECTED_COUNTS["eda"] - 1
    manager.close()


def test_complete_window_handoff_is_exactly_once_and_feature_compatible():
    completed: list[dict] = []
    predictions: list[dict] = []
    manager = WindowManager(
        predictor=FakePredictor(),
        window_callback=completed.append,
        prediction_callback=predictions.append,
        clock=AdvancingClock(),
        timeout_seconds=10_000,
    )
    for modality, count in EXPECTED_COUNTS.items():
        for sequence in range(count):
            assert manager.ingest(sensor("device-01", modality, sequence))

    assert len(completed) == 1
    handoff = completed[0]
    assert handoff["sample_counts"] == EXPECTED_COUNTS
    assert handoff["window_duration_seconds"] == 60
    assert {name: len(values) for name, values in handoff["reading"].items()} == {
        "bvp": 3840,
        "eda": 240,
        "skin_temperature": 240,
        "acc_x": 1920,
        "acc_y": 1920,
        "acc_z": 1920,
    }
    assert len(extract_window_features(handoff["reading"])) == 27
    assert not manager.ingest(sensor("device-01", "bvp", 0))
    manager.close()
    assert len(predictions) == 1
