"""Focused coverage for the active raw-window ingestion architecture."""
from __future__ import annotations

import json
import logging
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.container import build_container
from app.core.errors import InferenceError, StorageError
from tests.conftest import make_settings


def raw_window(device_id: str = "watch-7") -> dict:
    return {
        "device_id": device_id,
        "subject_id": "S2",
        "timestamp": "2026-01-01T12:00:00Z",
        "window_seconds": 60,
        "sample_rates_hz": {"bvp": 1, "eda": 1, "skin_temperature": 1, "acc_x": 1, "acc_y": 1, "acc_z": 1},
        "bvp": [0.0] * 60,
        "eda": [0.1] * 60,
        "skin_temperature": [33.0] * 60,
        "acc_x": [0.0] * 60,
        "acc_y": [0.0] * 60,
        "acc_z": [1.0] * 60,
    }


def test_full_window_is_featured_predicted_and_persisted(caplog):
    container = build_container(make_settings())
    try:
        with caplog.at_level(logging.INFO):
            outcome = container.pipeline.ingest(raw_window())
        assert outcome.status == "ok"
        assert outcome.reading.subject_id == "S2"
        assert outcome.prediction is not None and outcome.prediction.is_mock
        assert set(outcome.prediction.features) == set(container.inference.spec.feature_order)
        assert container.repo.latest_reading("watch-7").bvp == [0.0] * 60
        assert any("Prediction succeeded" in record.message for record in caplog.records)
    finally:
        container.repo.close()


def test_mqtt_progressive_samples_create_one_complete_window():
    container = build_container(make_settings(mqtt_topics="stress/sensor/+/data"))
    topic = "stress/sensor/watch-7/data"
    try:
        base = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        result = None
        for index in range(60):
            timestamp = (base.replace(second=index)).isoformat()
            for modality, fields in (
                ("bvp", {"value": 0.1}),
                ("eda", {"value": 0.2}),
                ("skin_temperature", {"value": 33.0}),
                ("accelerometer", {"acc_x": 0.0, "acc_y": 0.0, "acc_z": 1.0}),
            ):
                payload = {"subject_id": "S2", "timestamp": timestamp, "modality": modality, "sample_rate_hz": 1, **fields}
                result = container.pipeline.ingest_mqtt_message(topic, json.dumps(payload).encode()) or result
        assert result is not None and result.status == "ok"
        assert result.reading.device_id == "watch-7"
        assert result.reading.transport == "mqtt"
        assert len(result.reading.bvp) == len(result.reading.eda) == 60
    finally:
        container.repo.close()


def test_mqtt_incomplete_window_does_not_reach_completion_path():
    container = build_container(make_settings(mqtt_topics="stress/sensor/+/data"))
    topic = "stress/sensor/watch-7/data"
    try:
        for index in range(59):
            timestamp = f"2026-01-01T12:00:{index:02d}+00:00"
            for modality, fields in (
                ("bvp", {"value": 0.1}),
                ("eda", {"value": 0.2}),
                ("skin_temperature", {"value": 33.0}),
                ("accelerometer", {"acc_x": 0.0, "acc_y": 0.0, "acc_z": 1.0}),
            ):
                payload = {"subject_id": "S2", "timestamp": timestamp, "modality": modality, "sample_rate_hz": 1, **fields}
                assert container.pipeline.ingest_mqtt_message(topic, json.dumps(payload).encode()) is None
        assert container.repo.latest_reading("watch-7") is None
    finally:
        container.repo.close()


def test_two_consecutive_windows_are_processed_once_each():
    container = build_container(make_settings(mqtt_topics="stress/sensor/+/data"))
    topic = "stress/sensor/watch-7/data"
    try:
        first = _send_window(container, topic, datetime(2026, 1, 1, 12, tzinfo=timezone.utc))
        second = _send_window(container, topic, datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc), value=1.0)
        assert first is not None and second is not None
        assert container.repo.list_predictions(limit=10, offset=0)[1] == 2
        assert container.pipeline.stats.snapshot()["predicted"] == 2
    finally:
        container.repo.close()


def test_actual_simulator_rates_complete_and_persist_one_window():
    simulator_path = Path(__file__).parents[1] / "scripts" / "simulate_wesad.py"
    spec = importlib.util.spec_from_file_location("simulate_wesad", simulator_path)
    simulator = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(simulator)

    archive = Path(__file__).parents[1] / "data" / "archive (8).zip"
    from app.preprocessing.wesad import load_wesad_participant

    window = next(load_wesad_participant(
        archive,
        "S2",
        window_seconds=60.0,
        include_original_labels=False,
        max_windows=1,
    )).model_copy(update={"device_id": "wesad-s2"})
    container = build_container(make_settings(mqtt_topics="stress/sensor/+/data"))
    topic = "stress/sensor/wesad-s2/data"
    outcome = None
    try:
        for _, payload in simulator._raw_messages(window):
            outcome = container.pipeline.ingest_mqtt_message(
                topic, json.dumps(payload).encode()
            ) or outcome
        assert outcome is not None and outcome.status == "ok"
        assert outcome.prediction is not None
        assert container.repo.list_predictions(limit=10, offset=0)[1] == 1
        counters = container.pipeline.diagnostics()["counters"]
        assert counters["raw_mqtt_received"] == 6240
        assert counters["raw_samples_accepted"] == 6240
        assert counters["raw_windows_completed"] == 1
        assert counters["inference_succeeded"] == 1
        assert counters["persistence_succeeded"] == 1
    finally:
        container.repo.close()


def test_duplicate_and_out_of_order_samples_are_rejected(caplog):
    container = build_container(make_settings(mqtt_topics="stress/sensor/+/data"))
    topic = "stress/sensor/watch-7/data"
    try:
        with caplog.at_level(logging.WARNING):
            assert container.pipeline.ingest_mqtt_message(
                topic, _sample("bvp", 0.1, "2026-01-01T12:00:01+00:00")
            ) is None
            assert container.pipeline.ingest_mqtt_message(
                topic, _sample("bvp", 0.2, "2026-01-01T12:00:01+00:00")
            ) is None
            assert container.pipeline.ingest_mqtt_message(
                topic, _sample("bvp", 0.3, "2026-01-01T12:00:00+00:00")
            ) is None
        assert any("duplicate or out-of-order timestamp" in record.message for record in caplog.records)
        assert container.pipeline.live_telemetry().bvp == 0.1
    finally:
        container.repo.close()


def test_boundary_sample_starts_next_window_without_duplication():
    container = build_container(make_settings(mqtt_topics="stress/sensor/+/data"))
    topic = "stress/sensor/watch-7/data"
    try:
        assert container.pipeline.ingest_mqtt_message(
            topic, _sample("bvp", 0.1, "2026-01-01T12:00:00+00:00")
        ) is None
        assert container.pipeline.ingest_mqtt_message(
            topic, _sample("bvp", 0.2, "2026-01-01T12:01:00+00:00")
        ) is None
        window = container.pipeline._sample_windows["watch-7"]
        assert window.start.isoformat() == "2026-01-01T12:01:00+00:00"
        assert window.samples["bvp"] == [0.2]
    finally:
        container.repo.close()


def test_inference_failure_does_not_block_next_window(caplog, monkeypatch):
    container = build_container(make_settings(mqtt_topics="stress/sensor/+/data"))
    topic = "stress/sensor/watch-7/data"
    original_predict = container.inference.predict
    calls = 0

    def fail_once(values):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise InferenceError("test inference failure")
        return original_predict(values)

    monkeypatch.setattr(container.inference, "predict", fail_once)
    try:
        with caplog.at_level(logging.ERROR):
            first = _send_window(container, topic, datetime(2026, 1, 1, 12, tzinfo=timezone.utc))
        second = _send_window(container, topic, datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc))
        assert first is not None and first.status == "inference_failed"
        assert second is not None and second.prediction is not None
        assert any("Window inference failed" in record.message for record in caplog.records)
    finally:
        container.repo.close()


def test_persistence_failure_does_not_block_next_window(monkeypatch):
    container = build_container(make_settings(mqtt_topics="stress/sensor/+/data"))
    topic = "stress/sensor/watch-7/data"
    original_save = container.repo.save
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise StorageError("test persistence failure")
        return original_save(*args, **kwargs)

    monkeypatch.setattr(container.repo, "save", fail_once)
    try:
        first = _send_window(container, topic, datetime(2026, 1, 1, 12, tzinfo=timezone.utc))
        second = _send_window(container, topic, datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc))
        assert first is None
        assert second is not None and second.prediction is not None
    finally:
        container.repo.close()


def _sample(modality: str, value, timestamp: str, **extra) -> bytes:
    payload = {
        "subject_id": "S2",
        "timestamp": timestamp,
        "modality": modality,
        "sample_rate_hz": 1,
        **extra,
    }
    if modality == "accelerometer":
        payload.update(value)
    else:
        payload["value"] = value
    return json.dumps(payload).encode()


def _send_window(container, topic: str, start: datetime, value: float = 0.1):
    result = None
    for index in range(60):
        timestamp = (start + timedelta(seconds=index)).isoformat()
        for modality, fields in (
            ("bvp", {"value": value + index}),
            ("eda", {"value": 0.2}),
            ("skin_temperature", {"value": 33.0}),
            ("accelerometer", {"acc_x": 0.0, "acc_y": 0.0, "acc_z": 1.0}),
        ):
            payload = {"subject_id": "S2", "timestamp": timestamp, "modality": modality, "sample_rate_hz": 1, **fields}
            result = container.pipeline.ingest_mqtt_message(topic, json.dumps(payload).encode()) or result
    return result


def test_mqtt_sample_updates_live_telemetry_before_window_completes():
    container = build_container(make_settings(mqtt_topics="stress/sensor/+/data"))
    topic = "stress/sensor/watch-7/data"
    try:
        empty = container.pipeline.live_telemetry()
        assert empty.status == "no_data" and empty.has_data is False
        assert container.repo.latest_reading() is None

        outcome = container.pipeline.ingest_mqtt_message(
            topic, _sample("bvp", 0.42, "2026-01-01T12:00:00+00:00")
        )
        assert outcome is None
        assert container.repo.latest_reading() is None

        live = container.pipeline.live_telemetry()
        assert live.status == "receiving" and live.has_data is True
        assert live.device_id == "watch-7" and live.subject_id == "S2"
        assert live.bvp == 0.42
        assert live.eda is None and live.skin_temperature is None
        assert live.acc_x is None and live.acc_y is None and live.acc_z is None
        assert live.last_modality == "bvp"
        assert live.sample_counts["bvp"] == 1
        assert live.received_at is not None and live.timestamp is not None
        assert live.age_seconds is not None and live.age_seconds < live.stale_after_seconds

        container.pipeline.ingest_mqtt_message(
            topic, _sample("eda", 0.21, "2026-01-01T12:00:00+00:00")
        )
        container.pipeline.ingest_mqtt_message(
            topic,
            _sample("skin_temperature", 33.1, "2026-01-01T12:00:00+00:00"),
        )
        container.pipeline.ingest_mqtt_message(
            topic,
            _sample("accelerometer", {"acc_x": 0.01, "acc_y": 0.02, "acc_z": 0.98}, "2026-01-01T12:00:00+00:00"),
        )
        live = container.pipeline.live_telemetry()
        assert live.bvp == 0.42 and live.eda == 0.21 and live.skin_temperature == 33.1
        assert live.acc_x == 0.01 and live.acc_y == 0.02 and live.acc_z == 0.98
        assert live.last_modality == "accelerometer"
        assert container.repo.latest_reading() is None
        missing = container.pipeline.live_telemetry("missing-device")
        assert missing.status == "no_data"
    finally:
        container.repo.close()


def test_live_telemetry_keeps_latest_real_sample_and_completed_window_separate():
    container = build_container(make_settings(mqtt_topics="stress/sensor/+/data"))
    topic = "stress/sensor/watch-7/data"
    try:
        base = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        result = None
        for index in range(60):
            timestamp = (base.replace(second=index)).isoformat()
            for modality, fields in (
                ("bvp", {"value": 0.1 + index}),
                ("eda", {"value": 0.2}),
                ("skin_temperature", {"value": 33.0}),
                ("accelerometer", {"acc_x": 0.0, "acc_y": 0.0, "acc_z": 1.0}),
            ):
                payload = {"subject_id": "S2", "timestamp": timestamp, "modality": modality, "sample_rate_hz": 1, **fields}
                result = container.pipeline.ingest_mqtt_message(topic, json.dumps(payload).encode()) or result
        assert result is not None and result.status == "ok"
        stored = container.repo.latest_reading("watch-7")
        assert stored is not None and stored.bvp[0] == 0.1 and stored.bvp[-1] == 0.1 + 59
        live = container.pipeline.live_telemetry()
        assert live.bvp == 0.1 + 59
        assert live.has_data is True
    finally:
        container.repo.close()


def test_rejected_mqtt_sample_does_not_create_live_data():
    container = build_container(make_settings(mqtt_topics="stress/sensor/+/data"))
    try:
        container.pipeline.ingest_mqtt_message("stress/sensor/watch-7/data", b"not-json")
        live = container.pipeline.live_telemetry()
        assert live.status == "no_data" and live.has_data is False
        assert container.repo.latest_reading() is None
    finally:
        container.repo.close()


def test_live_telemetry_becomes_stale_from_reception_age(monkeypatch):
    from datetime import timedelta

    from app.core.timeutil import utcnow
    from app.services import pipeline as pipeline_module

    container = build_container(make_settings(mqtt_topics="stress/sensor/+/data"))
    topic = "stress/sensor/watch-7/data"
    try:
        container.pipeline.ingest_mqtt_message(topic, _sample("bvp", 1.5, "2026-01-01T12:00:00+00:00"))
        received_at = container.pipeline.live_telemetry().received_at
        monkeypatch.setattr(pipeline_module, "utcnow", lambda: received_at + timedelta(seconds=5))
        stale = container.pipeline.live_telemetry()
        assert stale.status == "stale" and stale.has_data is True
        assert stale.bvp == 1.5
        assert stale.age_seconds >= stale.stale_after_seconds
        monkeypatch.setattr(pipeline_module, "utcnow", utcnow)
    finally:
        container.repo.close()
