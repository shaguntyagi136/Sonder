from datetime import datetime, timezone

import pytest

from app.core.errors import SensorValidationError
from app.schemas.sensor import parse_sensor_payload


def test_valid_payload(valid_payload):
    p = parse_sensor_payload(valid_payload)
    assert p.device_id == "dev-1"
    assert p.timestamp == datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    assert p.accelerometer.z == 4.0


def test_string_numbers_are_coerced(valid_payload):
    valid_payload["heart_rate"] = "72.5"
    assert parse_sensor_payload(valid_payload).heart_rate == 72.5


@pytest.mark.parametrize(
    "mutation",
    [
        {"device_id": None},  # required
        {"device_id": ""},
        {"device_id": "bad/id"},  # would corrupt MQTT topics
        {"device_id": "a+b"},
        {"heart_rate": float("nan")},
        {"heart_rate": float("inf")},
        {"heart_rate": -5},
        {"heart_rate": 0},
        {"heart_rate": 10_000},
        {"heart_rate": True},  # bool is not a number
        {"heart_rate": "fast"},
        {"spo2": 150},
        {"spo2": -1},
        {"hrv": -3},
        {"skin_temperature": 500},
        {"accelerometer": {"x": 1, "y": 2}},  # incomplete axes
        {"accelerometer": "up"},
        {"timestamp": "not a time"},
        {"timestamp": True},
        {"timestamp": [1]},
    ],
)
def test_invalid_values_rejected(valid_payload, mutation):
    payload = {**valid_payload, **mutation}
    if payload.get("device_id") is None:
        payload.pop("device_id")
    with pytest.raises(SensorValidationError) as exc:
        parse_sensor_payload(payload)
    assert exc.value.errors


def test_error_details_do_not_echo_input(valid_payload):
    valid_payload["heart_rate"] = "SECRET-LOOKING-VALUE"
    with pytest.raises(SensorValidationError) as exc:
        parse_sensor_payload(valid_payload)
    assert "SECRET-LOOKING-VALUE" not in str(exc.value.errors)


def test_no_measurements_rejected():
    with pytest.raises(SensorValidationError):
        parse_sensor_payload({"device_id": "d"})


@pytest.mark.parametrize("bad", [[1, 2], "text", 5, None])
def test_non_object_rejected(bad):
    with pytest.raises(SensorValidationError):
        parse_sensor_payload(bad)


def test_partial_measurements_allowed():
    assert parse_sensor_payload({"device_id": "d", "heart_rate": 60}).hrv is None


@pytest.mark.parametrize(
    "ts", [1_767_268_800, 1_767_268_800_000, "1767268800", "2026-01-01T12:00:00+00:00", "2026-01-01T12:00:00"]
)
def test_timestamp_formats(valid_payload, ts):
    valid_payload["timestamp"] = ts
    assert parse_sensor_payload(valid_payload).timestamp.tzinfo is not None


def test_epoch_seconds_and_millis_agree(valid_payload):
    a = parse_sensor_payload({**valid_payload, "timestamp": 1_767_268_800}).timestamp
    b = parse_sensor_payload({**valid_payload, "timestamp": 1_767_268_800_000}).timestamp
    assert a == b


def test_default_device_id_from_topic():
    p = parse_sensor_payload({"heart_rate": 60}, default_device_id="from-topic")
    assert p.device_id == "from-topic"


def test_payload_device_id_wins_over_topic():
    p = parse_sensor_payload({"device_id": "own", "heart_rate": 60}, default_device_id="from-topic")
    assert p.device_id == "own"


def test_unknown_fields_ignored(valid_payload):
    valid_payload["firmware"] = "1.2.3"
    parse_sensor_payload(valid_payload)
