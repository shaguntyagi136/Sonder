"""MQTT handling without a broker: message decoding, the malformed-payload path, subscriber callbacks."""
import json
import logging
from types import SimpleNamespace

import pytest

from app.core.errors import MalformedMessageError
from app.mqtt.parsing import decode_message, device_id_from_topic
from app.mqtt.subscriber import MqttSubscriber
from tests.conftest import make_settings

TOPIC = "stress/sensor/watch-7/data"


def msg(payload: bytes, topic: str = TOPIC):
    return SimpleNamespace(topic=topic, payload=payload)


def good(**over) -> bytes:
    body = {"heart_rate": 70, "hrv": 40, "spo2": 97, "skin_temperature": 33, "accelerometer": {"x": 0, "y": 0, "z": 1}}
    body.update(over)
    return json.dumps(body).encode()


@pytest.mark.parametrize(
    "raw", [b"", b"not json", b"\xff\xfe\x00", b"[1,2,3]", b'"str"', b"123", b"null", b"{" * 100000, b"x" * 20000]
)
def test_decode_rejects_malformed(raw):
    with pytest.raises(MalformedMessageError):
        decode_message(raw, 16384)


def test_decode_accepts_object():
    assert decode_message(b'{"a": 1}', 16384) == {"a": 1}


def test_device_id_from_topic():
    subs = ["stress/sensor/+/data"]
    assert device_id_from_topic(TOPIC, subs) == "watch-7"
    assert device_id_from_topic("other/topic", subs) is None
    assert device_id_from_topic("stress/sensor/x/data", ["nothing/+"]) is None


@pytest.mark.parametrize(
    "raw",
    [b"", b"garbage", b"\xff\xfe", b"[]", b"{}", good(heart_rate="NaN"), good(spo2=1000), b'{"device_id": "a/b", "heart_rate": 1}'],
)
def test_malformed_or_invalid_messages_are_dropped_safely(container, raw, caplog):
    with caplog.at_level(logging.WARNING):
        assert container.pipeline.ingest_mqtt_message(TOPIC, raw) is None  # must not raise
    assert container.repo.latest_reading() is None  # nothing stored
    stats = container.pipeline.stats.snapshot()
    assert stats["rejected"] == 1 and stats["stored"] == 0
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_valid_message_flows_through_and_takes_device_from_topic(container):
    out = container.pipeline.ingest_mqtt_message(TOPIC, good())
    assert out is not None and out.status == "ok"
    assert out.reading.device_id == "watch-7" and out.reading.transport == "mqtt"
    assert container.repo.latest_prediction().device_id == "watch-7"


def test_subscriber_callback_never_raises_and_never_stores_bad_data(container):
    sub = MqttSubscriber(make_settings(), container.pipeline.ingest_mqtt_message)
    for raw in (b"", b"junk", b"\x00\x01", b"[1]"):
        sub._on_message(None, None, msg(raw))  # would kill paho's network thread if it raised
    assert container.repo.latest_reading() is None
    sub._on_message(None, None, msg(good()))
    assert container.repo.latest_reading() is not None


def test_subscriber_survives_crashing_handler():
    def crash(topic, payload):
        raise RuntimeError("boom")

    MqttSubscriber(make_settings(), crash)._on_message(None, None, msg(b"{}"))


def test_pipeline_survives_unexpected_exception(container, monkeypatch):
    monkeypatch.setattr(container.pipeline, "ingest", lambda *a, **k: 1 / 0)
    assert container.pipeline.ingest_mqtt_message(TOPIC, good()) is None
    assert container.pipeline.stats.snapshot()["unexpected_errors"] == 1


def test_subscribes_to_all_configured_topics_on_connect():
    s = make_settings(mqtt_topics="a/+/x, b/#", mqtt_qos=1)
    sub = MqttSubscriber(s, lambda t, p: None)
    calls = []
    client = SimpleNamespace(subscribe=lambda topics: calls.extend(topics) or (0, 7))
    sub._on_connect(client, None, None, SimpleNamespace(is_failure=False), None)
    assert calls == [("a/+/x", 1), ("b/#", 1)] and sub.connected
    sub._on_disconnect(client, None, None, SimpleNamespace(is_failure=True), None)
    assert not sub.connected


def test_suback_marks_the_configured_topics_active(caplog):
    sub = MqttSubscriber(make_settings(mqtt_topics="stress/sensor/+/data"), lambda t, p: None)
    client = SimpleNamespace(subscribe=lambda topics: (0, 41))
    with caplog.at_level(logging.INFO):
        sub._on_connect(client, None, None, SimpleNamespace(is_failure=False), None)
        sub._on_subscribe(client, None, 41, [SimpleNamespace(is_failure=False, value=1)], None)
    assert sub.subscribed_topics == ("stress/sensor/+/data",)
    assert sub.subscription_acknowledged
    assert "MQTT SUBACK received" in caplog.text


def test_published_raw_payload_reaches_pipeline_callback(container):
    settings = make_settings(mqtt_enabled=True)
    subscriber = MqttSubscriber(settings, container.pipeline.ingest_mqtt_message)
    payload = {
        "subject_id": "S2",
        "timestamp": "2026-01-01T12:00:00Z",
        "modality": "bvp",
        "sample_rate_hz": 64,
        "value": 0.25,
    }
    subscriber._on_message(None, None, msg(json.dumps(payload).encode()))
    assert subscriber.received_messages == 1
    assert container.pipeline.live_telemetry("watch-7").bvp == 0.25
    assert container.pipeline.stats.snapshot()["raw_mqtt_received"] == 1


def test_failed_connect_does_not_subscribe():
    sub = MqttSubscriber(make_settings(), lambda t, p: None)
    calls = []
    sub._on_connect(SimpleNamespace(subscribe=lambda *a, **k: calls.append(a)), None, None, SimpleNamespace(is_failure=True), None)
    assert calls == [] and not sub.connected


def test_password_never_logged(caplog):
    s = make_settings(mqtt_username="u", mqtt_password="hunter2-secret")
    with caplog.at_level(logging.DEBUG):
        sub = MqttSubscriber(s, lambda t, p: None)
        sub.start()
        sub.stop()
    assert "hunter2-secret" not in caplog.text
    assert "hunter2-secret" not in str(s.safe_dump()) and "hunter2-secret" not in repr(s)
