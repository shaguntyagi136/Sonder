import pytest


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["database"]["ok"] is True
    assert body["mqtt"]["enabled"] is False
    assert body["inference"]["is_mock"] is True
    assert body["pipeline"]["counters"]["raw_mqtt_received"] == 0
    assert body["pipeline"]["active_windows"] == {}
    assert "password" not in r.text.lower()


def test_health_degraded_when_model_missing(tmp_path):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.conftest import make_settings

    s = make_settings(inference_mode="model", model_path=str(tmp_path / "x"), model_metadata_path=str(tmp_path / "y"))
    with TestClient(create_app(s)) as c:
        body = c.get("/health").json()
        assert body["status"] == "degraded" and body["inference"]["model_loaded"] is False


def test_empty_state_returns_404(client):
    assert client.get("/api/sensor/latest").status_code == 404
    assert client.get("/api/predictions/latest").status_code == 404
    page = client.get("/api/predictions").json()
    assert page["items"] == [] and page["total"] == 0


def test_live_telemetry_empty_state_is_explicit(client):
    response = client.get("/api/sensor/live")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "no_data" and body["has_data"] is False
    assert body["bvp"] is None and body["eda"] is None
    assert body["skin_temperature"] is None
    assert body["acc_x"] is None and body["acc_y"] is None and body["acc_z"] is None
    assert body["sample_counts"] == {
        "bvp": 0, "eda": 0, "skin_temperature": 0, "acc_x": 0, "acc_y": 0, "acc_z": 0,
    }


def test_live_endpoint_updates_from_mqtt_without_completed_window(client):
    import json

    topic = "stress/sensor/watch-7/data"
    payload = {
        "subject_id": "S2",
        "timestamp": "2026-01-01T12:00:00+00:00",
        "modality": "bvp",
        "sample_rate_hz": 1,
        "value": 0.77,
    }
    outcome = client.app.state.container.pipeline.ingest_mqtt_message(topic, json.dumps(payload).encode())
    assert outcome is None
    assert client.get("/api/sensor/latest").status_code == 404
    live = client.get("/api/sensor/live").json()
    assert live["status"] == "receiving"
    assert live["device_id"] == "watch-7"
    assert live["bvp"] == 0.77
    assert live["has_data"] is True


def test_post_sensor_data_full_flow(client, valid_payload):
    r = client.post("/api/sensor/data", json=valid_payload)
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "ok" and body["reading"]["id"] == 1
    assert body["reading"]["transport"] == "http"
    assert body["prediction"]["is_mock"] is True

    latest = client.get("/api/sensor/latest").json()
    assert latest["device_id"] == "dev-1" and latest["heart_rate"] == 72.0
    pred = client.get("/api/predictions/latest").json()
    assert pred["reading_id"] == 1 and list(pred["features"]) == [
        "heart_rate", "hrv", "spo2", "skin_temperature", "accel_magnitude"]
    assert client.get("/api/predictions").json()["total"] == 1


@pytest.mark.parametrize(
    "bad", [{"device_id": "d", "heart_rate": -1}, {"heart_rate": 70}, {"device_id": "d"}, {"device_id": "d", "spo2": "x"}]
)
def test_post_invalid_payload_422_and_not_stored(client, bad):
    r = client.post("/api/sensor/data", json=bad)
    assert r.status_code == 422
    assert client.get("/api/sensor/latest").status_code == 404


def test_post_non_json_or_non_object(client):
    assert client.post("/api/sensor/data", content=b"{nope", headers={"content-type": "application/json"}).status_code == 422
    assert client.post("/api/sensor/data", json=[1, 2]).status_code == 422


def test_incomplete_features_stores_reading_without_prediction(client):
    r = client.post("/api/sensor/data", json={"device_id": "d", "heart_rate": 70})
    body = r.json()
    assert r.status_code == 201 and body["status"] == "preprocessing_failed"
    assert body["prediction"] is None and "missing" in body["error"]
    assert client.get("/api/sensor/latest").status_code == 200
    assert client.get("/api/predictions/latest").status_code == 404


def test_predictions_pagination_and_filter(client, valid_payload):
    for i, dev in enumerate(["a", "b", "a"]):
        client.post("/api/sensor/data", json={**valid_payload, "device_id": dev, "timestamp": 1_767_268_800 + i})
    assert client.get("/api/predictions?limit=2").json()["total"] == 3
    assert len(client.get("/api/predictions?limit=2").json()["items"]) == 2
    assert client.get("/api/predictions?device_id=a").json()["total"] == 2
    assert client.get("/api/predictions?limit=0").status_code == 422


def test_summary(client, valid_payload):
    from app.core.timeutil import utcnow

    now_iso = utcnow().isoformat()
    client.post("/api/sensor/data", json={**valid_payload, "timestamp": now_iso, "source": "simulator"})
    client.post("/api/sensor/data", json={**valid_payload, "timestamp": now_iso, "heart_rate": 80})
    client.post("/api/sensor/data", json={"device_id": "dev-1", "timestamp": now_iso, "heart_rate": 1})  # no prediction
    s = client.get("/api/summary?window_minutes=60").json()
    assert s["readings_in_window"] == 3 and s["predictions_in_window"] == 2
    assert s["prediction_status_counts"] == {"ok": 2, "preprocessing_failed": 1}
    assert s["reading_sources"] == {"simulator": 1, "unspecified": 2}
    assert s["includes_mock_predictions"] is True
    assert s["devices"] == ["dev-1"] and s["latest_prediction"] is not None
    assert s["feature_averages"]["heart_rate"] == pytest.approx((72 + 80 + 1) / 3)


def test_summary_excludes_old_data(client, valid_payload):
    client.post("/api/sensor/data", json={**valid_payload, "timestamp": "2020-01-01T00:00:00Z"})
    s = client.get("/api/summary?window_minutes=60").json()
    assert s["readings_in_window"] == 0 and s["readings_total"] == 1


def test_storage_failure_returns_503_without_details(client, valid_payload, monkeypatch):
    from app.core.errors import StorageError

    def boom(*a, **k):
        raise StorageError("db exploded /secret/path")

    monkeypatch.setattr(client.app.state.container.repo, "save", boom)
    r = client.post("/api/sensor/data", json=valid_payload)
    assert r.status_code == 503 and "secret" not in r.text
