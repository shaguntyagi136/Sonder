from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import main


class FakeMQTTHandler:
    connected = False
    last_error = None

    def __init__(self, config, **callbacks):
        self.prediction_callback = callbacks["prediction_callback"]

    def start(self):
        return None

    def stop(self):
        return None

    def publish_prediction(self, prediction):
        return None


@pytest.fixture
def api_client(monkeypatch):
    monkeypatch.setattr(main, "MQTTConfig", type(
        "FakeMQTTConfig",
        (),
        {"from_env": classmethod(lambda cls: object())},
    ))
    monkeypatch.setattr(main, "MQTTHandler", FakeMQTTHandler)
    with TestClient(main.app) as client:
        yield client


def test_latest_prediction_returns_model_result(api_client):
    prediction = {
        "device_id": "device-01",
        "window_id": "device-01-0",
        "prediction": 1,
        "label": "STRESS",
        "confidence": 0.94,
    }
    main.mqtt_handler.prediction_callback(prediction)

    response = api_client.get("/api/predictions/latest")

    assert response.status_code == 200
    assert response.json() == prediction


def test_latest_prediction_returns_empty_result_before_first_prediction(api_client):
    response = api_client.get("/api/predictions/latest")

    assert response.status_code == 200
    assert response.json() == {"prediction": None}
