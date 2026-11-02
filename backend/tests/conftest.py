import pytest
from fastapi.testclient import TestClient

from app.container import build_container
from app.core.config import Settings
from app.main import create_app


def make_settings(**overrides) -> Settings:
    base = dict(mqtt_enabled=False, inference_mode="mock", database_url="sqlite:///:memory:", app_env="test")
    base.update(overrides)
    return Settings(_env_file=None, **base)  # ignore any developer .env


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def container(settings):
    c = build_container(settings)
    yield c
    c.repo.close()


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture
def valid_payload() -> dict:
    return {
        "device_id": "dev-1",
        "user_id": "user-1",
        "session_id": "sess-1",
        "timestamp": "2026-01-01T12:00:00Z",
        "heart_rate": 72.0,
        "hrv": 45.0,
        "spo2": 97.0,
        "skin_temperature": 33.5,
        "accelerometer": {"x": 0.0, "y": 3.0, "z": 4.0},
    }
