"""Configuration. Everything comes from environment variables or backend/.env - nothing is hard-coded."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ directory. Relative paths in settings are resolved against this, never against the CWD.
BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        protected_namespaces=(),  # we legitimately have fields named model_*
    )

    # --- general ---
    app_env: str = "development"
    app_debug: bool = False
    log_level: str = "INFO"

    # --- API ---
    api_host: str = "127.0.0.1"
    api_port: int = Field(8000, ge=1, le=65535)
    cors_origins: str = "http://localhost:5500,http://127.0.0.1:5500,http://localhost:3000,http://localhost:5173" # comma separated

    # --- storage ---
    database_url: str = "sqlite:///./data/stress.db"

    # --- ML ---
    inference_mode: Literal["model", "mock"] = "model"

    model_path: str = "models/stress_model.joblib"
    model_metadata_path: str = "models/stress_model.metadata.json"

    missing_feature_policy: Literal["reject", "impute"] = "reject"

    # --- wearable-window preprocessing ---
    # Raw signals are processed as timestamped windows.  Dataset loaders may override
    # these values per record through SensorPayload.sample_rates_hz/window_seconds.
    default_window_seconds: float = Field(60.0, gt=0, le=3600)
    default_bvp_sample_rate_hz: float = Field(64.0, gt=0, le=2048)
    default_eda_sample_rate_hz: float = Field(4.0, gt=0, le=512)
    default_temperature_sample_rate_hz: float = Field(4.0, gt=0, le=512)
    default_accelerometer_sample_rate_hz: float = Field(32.0, gt=0, le=2048)

    # --- MQTT ---
    mqtt_enabled: bool = True
    mqtt_broker_host: str = ""
    mqtt_broker_port: int = 8883
    mqtt_username: Optional[str] = None
    mqtt_password: Optional[SecretStr] = None
    mqtt_use_tls: bool = True
    mqtt_client_id: str = "stress-backend"
    mqtt_topics: str = "stress/sensor/+/data"
    mqtt_qos: int = Field(1, ge=0, le=2)
    mqtt_keepalive: int = Field(60, ge=5)
    mqtt_max_payload_bytes: int = Field(16384, ge=64)



    @field_validator("mqtt_username", "mqtt_password", mode="before")
    @classmethod
    def _blank_is_none(cls, v: Any) -> Any:
        return None if v is None or (isinstance(v, str) and not v.strip()) else v

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @field_validator("database_url")
    @classmethod
    def _supported_db(cls, v: str) -> str:
        if not v.startswith("sqlite:///"):
            raise ValueError(
                "Only sqlite:/// URLs are implemented. To use another database, implement "
                "app.database.base.Repository and register it in app.database.create_repository()."
            )
        return v

    # --- helpers ---
    @property
    def mqtt_topic_list(self) -> list[str]:
        return [t.strip() for t in self.mqtt_topics.split(",") if t.strip()]

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def sqlite_target(self) -> str:
        target = self.database_url[len("sqlite:///"):]
        if target in ("", ":memory:"):
            return ":memory:"
        return str(resolve_path(target))

    def safe_dump(self) -> dict[str, Any]:
        """Settings suitable for logging: secrets are masked."""
        data = self.model_dump()
        data["mqtt_password"] = "***" if self.mqtt_password else None
        return data


def resolve_path(value: str) -> Path:
    """Resolve a configured path: absolute paths are used as-is, relative ones live under backend/."""
    p = Path(value).expanduser()
    return p if p.is_absolute() else (BACKEND_ROOT / p).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
