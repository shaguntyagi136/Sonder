"""Raw wearable-window contract shared by dataset, HTTP, and MQTT paths.

The ML contract is raw BVP/PPG, EDA/GSR, skin-temperature, and accelerometer
windows.  Heart-rate and HRV values are deliberately not accepted as model inputs:
they are derived later by feature extraction when the available signal quality and
tooling support it.  ``subject_id`` is carried unchanged so offline evaluation can
use subject-independent splits.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

_ID_PATTERN = r"^[A-Za-z0-9._:\-]{1,64}$"

RAW_SIGNAL_FIELDS = ("bvp", "eda", "skin_temperature", "acc_x", "acc_y", "acc_z")


def _normalise_timestamp(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


class SensorReading(BaseModel):
    """Internal, validated wearable window used by preprocessing and persistence."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: int | None = None
    device_id: str | None = Field(None, pattern=_ID_PATTERN)
    subject_id: str | None = Field(None, pattern=_ID_PATTERN)
    user_id: str | None = Field(None, pattern=_ID_PATTERN)
    session_id: str | None = Field(None, pattern=_ID_PATTERN)
    timestamp: datetime
    received_at: datetime
    transport: str = Field("dataset", pattern=r"^[A-Za-z0-9._\-]{1,32}$")
    source: str = Field("wearable", pattern=r"^[A-Za-z0-9._\-]{1,32}$")
    window_seconds: float = Field(gt=0, le=3600)
    sample_rates_hz: dict[str, float] = Field(default_factory=dict)
    bvp: list[float]
    eda: list[float]
    skin_temperature: list[float]
    acc_x: list[float]
    acc_y: list[float]
    acc_z: list[float]
    # Preserve the dataset's original integer annotation separately from the
    # binary target used by a future stress/non-stress experiment.
    wesad_label: int | None = Field(None, ge=0, le=7)
    stress_label: Literal["STRESS", "NON_STRESS"] | None = None

    @field_validator("timestamp", "received_at")
    @classmethod
    def _utc_timestamps(cls, value: datetime) -> datetime:
        return _normalise_timestamp(value)  # type: ignore[return-value]

    @field_validator(*RAW_SIGNAL_FIELDS, mode="before")
    @classmethod
    def _valid_signal(cls, values: list[float]) -> list[float]:
        if not values:
            raise ValueError("signal window must not be empty")
        for value in values:
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError("signal samples must be finite numeric values")
        return [float(value) for value in values]

    @model_validator(mode="after")
    def _aligned_accelerometer(self) -> "SensorReading":
        lengths = {len(self.acc_x), len(self.acc_y), len(self.acc_z)}
        if len(lengths) != 1:
            raise ValueError("acc_x, acc_y, and acc_z must have equal window lengths")
        return self


class SensorPayload(BaseModel):
    """External raw-window payload. Dataset labels are optional for live inference."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    device_id: str | None = Field(None, pattern=_ID_PATTERN)
    subject_id: str | None = Field(None, pattern=_ID_PATTERN)
    user_id: str | None = Field(None, pattern=_ID_PATTERN)
    session_id: str | None = Field(None, pattern=_ID_PATTERN)
    timestamp: datetime | None = None
    source: str = Field("wearable", pattern=r"^[A-Za-z0-9._\-]{1,32}$")
    window_seconds: float = Field(60.0, gt=0, le=3600)
    sample_rates_hz: dict[str, float] = Field(default_factory=dict)
    bvp: list[float] = Field(validation_alias=AliasChoices("bvp", "ppg"))
    eda: list[float] = Field(validation_alias=AliasChoices("eda", "gsr"))
    skin_temperature: list[float]
    acc_x: list[float]
    acc_y: list[float]
    acc_z: list[float]
    wesad_label: int | None = Field(None, ge=0, le=7)
    stress_label: Literal["STRESS", "NON_STRESS"] | None = None

    @field_validator("timestamp")
    @classmethod
    def _utc_timestamp(cls, value: datetime | None) -> datetime | None:
        return _normalise_timestamp(value)

    @field_validator(*RAW_SIGNAL_FIELDS, mode="before")
    @classmethod
    def _valid_signal(cls, values: list[float]) -> list[float]:
        return SensorReading._valid_signal(values)

    @field_validator("sample_rates_hz")
    @classmethod
    def _valid_sample_rates(cls, values: dict[str, float]) -> dict[str, float]:
        unknown = set(values) - set(RAW_SIGNAL_FIELDS)
        if unknown:
            raise ValueError(f"unknown sample-rate keys: {sorted(unknown)}")
        if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError("sample rates must be positive finite numbers")
        return {name: float(rate) for name, rate in values.items()}

    @model_validator(mode="after")
    def _aligned_accelerometer(self) -> "SensorPayload":
        if len({len(self.acc_x), len(self.acc_y), len(self.acc_z)}) != 1:
            raise ValueError("acc_x, acc_y, and acc_z must have equal window lengths")
        return self

    def to_reading(self, received_at: datetime | None = None, transport: str = "http") -> SensorReading:
        received = received_at or datetime.now(timezone.utc)
        return SensorReading(
            device_id=self.device_id,
            subject_id=self.subject_id,
            user_id=self.user_id,
            session_id=self.session_id,
            timestamp=self.timestamp or received,
            received_at=received,
            transport=transport,
            source=self.source,
            window_seconds=self.window_seconds,
            sample_rates_hz=self.sample_rates_hz,
            bvp=self.bvp,
            eda=self.eda,
            skin_temperature=self.skin_temperature,
            acc_x=self.acc_x,
            acc_y=self.acc_y,
            acc_z=self.acc_z,
            wesad_label=self.wesad_label,
            stress_label=self.stress_label,
        )
