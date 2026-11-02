"""Persisted records, also used directly as API response bodies."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel

from app.schemas.prediction import PredictionResult
from app.schemas.sensor import SensorReading

# prediction_status values:
#   ok | model_unavailable | preprocessing_failed | inference_failed


class StoredReading(SensorReading):
    id: int
    prediction_status: str
    prediction_error: Optional[str] = None


class StoredPrediction(PredictionResult):
    id: int
    reading_id: int
    device_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime  # timestamp of the reading it was made from
    features: dict[str, float]  # model inputs before normalisation, in model order
    imputed_features: list[str] = []
    created_at: datetime


class PredictionPage(BaseModel):
    items: list[StoredPrediction]
    total: int
    limit: int
    offset: int


class ProcessingResult(BaseModel):
    status: str
    error: Optional[str] = None
    reading: StoredReading
    prediction: Optional[StoredPrediction] = None


class LiveTelemetry(BaseModel):
    """Most recently accepted MQTT sample. Independent of the 60-second ML window."""

    status: Literal["no_data", "receiving", "stale"]
    has_data: bool
    device_id: Optional[str] = None
    subject_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    received_at: Optional[datetime] = None
    age_seconds: Optional[float] = None
    stale_after_seconds: float
    last_modality: Optional[str] = None
    bvp: Optional[float] = None
    eda: Optional[float] = None
    skin_temperature: Optional[float] = None
    acc_x: Optional[float] = None
    acc_y: Optional[float] = None
    acc_z: Optional[float] = None
    sample_counts: dict[str, int]


class ValueStats(BaseModel):
    average: Optional[float] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None


class Summary(BaseModel):
    generated_at: datetime
    window_minutes: int
    window_start: datetime
    device_id: Optional[str] = None
    readings_total: int
    predictions_total: int
    readings_in_window: int
    predictions_in_window: int
    first_reading_at: Optional[datetime] = None
    last_reading_at: Optional[datetime] = None
    devices: list[str]
    prediction_status_counts: dict[str, int]
    label_counts: dict[str, int]
    reading_sources: dict[str, int]
    includes_mock_predictions: bool
    feature_averages: dict[str, Optional[float]]
    prediction_value_stats: Optional[ValueStats] = None
    latest_prediction: Optional[StoredPrediction] = None
