"""Storage repository interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from app.preprocessing.features import FeatureVector
from app.schemas.prediction import PredictionResult
from app.schemas.records import StoredPrediction, StoredReading
from app.schemas.sensor import SensorReading


class Repository(ABC):
    @abstractmethod
    def save(
        self,
        reading: SensorReading,
        features: Optional[FeatureVector],
        prediction: Optional[PredictionResult],
        status: str,
        error: Optional[str],
    ) -> tuple[StoredReading, Optional[StoredPrediction]]:
        ...

    @abstractmethod
    def latest_reading(
        self,
        device_id: Optional[str] = None,
    ) -> Optional[StoredReading]:
        ...

    @abstractmethod
    def latest_prediction(
        self,
        device_id: Optional[str] = None,
    ) -> Optional[StoredPrediction]:
        ...

    @abstractmethod
    def list_predictions(
        self,
        *,
        limit: int,
        offset: int,
        device_id: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> tuple[list[StoredPrediction], int]:
        ...

    @abstractmethod
    def aggregate(
        self,
        since: datetime,
        device_id: Optional[str] = None,
    ) -> dict:
        ...

    @abstractmethod
    def ping(self) -> bool:
        ...

    @abstractmethod
    def close(self) -> None:
        ...
        