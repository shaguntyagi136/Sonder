from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from app.schemas.prediction import PredictionResult


class Predictor(ABC):
    """Anything that maps an ordered feature vector to a PredictionResult."""

    @abstractmethod
    def predict(self, values: Sequence[float]) -> PredictionResult: ...
