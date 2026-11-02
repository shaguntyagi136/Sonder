from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class PredictionResult(BaseModel):
    """What a predictor returns. Classification fills label/class_index, regression fills value."""

    task: Literal["classification", "regression"] = "classification"
    label: Optional[str] = None
    class_index: Optional[int] = None
    value: Optional[float] = None
    confidence: Optional[float] = None  # max class probability, only if the model exposes probabilities
    probabilities: Optional[dict[str, float]] = None
    model_name: str
    model_version: str = "unknown"
    is_mock: bool = False  # True => output is NOT from a trained model
