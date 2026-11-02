"""MOCK predictor for backend/API/dashboard development ONLY.

It is NOT a model: it maps the inputs to an arbitrary-but-deterministic class via a hash.
It carries no information about stress and every result it returns is flagged is_mock=True.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

from app.ml.base import Predictor
from app.preprocessing.spec import ModelSpec
from app.schemas.prediction import PredictionResult


class MockPredictor(Predictor):
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.labels = spec.class_labels or ["class_0", "class_1", "class_2"]

    def predict(self, values: Sequence[float]) -> PredictionResult:
        digest = hashlib.sha256(",".join(f"{v:.1f}" for v in values).encode()).digest()
        idx = digest[0] % len(self.labels)
        return PredictionResult(
            task="classification",
            label=self.labels[idx],
            class_index=idx,
            model_name=self.spec.model_name,
            model_version=self.spec.model_version,
            is_mock=True,
        )
