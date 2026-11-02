"""ModelSpec: the contract between the trained model and the backend.

It is loaded from the model_metadata.json that must ship next to the model artifact, so the
feature order is defined by whoever trained the model and never guessed by the backend.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError, model_validator

from app.core.errors import ModelLoadError
from app.preprocessing.features import DEFAULT_FEATURE_ORDER, FEATURE_EXTRACTORS


class Normalization(BaseModel):
    """z-score parameters, only needed if the model artifact does NOT already embed a scaler.
    Do not configure this if your saved sklearn Pipeline already scales (it would double-scale)."""

    mean: dict[str, float]
    std: dict[str, float]


class ModelSpec(BaseModel):
    model_name: str
    model_version: str = "unknown"
    task: Literal["classification", "regression"] = "classification"
    feature_order: list[str] = Field(min_length=1)
    class_labels: Optional[list[str]] = None  # names for model.classes_ in order (classification)
    normalization: Optional[Normalization] = None
    impute: dict[str, float] = Field(default_factory=dict)  # only used if MISSING_FEATURE_POLICY=impute

    @model_validator(mode="after")
    def _check(self) -> "ModelSpec":
        unknown = [f for f in self.feature_order if f not in FEATURE_EXTRACTORS]
        if unknown:
            raise ValueError(f"unknown feature(s) {unknown}; supported: {sorted(FEATURE_EXTRACTORS)}")
        if len(set(self.feature_order)) != len(self.feature_order):
            raise ValueError("feature_order contains duplicates")
        bad_impute = [f for f in self.impute if f not in self.feature_order]
        if bad_impute:
            raise ValueError(f"impute refers to features not in feature_order: {bad_impute}")
        if self.normalization:
            for f in self.feature_order:
                if f not in self.normalization.mean or f not in self.normalization.std:
                    raise ValueError(f"normalization missing mean/std for '{f}'")
                if self.normalization.std[f] == 0:
                    raise ValueError(f"normalization std for '{f}' is zero")
        return self


def demo_spec() -> ModelSpec:
    """Spec for MOCK mode only. Class names are deliberately meaningless."""
    return ModelSpec(
        model_name="mock-demo-model",
        model_version="mock",
        feature_order=list(DEFAULT_FEATURE_ORDER),
        class_labels=["class_0", "class_1", "class_2"],
    )


def load_model_spec(path: Path) -> ModelSpec:
    if not path.is_file():
        raise ModelLoadError("model metadata file not found (see model_artifacts/README.md)")
    try:
        return ModelSpec.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        raise ModelLoadError(f"model metadata is not valid JSON: {exc.msg}") from None
    except ValidationError as exc:
        raise ModelLoadError(f"model metadata is invalid: {exc.errors()[0]['msg']}") from None
