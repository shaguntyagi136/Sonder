"""Loads a scikit-learn-compatible artifact saved with joblib and runs predictions.

SECURITY: joblib artifacts are pickles - only load files you trained or fully trust.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

from app.core.errors import InferenceError, ModelLoadError
from app.ml.base import Predictor
from app.preprocessing.spec import ModelSpec
from app.schemas.prediction import PredictionResult


class SklearnPredictor(Predictor):
    def __init__(self, model, spec: ModelSpec):
        if not hasattr(model, "predict"):
            raise ModelLoadError("loaded object has no predict() method")
        n_expected = getattr(model, "n_features_in_", None)
        if n_expected is not None and n_expected != len(spec.feature_order):
            raise ModelLoadError(
                f"model expects {n_expected} features but metadata feature_order has {len(spec.feature_order)}"
            )
        trained_names = getattr(model, "feature_names_in_", None)
        if trained_names is not None and list(trained_names) != spec.feature_order:
            raise ModelLoadError("model feature_names_in_ do not match metadata feature_order (names or order differ)")

        # The service scores one wearable window at a time.  Running the forest
        # serially avoids worker-pool setup at request time (which is unavailable
        # in some Windows service environments) without changing learned trees,
        # predictions, or the serialized deployment artifact.
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1

        self.model = model
        self.spec = spec

    @classmethod
    def load(cls, path: Path, spec: ModelSpec) -> "SklearnPredictor":
        if not path.is_file():
            raise ModelLoadError("model artifact not found (see model_artifacts/README.md)")
        try:
            model = joblib.load(path)
        except Exception as exc:  # unpickling can fail in many ways (version skew, corrupt file, ...)
            raise ModelLoadError(f"could not load model artifact: {type(exc).__name__}") from None
        return cls(model, spec)

    def predict(self, values: Sequence[float]) -> PredictionResult:
        x = np.asarray([list(values)], dtype=float)
        try:
            with warnings.catch_warnings():
                # feature names were verified in __init__; sklearn warns when fitted on a DataFrame
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                raw_pred = self.model.predict(x)[0]
                proba = self.model.predict_proba(x)[0] if hasattr(self.model, "predict_proba") else None
        except Exception as exc:
            raise InferenceError(f"model.predict failed: {type(exc).__name__}: {exc}") from None
        return self._to_result(raw_pred, proba)

    def _to_result(self, raw_pred, proba) -> PredictionResult:
        base = dict(model_name=self.spec.model_name, model_version=self.spec.model_version, task=self.spec.task)
        if self.spec.task == "regression":
            return PredictionResult(value=float(raw_pred), **base)

        classes = list(getattr(self.model, "classes_", []))
        labels = self.spec.class_labels
        if classes:
            try:
                idx = classes.index(raw_pred)
            except ValueError:
                raise InferenceError("model returned a class that is not in classes_") from None
            names = labels if labels and len(labels) == len(classes) else [str(c) for c in classes]
            if labels and len(labels) != len(classes):
                raise InferenceError("metadata class_labels length differs from model.classes_")
            probabilities = (
                {n: float(p) for n, p in zip(names, proba)} if proba is not None and len(proba) == len(names) else None
            )
            return PredictionResult(
                label=names[idx],
                class_index=idx,
                probabilities=probabilities,
                confidence=max(probabilities.values()) if probabilities else None,
                **base,
            )
        return PredictionResult(label=str(raw_pred), **base)
