"""ML inference service. Knows nothing about MQTT, HTTP or the database.

Required trained-model artifact (see model_artifacts/README.md):
  * a joblib-serialised object exposing ``predict(X)`` for a 2-D array of shape
    (n_samples, n_features), columns in the order given by ``feature_order`` in
    model_meta.json. ``predict_proba`` / ``classes_`` / ``n_features_in_`` are used
    if present.

joblib uses pickle: ONLY load model files you trust.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from app.core.config import Settings
from app.core.errors import InferenceError, ModelLoadError
from app.models.prediction import RawPrediction
from app.preprocessing.features import FeatureSpec

logger = logging.getLogger(__name__)

MOCK_LABELS = ("mock_class_0", "mock_class_1", "mock_class_2")


@dataclass(frozen=True)
class ModelStatus:
    mode: str  # "model" | "mock"
    loaded: bool
    is_mock: bool
    model_name: str | None = None
    model_version: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class Predictor(ABC):
    @abstractmethod
    def predict(self, features: Sequence[float], feature_names: Sequence[str]) -> RawPrediction:
        """Return a prediction for one already-preprocessed feature vector."""

    @abstractmethod
    def status(self) -> ModelStatus: ...


class MockPredictor(Predictor):
    """Deterministic stand-in for backend/plumbing tests ONLY.

    The output is a hash of the inputs. It has no relationship to stress or physiology
    and every result it produces is flagged ``is_mock=True``.
    """

    def predict(self, features, feature_names):
        t0 = time.perf_counter()
        digest = hashlib.sha256(",".join(f"{v:.6f}" for v in features).encode()).digest()
        idx = digest[0] % len(MOCK_LABELS)
        return RawPrediction(
            value=str(idx), label=MOCK_LABELS[idx], model_name="mock-predictor",
            model_version=None, is_mock=True,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def status(self):
        return ModelStatus("mock", True, True, "mock-predictor", None,
                           "MOCK MODE: outputs are meaningless placeholders")


class UnavailablePredictor(Predictor):
    """Used when no usable model exists. The backend keeps running and storing data."""

    def __init__(self, reason: str):
        self.reason = reason

    def predict(self, features, feature_names):
        raise InferenceError(f"model unavailable: {self.reason}")

    def status(self):
        return ModelStatus("model", False, False, detail=self.reason)


class SklearnJoblibPredictor(Predictor):
    def __init__(self, model, spec: FeatureSpec, model_name: str):
        self._model = model
        self._spec = spec
        self._name = model_name
        self._lock = threading.Lock()

    @classmethod
    def load(cls, path: Path, spec: FeatureSpec) -> "SklearnJoblibPredictor":
        import joblib  # local import: only needed when a real model is used

        if not Path(path).is_file():
            raise ModelLoadError(f"model file not found: {path}")
        try:
            model = joblib.load(path)
        except Exception as exc:  # unpickling can raise almost anything
            raise ModelLoadError(f"failed to load model ({type(exc).__name__})") from exc
        if not callable(getattr(model, "predict", None)):
            raise ModelLoadError("loaded object has no predict() method")
        n_in = getattr(model, "n_features_in_", None)
        if n_in is not None and n_in != len(spec.feature_order):
            raise ModelLoadError(
                f"model expects {n_in} features but feature_order has {len(spec.feature_order)}")
        return cls(model, spec, spec.model_name or Path(path).stem)

    def predict(self, features, feature_names):
        if tuple(feature_names) != self._spec.feature_order:
            raise InferenceError("feature order does not match the model specification")
        try:
            import numpy as np

            X = np.asarray([list(features)], dtype=float)
            t0 = time.perf_counter()
            with self._lock:
                y = self._model.predict(X)[0]
                proba = None
                if hasattr(self._model, "predict_proba"):
                    p = self._model.predict_proba(X)[0]
                    classes = getattr(self._model, "classes_", range(len(p)))
                    proba = {
                        self._spec.class_labels.get(str(c.item() if hasattr(c, "item") else c),
                                                    str(c.item() if hasattr(c, "item") else c)): float(pr)
                        for c, pr in zip(classes, p)
                    }
            latency = (time.perf_counter() - t0) * 1000
            value = str(y.item() if hasattr(y, "item") else y)
            return RawPrediction(
                value=value,
                label=self._spec.class_labels.get(value, value),
                model_name=self._name,
                model_version=self._spec.model_version,
                is_mock=False,
                latency_ms=latency,
                probabilities=proba,
                confidence=max(proba.values()) if proba else None,
            )
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(f"inference failed ({type(exc).__name__})") from exc

    def status(self):
        return ModelStatus("model", True, False, self._name, self._spec.model_version)


def build_predictor(settings: Settings, spec: FeatureSpec) -> Predictor:
    if settings.inference_mode == "mock":
        logger.warning("INFERENCE_MODE=mock: predictions are meaningless placeholders")
        return MockPredictor()
    path = settings.resolved_model_path
    try:
        predictor = SklearnJoblibPredictor.load(path, spec)
        logger.info("Model loaded from %s", path)
        return predictor
    except ModelLoadError as exc:
        logger.error("Model not available (%s). Sensor data will be stored but not scored.", exc)
        return UnavailablePredictor(str(exc))
