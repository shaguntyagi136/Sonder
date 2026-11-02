"""InferenceService: owns model loading + prediction. Knows nothing about MQTT, HTTP or the database."""
from __future__ import annotations

import logging
from typing import Optional, Sequence

from app.core.config import Settings, resolve_path
from app.core.errors import InferenceError, ModelLoadError, ModelUnavailableError
from app.ml.base import Predictor
from app.ml.mock_predictor import MockPredictor
from app.ml.sklearn_predictor import SklearnPredictor
from app.preprocessing.spec import ModelSpec, demo_spec, load_model_spec
from app.schemas.prediction import PredictionResult

log = logging.getLogger(__name__)


class InferenceService:
    def __init__(
        self,
        predictor: Optional[Predictor],
        spec: Optional[ModelSpec],
        mode: str,
        load_error: Optional[str] = None,
    ):
        self.predictor = predictor
        self.spec = spec  # may be set even if the artifact failed to load
        self.mode = mode
        self.load_error = load_error

    @classmethod
    def from_settings(cls, settings: Settings) -> "InferenceService":
        """Never raises: a missing model is a reportable state, not a crash."""
        if settings.inference_mode == "mock":
            spec = demo_spec()
            log.warning("Inference running in MOCK mode: predictions are NOT from a trained model")
            return cls(MockPredictor(spec), spec, "mock")

        spec: Optional[ModelSpec] = None
        try:
            spec = load_model_spec(resolve_path(settings.model_metadata_path))
            predictor = SklearnPredictor.load(resolve_path(settings.model_path), spec)
        except ModelLoadError as exc:
            log.error("Model not loaded: %s (readings will be stored without predictions)", exc)
            return cls(None, spec, "model", load_error=str(exc))
        log.info("Model loaded: %s (version %s), %d features", spec.model_name, spec.model_version, len(spec.feature_order))
        return cls(predictor, spec, "model")

    @property
    def ready(self) -> bool:
        return self.predictor is not None

    def predict(self, values: Sequence[float]) -> PredictionResult:
        if self.predictor is None:
            raise ModelUnavailableError(self.load_error or "no model loaded")
        try:
            return self.predictor.predict(values)
        except InferenceError:
            raise
        except Exception as exc:  # defensive: predictors must not leak arbitrary exceptions
            raise InferenceError(f"unexpected inference failure: {type(exc).__name__}") from None

    def status(self) -> dict:
        return {
            "mode": self.mode,
            "model_loaded": self.ready,
            "is_mock": self.mode == "mock",
            "model_name": self.spec.model_name if self.spec else None,
            "model_version": self.spec.model_version if self.spec else None,
            "feature_order": list(self.spec.feature_order) if self.spec else None,
            "error": self.load_error,
        }
