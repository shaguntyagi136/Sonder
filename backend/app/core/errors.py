"""Domain exceptions. Each layer raises its own error type so callers can react precisely."""
from __future__ import annotations


class BackendError(Exception):
    """Base class for all expected backend errors."""


class SensorValidationError(BackendError):
    """Incoming sensor payload failed validation. `errors` never contains raw input values."""

    def __init__(self, errors: list[dict]):
        self.errors = errors
        summary = "; ".join(f"{e.get('field', '$')}: {e.get('message', '')}" for e in errors[:5])
        super().__init__(summary or "invalid sensor payload")


class MalformedMessageError(BackendError):
    """An MQTT message could not even be decoded into a JSON object."""


class PreprocessingError(BackendError):
    """A validated reading cannot be turned into the model's feature vector."""


class ModelLoadError(BackendError):
    """The model artifact or its metadata is missing or unusable."""


class ModelUnavailableError(BackendError):
    """Inference was requested but no model is loaded."""


class InferenceError(BackendError):
    """The loaded model failed while predicting."""


class StorageError(BackendError):
    """A database operation failed."""
