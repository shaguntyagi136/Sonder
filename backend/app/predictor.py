
import json
import joblib
import numpy as np
from pathlib import Path

if __package__:
    from .feature_extractor import FEATURE_ORDER, extract_window_features
else:
    from feature_extractor import FEATURE_ORDER, extract_window_features

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "ml" / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "stress_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"


class StressPredictor:

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

        with open(METADATA_PATH, "r") as file:
            self.metadata = json.load(file)

        self.feature_order = self.metadata["feature_order"]

        if len(self.feature_order) != self.model.n_features_in_:
            raise ValueError("Feature count mismatch!")
        if tuple(self.feature_order) != FEATURE_ORDER:
            raise ValueError("Model feature order does not match the extractor!")

    def predict_window(self, window):
        """Extract model features from a completed sensor window and predict it."""
        try:
            features = extract_window_features(window)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid sensor window: {error}") from error
        return self.predict(features)

    def predict(self, features):
        if isinstance(features, dict):
            missing = set(self.feature_order) - set(features)
            extra = set(features) - set(self.feature_order)

            if missing or extra:
                raise ValueError(
                    f"Feature mismatch. Missing: {missing}, Extra: {extra}"
                )

            values = [features[name] for name in self.feature_order]

        else:
            try:
                values = list(features)
            except (TypeError, ValueError) as error:
                raise ValueError("Features must be a mapping or sequence") from error

        if len(values) != len(self.feature_order):
            raise ValueError("Incorrect number of features!")

        if any(isinstance(value, bool) for value in values):
            raise ValueError("Features must contain numeric values")

        try:
            values = np.asarray(values, dtype=np.float64).reshape(1, -1)
        except (TypeError, ValueError) as error:
            raise ValueError("Features must contain numeric values") from error

        if not np.isfinite(values).all():
            raise ValueError("Features contain NaN or infinite values!")

        try:
            prediction = int(self.model.predict(values)[0])
            probabilities = self.model.predict_proba(values)[0]
        except Exception as error:
            raise RuntimeError("Model prediction failed") from error

        stress_probability = float(probabilities[1])
        baseline_probability = float(probabilities[0])

        return {
            "prediction": prediction,
            "label": "STRESS" if prediction == 1 else "BASELINE",
            "stress_probability": round(stress_probability, 4),
            "baseline_probability": round(baseline_probability, 4),
            "confidence": round(max(stress_probability, baseline_probability), 4),
        }
