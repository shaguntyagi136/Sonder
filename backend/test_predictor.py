
import sys
from pathlib import Path

# Allow importing from backend/app
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent / "app")
)

from predictor import StressPredictor

# Initialize predictor
predictor = StressPredictor()

# Create a controlled test input using the model's feature order
test_features = {
    name: 0.5
    for name in predictor.feature_order
}

# Run prediction
result = predictor.predict(test_features)

print("\nPREDICTION TEST")
print("----------------------------")
print("Prediction:", result["prediction"])
print("Label:", result["label"])
print("Stress probability:", result["stress_probability"])
print("Baseline probability:", result["baseline_probability"])
print("----------------------------")

assert result["prediction"] in [0, 1]
assert result["label"] in ["BASELINE", "STRESS"]
assert 0 <= result["stress_probability"] <= 1

print("PREDICTOR TEST PASSED!")
