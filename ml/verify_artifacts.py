
import json
import joblib
from pathlib import Path

ARTIFACT_DIR = Path(__file__).parent / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "stress_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

# Check that both files exist
assert MODEL_PATH.exists(), f"Missing model: {MODEL_PATH}"
assert METADATA_PATH.exists(), f"Missing metadata: {METADATA_PATH}"

# Load model and metadata
model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

# Validate model and metadata
assert model.n_features_in_ == 27
assert metadata["feature_count"] == 27
assert len(metadata["feature_order"]) == 27
assert model.classes_.tolist() == [0, 1]

print("ARTIFACT VERIFICATION PASSED")
print("--------------------------------")
print("Model type:", type(model).__name__)
print("Number of features:", model.n_features_in_)
print("Classes:", model.classes_.tolist())
print("Training windows:", metadata["training"]["windows"])
print("Subjects:", len(metadata["training"]["subjects"]))
print("Feature order:")
for i, feature in enumerate(metadata["feature_order"], start=1):
    print(f"{i:2}. {feature}")
