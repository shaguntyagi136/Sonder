"""Train and package the final SONDER Random Forest deployment artifact.

This is intentionally a full-data deployment-training script. It does not report an
evaluation score or make a train/test split; subject-independent evaluation belongs
to the LOSO evaluation workflow.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier

from app.preprocessing.features import DEFAULT_FEATURE_ORDER
from app.preprocessing.preprocessor import build_tabular_dataset
from app.preprocessing.spec import ModelSpec
from app.preprocessing.wesad import available_subject_ids, prepare_binary_wesad_windows

BACKEND_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_PATH = BACKEND_ROOT / "data" / "archive (8).zip"
MODEL_PATH = BACKEND_ROOT / "models" / "stress_model.joblib"
METADATA_PATH = BACKEND_ROOT / "models" / "stress_model.metadata.json"
WINDOW_SECONDS = 60.0
SAMPLE_RATES_HZ = {
    "bvp": 64.0,
    "eda": 4.0,
    "skin_temperature": 4.0,
    "acc_x": 32.0,
    "acc_y": 32.0,
    "acc_z": 32.0,
}
MODEL_PARAMETERS = {
    "n_estimators": 300,
    "random_state": 42,
    "class_weight": "balanced",
    "n_jobs": -1,
}
LABEL_MAPPING = {
    "wesad_original": {"1": "BASELINE", "2": "STRESS"},
    "binary_target": {"BASELINE": "NON_STRESS", "STRESS": "STRESS"},
    "excluded": ["UNDEFINED", "AMUSEMENT", "MEDITATION", "UNKNOWN"],
}


def main() -> None:
    if not ARCHIVE_PATH.is_file():
        raise FileNotFoundError(f"WESAD archive not found: {ARCHIVE_PATH}")

    subject_ids = available_subject_ids(ARCHIVE_PATH)
    labelled_windows, window_summary = prepare_binary_wesad_windows(
        ARCHIVE_PATH,
        subject_ids,
        window_seconds=WINDOW_SECONDS,
    )
    spec = ModelSpec(
        model_name="SONDER Random Forest",
        model_version="1.0.0",
        task="classification",
        feature_order=list(DEFAULT_FEATURE_ORDER),
        class_labels=["NON_STRESS", "STRESS"],
    )
    dataset = build_tabular_dataset(labelled_windows, spec)
    if not dataset.rows:
        raise RuntimeError("no usable binary WESAD windows were produced")
    counts = Counter(dataset.y)

    print(f"subject IDs: {', '.join(subject_ids)}")
    print(f"total usable rows: {len(dataset.rows)}")
    print(f"stress count: {counts['STRESS']}")
    print(f"non-stress count: {counts['NON_STRESS']}")
    print(f"feature count: {len(dataset.feature_names)}")
    print(f"feature names: {list(dataset.feature_names)}")

    model = RandomForestClassifier(**MODEL_PARAMETERS)
    model.fit(dataset.X, dataset.y)

    metadata = {
        "model_name": spec.model_name,
        "model_version": spec.model_version,
        "task": spec.task,
        "feature_order": list(spec.feature_order),
        "class_labels": spec.class_labels,
        "model_type": "sklearn.ensemble.RandomForestClassifier",
        "model_parameters": model.get_params(),
        "window_seconds": WINDOW_SECONDS,
        "sample_rates_hz": SAMPLE_RATES_HZ,
        "label_mapping": LABEL_MAPPING,
        "preprocessing": {
            "contract": "app.schemas.sensor.SensorReading",
            "extractor": "app.preprocessing.features.extract_window_features",
            "normalization": None,
            "imputation": {},
        },
        "training": {
            "rows": len(dataset.rows),
            "subjects": subject_ids,
            "window_summary": {
                "total": window_summary.total_windows,
                "stress": window_summary.stress_windows,
                "non_stress": window_summary.non_stress_windows,
                "excluded": window_summary.excluded_windows,
            },
        },
    }
    # Validate the inference-critical subset before persisting it.
    ModelSpec.model_validate(metadata)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    loaded_model = joblib.load(MODEL_PATH)
    if loaded_model.n_features_in_ != len(dataset.feature_names):
        raise RuntimeError("reloaded model feature dimension does not match training metadata")
    sample_x = [dataset.X[0]]
    prediction = loaded_model.predict(sample_x)[0]
    probability = float(max(loaded_model.predict_proba(sample_x)[0]))
    print(
        "Training artifact saved successfully: "
        f"path={MODEL_PATH}, rows={len(dataset.rows)}, features={len(dataset.feature_names)}, "
        f"example_prediction={prediction}, example_probability={probability:.4f}"
    )


if __name__ == "__main__":
    main()
