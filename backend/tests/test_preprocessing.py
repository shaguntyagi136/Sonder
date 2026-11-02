from datetime import datetime, timezone
import math

from app.preprocessing.features import DEFAULT_FEATURE_ORDER, extract_window_features
from app.preprocessing.preprocessor import build_tabular_dataset
from app.schemas.sensor import SensorReading

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def reading() -> SensorReading:
    bvp = [math.sin(2 * math.pi * 1.2 * index / 64) for index in range(60 * 64)]
    eda = [0.1] * (60 * 4)
    for index in (20, 100, 180):
        eda[index] = 0.6
    return SensorReading(
        subject_id="S2", timestamp=NOW, received_at=NOW, window_seconds=60,
        sample_rates_hz={"bvp": 64, "eda": 4, "skin_temperature": 4, "acc_x": 32, "acc_y": 32, "acc_z": 32},
        bvp=bvp, eda=eda, skin_temperature=[30 + index / 100 for index in range(60 * 4)],
        acc_x=[float(index % 3) for index in range(60 * 32)],
        acc_y=[0.0] * (60 * 32), acc_z=[1.0] * (60 * 32),
    )


def test_window_features_are_deterministic_and_cover_default_order():
    first = extract_window_features(reading())
    second = extract_window_features(reading())
    assert first == second
    assert set(DEFAULT_FEATURE_ORDER).issubset(first)
    assert all(math.isfinite(first[name]) for name in DEFAULT_FEATURE_ORDER)


def test_bvp_heart_rate_is_estimated_without_creating_hrv():
    features = extract_window_features(reading())
    assert 68.0 < features["bvp_estimated_heart_rate_bpm"] < 76.0
    assert not any("hrv" in name.lower() for name in features)


def test_eda_peak_features_are_robust_and_movement_features_are_present():
    features = extract_window_features(reading())
    assert features["eda_peak_count"] == 3.0
    assert features["eda_peak_mean_amplitude"] == 0.5
    assert features["eda_peak_max_amplitude"] == 0.5
    assert features["acc_magnitude_rms"] > 0
    assert features["acc_magnitude_mean_abs_diff"] > 0


def test_tabular_dataset_preserves_rows_labels_features_and_subject_groups():
    stress = reading().model_copy(update={"subject_id": "S2", "wesad_label": 2, "stress_label": "STRESS"})
    non_stress = reading().model_copy(update={"subject_id": "S3", "wesad_label": 1, "stress_label": "NON_STRESS"})
    unlabeled = reading().model_copy(update={"subject_id": "S4", "wesad_label": 3, "stress_label": None})

    dataset = build_tabular_dataset([stress, non_stress, unlabeled])

    assert len(dataset.rows) == 2
    assert dataset.feature_names == DEFAULT_FEATURE_ORDER
    assert dataset.y == ("STRESS", "NON_STRESS")
    assert dataset.groups == ("S2", "S3")
    assert dataset.rows[0].raw_features == {
        name: extract_window_features(stress)[name] for name in DEFAULT_FEATURE_ORDER
    }
    assert dataset.X[0] == tuple(dataset.rows[0].raw_features[name] for name in dataset.feature_names)
