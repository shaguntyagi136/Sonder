"""Focused compatibility tests for the deployed model feature extractor."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.feature_extractor import FEATURE_ORDER, extract_window_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def completed_window() -> dict:
    """A deterministic completed handoff matching WindowManager's schema."""
    return {
        "device_id": "test-device",
        "window_id": "test-device-0",
        "window_start": "2026-01-01T00:00:00+00:00",
        "window_duration_seconds": 60,
        "sample_counts": {
            "bvp": 3840,
            "eda": 240,
            "temperature": 240,
            "accelerometer": 1920,
        },
        "reading": {
            "bvp": [float(index % 9 - 4) for index in range(3840)],
            "eda": [0.2, 0.3, 1.0, 0.3] * 60,
            "skin_temperature": [32.0 + index / 240 for index in range(240)],
            "acc_x": [3.0] * 1920,
            "acc_y": [4.0] * 1920,
            "acc_z": [0.0] * 1920,
        },
    }


def test_completed_window_produces_metadata_ordered_finite_features():
    metadata = json.loads((PROJECT_ROOT / "ml" / "artifacts" / "metadata.json").read_text())
    features = extract_window_features(completed_window())

    assert tuple(features) == FEATURE_ORDER == tuple(metadata["feature_order"])
    assert len(features) == metadata["feature_count"] == 27
    assert all(math.isfinite(value) for value in features.values())
    assert features["acc_magnitude_mean"] == pytest.approx(5.0)
    assert features["acc_magnitude_rms"] == pytest.approx(5.0)
    assert features["acc_magnitude_mean_abs_diff"] == pytest.approx(0.0)
    assert features["eda_peak_count"] == 60.0
    assert features["eda_peak_mean_amplitude"] == pytest.approx(0.7)


def test_reading_payload_remains_supported_for_existing_callers():
    handoff = completed_window()
    assert extract_window_features(handoff["reading"]) == extract_window_features(handoff)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda handoff: handoff["reading"].pop("eda"), "Missing sensor signals"),
        (lambda handoff: handoff["reading"]["bvp"].pop(), "exactly 3840"),
        (lambda handoff: handoff["reading"]["eda"].__setitem__(0, float("nan")), "NaN"),
        (lambda handoff: handoff.__setitem__("window_duration_seconds", 30), "60 seconds"),
    ],
)
def test_invalid_or_incomplete_windows_are_rejected_safely(mutate, message):
    handoff = completed_window()
    mutate(handoff)

    with pytest.raises(ValueError, match=message):
        extract_window_features(handoff)
