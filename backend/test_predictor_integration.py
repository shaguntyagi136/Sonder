from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.predictor import StressPredictor


def completed_window() -> dict:
    return {
        "window_duration_seconds": 60,
        "reading": {
            "bvp": [float(index % 9 - 4) for index in range(3840)],
            "eda": [0.2, 0.3, 1.0, 0.3] * 60,
            "skin_temperature": [32.0 + index / 240 for index in range(240)],
            "acc_x": [3.0] * 1920,
            "acc_y": [4.0] * 1920,
            "acc_z": [0.0] * 1920,
        },
    }


def test_predict_window_extracts_features_and_returns_prediction():
    result = StressPredictor().predict_window(completed_window())

    assert result["prediction"] in (0, 1)
    assert result["label"] in ("BASELINE", "STRESS")
    assert 0 <= result["confidence"] <= 1
    assert result["confidence"] == max(
        result["stress_probability"],
        result["baseline_probability"],
    )


@pytest.mark.parametrize(
    "window",
    [
        None,
        {"window_duration_seconds": 60, "reading": {}},
    ],
)
def test_predict_window_rejects_invalid_input(window):
    with pytest.raises(ValueError, match="Invalid sensor window"):
        StressPredictor().predict_window(window)
