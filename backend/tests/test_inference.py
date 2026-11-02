"""Prediction-service interface tests.

NOTE: `tiny_model` below is a throw-away fixture trained on a few made-up numbers, ONLY to
exercise the artifact-loading code path. It has no predictive meaning and is never shipped.
"""
import json

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from app.core.errors import InferenceError, ModelUnavailableError
from app.ml.service import InferenceService
from app.ml.sklearn_predictor import SklearnPredictor
from app.preprocessing.spec import ModelSpec
from tests.conftest import make_settings

FEATURES = ["heart_rate", "hrv", "spo2"]


@pytest.fixture
def tiny_model(tmp_path):
    X = np.array([[60, 60, 98], [65, 55, 97], [90, 20, 95], [95, 25, 96]], dtype=float)
    y = np.array([0, 0, 1, 1])
    joblib.dump(LogisticRegression().fit(X, y), tmp_path / "m.joblib")
    (tmp_path / "meta.json").write_text(
        json.dumps({"model_name": "fixture", "model_version": "t1", "feature_order": FEATURES, "class_labels": ["a", "b"]})
    )
    return tmp_path


def model_settings(d, **kw):
    return make_settings(inference_mode="model", model_path=str(d / "m.joblib"), model_metadata_path=str(d / "meta.json"), **kw)


def test_mock_mode_interface():
    svc = InferenceService.from_settings(make_settings())
    assert svc.ready and svc.status()["is_mock"] is True
    r1, r2 = svc.predict([70, 40, 97, 33, 1.0]), svc.predict([70, 40, 97, 33, 1.0])
    assert r1 == r2  # deterministic
    assert r1.is_mock and r1.label in {"class_0", "class_1", "class_2"} and r1.model_name == "mock-demo-model"


def test_model_mode_without_artifact_is_reported_not_faked(tmp_path):
    svc = InferenceService.from_settings(model_settings(tmp_path))  # nothing in tmp_path
    assert not svc.ready
    assert svc.status()["model_loaded"] is False and svc.status()["error"]
    with pytest.raises(ModelUnavailableError):
        svc.predict([1, 2, 3])


def test_default_settings_do_not_silently_mock():
    assert make_settings(inference_mode="model").inference_mode == "model"


def test_loads_and_predicts_with_real_artifact(tiny_model):
    svc = InferenceService.from_settings(model_settings(tiny_model))
    assert svc.ready and not svc.status()["is_mock"]
    res = svc.predict([62, 58, 98])
    assert res.label in {"a", "b"} and res.class_index in (0, 1)
    assert res.is_mock is False and res.model_version == "t1"
    assert set(res.probabilities) == {"a", "b"} and res.confidence == max(res.probabilities.values())


def test_metadata_missing_means_not_loaded(tiny_model):
    (tiny_model / "meta.json").unlink()
    svc = InferenceService.from_settings(model_settings(tiny_model))
    assert not svc.ready and "metadata" in svc.load_error


def test_corrupt_artifact_handled(tiny_model):
    (tiny_model / "m.joblib").write_bytes(b"not a pickle")
    svc = InferenceService.from_settings(model_settings(tiny_model))
    assert not svc.ready and "could not load" in svc.load_error


def test_feature_count_mismatch_detected(tiny_model):
    meta = json.loads((tiny_model / "meta.json").read_text())
    meta["feature_order"] = ["heart_rate", "hrv"]
    (tiny_model / "meta.json").write_text(json.dumps(meta))
    svc = InferenceService.from_settings(model_settings(tiny_model))
    assert not svc.ready and "features" in svc.load_error


def test_feature_name_order_mismatch_detected(tiny_model):
    model = joblib.load(tiny_model / "m.joblib")
    model.feature_names_in_ = np.array(["hrv", "heart_rate", "spo2"])  # trained in a different order
    with pytest.raises(Exception, match="feature_names_in_"):
        SklearnPredictor(model, ModelSpec(model_name="x", feature_order=FEATURES))


def test_inference_failure_is_wrapped(tiny_model):
    svc = InferenceService.from_settings(model_settings(tiny_model))
    with pytest.raises(InferenceError):
        svc.predict([1, 2])  # wrong vector length
