from pathlib import Path

import pytest

from app.preprocessing.wesad import available_subject_ids, load_wesad_participant, prepare_binary_wesad_windows
from app.schemas.sensor import SensorReading


ARCHIVE = Path(__file__).resolve().parents[1] / "data" / "archive (8).zip"


def test_archive_exposes_participants_and_one_raw_e4_window():
    if not ARCHIVE.is_file():
        pytest.skip("WESAD archive is not available")
    subject_id = available_subject_ids(ARCHIVE)[0]
    window = next(load_wesad_participant(ARCHIVE, subject_id, include_original_labels=False, max_windows=1))

    assert window.subject_id == subject_id
    assert window.bvp and window.eda and window.skin_temperature
    assert len(window.acc_x) == len(window.acc_y) == len(window.acc_z)
    assert len(window.bvp) == 60 * 64
    assert len(window.eda) == len(window.skin_temperature) == 60 * 4
    assert len(window.acc_x) == len(window.acc_y) == len(window.acc_z) == 60 * 32
    assert window.sample_rates_hz == {
        "bvp": 64.0, "eda": 4.0, "skin_temperature": 4.0,
        "acc_x": 32.0, "acc_y": 32.0, "acc_z": 32.0,
    }


def _window(subject_id: str, wesad_label: int | None, stress_label: str | None) -> SensorReading:
    return SensorReading(
        subject_id=subject_id,
        timestamp="2020-01-01T00:00:00Z",
        received_at="2020-01-01T00:00:00Z",
        window_seconds=60,
        sample_rates_hz={"bvp": 64, "eda": 4, "skin_temperature": 4, "acc_x": 32, "acc_y": 32, "acc_z": 32},
        bvp=[0.0] * (60 * 64), eda=[0.0] * (60 * 4), skin_temperature=[30.0] * (60 * 4),
        acc_x=[0.0] * (60 * 32), acc_y=[0.0] * (60 * 32), acc_z=[0.0] * (60 * 32),
        wesad_label=wesad_label, stress_label=stress_label,
    )


def test_binary_window_preparation_keeps_subjects_and_excludes_unlabelled(monkeypatch):
    windows = {
        "S2": [_window("S2", 2, "STRESS"), _window("S2", 3, None)],
        "S3": [_window("S3", 1, "NON_STRESS"), _window("S3", None, None)],
    }

    def fake_loader(_archive, subject_id, **_kwargs):
        yield from windows[subject_id]

    monkeypatch.setattr("app.preprocessing.wesad.load_wesad_participant", fake_loader)
    labelled, summary = prepare_binary_wesad_windows("unused.zip", ["S2", "S3"])

    labelled = list(labelled)
    assert [window.stress_label for window in labelled] == ["STRESS", "NON_STRESS"]
    assert [window.subject_id for window in labelled] == ["S2", "S3"]
    assert summary.total_windows == 4
    assert summary.stress_windows == 1 and summary.non_stress_windows == 1
    assert summary.excluded_windows == 2 and summary.subject_ids == ("S2", "S3")
