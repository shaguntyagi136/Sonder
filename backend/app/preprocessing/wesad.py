"""WESAD archive ingestion into the project's raw wearable-window contract.

The supplied WESAD release stores raw Empatica E4 signals as CSV files inside
``WESAD/S<id>/S<id>_E4_Data``.  Each CSV starts with its UTC unix start time and
sample rate.  Original WESAD annotations reside in ``S<id>.pkl`` as a 700 Hz label
series.  This module keeps those annotations separate from the future binary target.
"""
from __future__ import annotations

import csv
import io
import pickle
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from app.schemas.sensor import SensorReading

WESAD_LABEL_NAMES: dict[int, str] = {
    0: "UNDEFINED",
    1: "BASELINE",
    2: "STRESS",
    3: "AMUSEMENT",
    4: "MEDITATION",
    5: "UNKNOWN_5",
    6: "UNKNOWN_6",
    7: "UNKNOWN_7",
}
WESAD_LABEL_SAMPLE_RATE_HZ = 700.0


class WesadLoadError(ValueError):
    """The supplied archive does not contain a usable WESAD participant record."""


@dataclass(frozen=True)
class _Signal:
    start_time: float
    sample_rate_hz: float
    samples: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class WesadWindowSummary:
    """Counts from deterministic binary-label preparation; no split is performed."""

    total_windows: int
    stress_windows: int
    non_stress_windows: int
    excluded_windows: int
    subject_ids: tuple[str, ...]


def available_subject_ids(archive_path: str | Path) -> list[str]:
    """Return participant IDs actually present in the supplied WESAD archive."""
    try:
        with zipfile.ZipFile(archive_path) as archive:
            subjects = {
                member.split("/")[1]
                for member in archive.namelist()
                if member.startswith("WESAD/S") and member.count("/") >= 2 and member.split("/")[1].startswith("S")
            }
    except (OSError, zipfile.BadZipFile) as exc:
        raise WesadLoadError(f"cannot open WESAD archive: {exc}") from exc
    return sorted(subjects, key=lambda subject: int(subject[1:]) if subject[1:].isdigit() else subject)


def _read_e4_signal(archive: zipfile.ZipFile, member: str, columns: int) -> _Signal:
    try:
        binary = archive.open(member)
    except KeyError:
        raise WesadLoadError(f"WESAD signal is missing: {member}") from None
    with binary, io.TextIOWrapper(binary, encoding="utf-8", newline="") as text:
        rows = csv.reader(text)
        try:
            start = tuple(float(value.strip()) for value in next(rows))
            rates = tuple(float(value.strip()) for value in next(rows))
        except (StopIteration, ValueError) as exc:
            raise WesadLoadError(f"WESAD signal has invalid header: {member}") from exc
        if len(start) != columns or len(rates) != columns:
            raise WesadLoadError(f"WESAD signal has unexpected column count: {member}")
        if any(rate <= 0 for rate in rates) or len(set(rates)) != 1:
            raise WesadLoadError(f"WESAD signal has invalid or unequal sample rates: {member}")
        if len(set(start)) != 1:
            raise WesadLoadError(f"WESAD signal has unequal axis start times: {member}")
        samples: list[tuple[float, ...]] = []
        for row_number, row in enumerate(rows, start=3):
            if len(row) != columns:
                raise WesadLoadError(f"WESAD signal has malformed row {row_number}: {member}")
            try:
                samples.append(tuple(float(value.strip()) for value in row))
            except ValueError as exc:
                raise WesadLoadError(f"WESAD signal has non-numeric row {row_number}: {member}") from exc
    if not samples:
        raise WesadLoadError(f"WESAD signal has no samples: {member}")
    return _Signal(start[0], rates[0], tuple(samples))


def _load_original_labels(archive: zipfile.ZipFile, subject_id: str) -> tuple[int, ...]:
    """Read the original WESAD 700 Hz annotation series without remapping it."""
    member = f"WESAD/{subject_id}/{subject_id}.pkl"
    try:
        stream = archive.open(member)
    except KeyError:
        raise WesadLoadError(f"WESAD label file is missing: {member}") from None
    try:
        # WESAD pickle files contain NumPy arrays. Importing here keeps archive
        # discovery and CSV-only ingestion independent of NumPy at module import.
        import numpy  # noqa: F401

        with stream:
            payload = pickle.load(stream, encoding="latin1")
        labels = payload["label"]
        return tuple(int(value) for value in labels)
    except ModuleNotFoundError as exc:
        raise WesadLoadError("loading WESAD original labels requires NumPy for the supplied .pkl format") from exc
    except (KeyError, pickle.UnpicklingError, EOFError, ValueError, TypeError) as exc:
        raise WesadLoadError(f"cannot read WESAD original labels for {subject_id}") from exc


def _binary_label(original_label: int | None) -> Literal["STRESS", "NON_STRESS"] | None:
    # Only baseline and stress are assigned to the initial binary target. Amusement,
    # meditation, undefined, and undocumented values remain explicitly unmapped.
    if original_label == 2:
        return "STRESS"
    if original_label == 1:
        return "NON_STRESS"
    return None


def _window_label(labels: tuple[int, ...], start_seconds: float, duration_seconds: float) -> int | None:
    start = round(start_seconds * WESAD_LABEL_SAMPLE_RATE_HZ)
    end = round((start_seconds + duration_seconds) * WESAD_LABEL_SAMPLE_RATE_HZ)
    segment = labels[start:end]
    # A transition window does not receive a fabricated majority label.
    return segment[0] if segment and len(set(segment)) == 1 else None


def load_wesad_participant(
    archive_path: str | Path,
    subject_id: str,
    *,
    window_seconds: float = 60.0,
    include_original_labels: bool = True,
    label_start_offset_seconds: float = 0.0,
    max_windows: int | None = None,
) -> Iterator[SensorReading]:
    """Yield aligned raw E4 windows for one WESAD participant.

    ``label_start_offset_seconds`` explicitly records any verified offset between
    the E4 session start and the 700 Hz label series. The WESAD default is zero;
    callers must provide a non-zero verified offset rather than silently shifting
    labels. Windows containing more than one original label remain unlabeled.
    """
    if window_seconds <= 0:
        raise WesadLoadError("window_seconds must be positive")
    if max_windows is not None and max_windows < 1:
        raise WesadLoadError("max_windows must be at least 1 when provided")
    if subject_id not in available_subject_ids(archive_path):
        raise WesadLoadError(f"WESAD participant is not present: {subject_id}")

    base = f"WESAD/{subject_id}/{subject_id}_E4_Data"
    try:
        with zipfile.ZipFile(archive_path) as archive:
            bvp = _read_e4_signal(archive, f"{base}/BVP.csv", 1)
            eda = _read_e4_signal(archive, f"{base}/EDA.csv", 1)
            temperature = _read_e4_signal(archive, f"{base}/TEMP.csv", 1)
            accelerometer = _read_e4_signal(archive, f"{base}/ACC.csv", 3)
            signals = (bvp, eda, temperature, accelerometer)
            starts = {signal.start_time for signal in signals}
            if len(starts) != 1:
                raise WesadLoadError("WESAD E4 modalities have different session start times; alignment is not inferred")
            labels = _load_original_labels(archive, subject_id) if include_original_labels else ()

        samples_per_window = {
            "bvp": round(bvp.sample_rate_hz * window_seconds),
            "eda": round(eda.sample_rate_hz * window_seconds),
            "skin_temperature": round(temperature.sample_rate_hz * window_seconds),
            "acc": round(accelerometer.sample_rate_hz * window_seconds),
        }
        if any(count < 1 for count in samples_per_window.values()):
            raise WesadLoadError("window duration is shorter than a source sample interval")
        duration = min(
            len(bvp.samples) / bvp.sample_rate_hz,
            len(eda.samples) / eda.sample_rate_hz,
            len(temperature.samples) / temperature.sample_rate_hz,
            len(accelerometer.samples) / accelerometer.sample_rate_hz,
        )
        window_count = int(duration // window_seconds)
        if max_windows is not None:
            window_count = min(window_count, max_windows)

        for index in range(window_count):
            bvp_start = index * samples_per_window["bvp"]
            eda_start = index * samples_per_window["eda"]
            temp_start = index * samples_per_window["skin_temperature"]
            acc_start = index * samples_per_window["acc"]
            original_label = (
                _window_label(labels, index * window_seconds + label_start_offset_seconds, window_seconds)
                if labels else None
            )
            yield SensorReading(
                subject_id=subject_id,
                device_id="wesad-e4",
                timestamp=datetime.fromtimestamp(bvp.start_time + index * window_seconds, tz=timezone.utc),
                received_at=datetime.now(timezone.utc),
                transport="dataset",
                source="wesad_e4",
                window_seconds=window_seconds,
                sample_rates_hz={
                    "bvp": bvp.sample_rate_hz,
                    "eda": eda.sample_rate_hz,
                    "skin_temperature": temperature.sample_rate_hz,
                    "acc_x": accelerometer.sample_rate_hz,
                    "acc_y": accelerometer.sample_rate_hz,
                    "acc_z": accelerometer.sample_rate_hz,
                },
                bvp=[sample[0] for sample in bvp.samples[bvp_start:bvp_start + samples_per_window["bvp"]]],
                eda=[sample[0] for sample in eda.samples[eda_start:eda_start + samples_per_window["eda"]]],
                skin_temperature=[sample[0] for sample in temperature.samples[temp_start:temp_start + samples_per_window["skin_temperature"]]],
                acc_x=[sample[0] for sample in accelerometer.samples[acc_start:acc_start + samples_per_window["acc"]]],
                acc_y=[sample[1] for sample in accelerometer.samples[acc_start:acc_start + samples_per_window["acc"]]],
                acc_z=[sample[2] for sample in accelerometer.samples[acc_start:acc_start + samples_per_window["acc"]]],
                wesad_label=original_label,
                stress_label=_binary_label(original_label),
            )
    except (OSError, zipfile.BadZipFile) as exc:
        raise WesadLoadError(f"cannot open WESAD archive: {exc}") from exc


def prepare_binary_wesad_windows(
    archive_path: str | Path,
    subject_ids: Iterable[str],
    *,
    window_seconds: float = 60.0,
    label_start_offset_seconds: float = 0.0,
) -> tuple[Iterator[SensorReading], WesadWindowSummary]:
    """Return binary-labelled windows and their provenance-preserving summary.

    Participants are read one at a time and never combined into a window. Windows
    without the conservative binary mapping performed by ``load_wesad_participant``
    are counted as excluded and are not yielded. This function intentionally does
    not shuffle or split the resulting subject-aware collection.
    """
    selected_subjects = tuple(subject_ids)
    if not selected_subjects:
        raise WesadLoadError("at least one WESAD subject_id is required")
    if len(set(selected_subjects)) != len(selected_subjects):
        raise WesadLoadError("subject_ids must not contain duplicates")

    labelled: list[SensorReading] = []
    total = stress = non_stress = excluded = 0
    for subject_id in selected_subjects:
        for window in load_wesad_participant(
            archive_path,
            subject_id,
            window_seconds=window_seconds,
            include_original_labels=True,
            label_start_offset_seconds=label_start_offset_seconds,
        ):
            if window.subject_id != subject_id:
                raise WesadLoadError("loader returned a window belonging to a different subject")
            total += 1
            if window.stress_label == "STRESS":
                stress += 1
                labelled.append(window)
            elif window.stress_label == "NON_STRESS":
                non_stress += 1
                labelled.append(window)
            else:
                excluded += 1

    return iter(labelled), WesadWindowSummary(
        total_windows=total,
        stress_windows=stress,
        non_stress_windows=non_stress,
        excluded_windows=excluded,
        subject_ids=selected_subjects,
    )
