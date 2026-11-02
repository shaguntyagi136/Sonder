"""Deterministic baseline features from one raw wearable window.

The module intentionally does not create HRV: that requires a validated beat-to-beat
interval pipeline.  It may expose an estimated heart rate only when the BVP waveform
contains a sufficiently regular sequence of peaks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

from app.schemas.sensor import SensorReading


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _std(values: Sequence[float]) -> float:
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def _slope(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    x_mean = (len(values) - 1) / 2
    y_mean = _mean(values)
    denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
    return sum((index - x_mean) * (value - y_mean) for index, value in enumerate(values)) / denominator


def _magnitude(reading: SensorReading) -> list[float]:
    return [math.sqrt(x * x + y * y + z * z) for x, y, z in zip(reading.acc_x, reading.acc_y, reading.acc_z)]


def _signal_features(prefix: str, values: Sequence[float]) -> dict[str, float]:
    return {
        f"{prefix}_mean": _mean(values),
        f"{prefix}_std": _std(values),
        f"{prefix}_range": max(values) - min(values),
        f"{prefix}_slope": _slope(values),
    }


def _rms(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def _mean_abs_diff(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return sum(abs(current - previous) for previous, current in zip(values, values[1:])) / (len(values) - 1)


def _local_maxima(values: Sequence[float], threshold: float, minimum_distance: int) -> list[int]:
    """Return deterministic, distance-suppressed local maxima above a threshold."""
    candidates = [
        index for index in range(1, len(values) - 1)
        if values[index] > values[index - 1] and values[index] >= values[index + 1] and values[index] > threshold
    ]
    selected: list[int] = []
    for index in candidates:
        if not selected or index - selected[-1] >= minimum_distance:
            selected.append(index)
        elif values[index] > values[selected[-1]]:
            selected[-1] = index
    return selected


def _sample_rate(reading: SensorReading, modality: str) -> float | None:
    rate = reading.sample_rates_hz.get(modality)
    return float(rate) if rate is not None and rate > 0 else None


def _estimated_heart_rate(reading: SensorReading) -> float | None:
    """Estimate BPM from regular BVP maxima, returning None for unreliable windows."""
    rate = _sample_rate(reading, "bvp")
    values = reading.bvp
    if rate is None or len(values) < round(rate * 5) or _std(values) == 0:
        return None
    threshold = _median(values) + 0.25 * _std(values)
    peaks = _local_maxima(values, threshold, max(1, round(rate * 0.30)))
    intervals = [(right - left) / rate for left, right in zip(peaks, peaks[1:])]
    valid_intervals = [interval for interval in intervals if 0.30 <= interval <= 2.0]
    if len(valid_intervals) < 3:
        return None
    median_interval = _median(valid_intervals)
    interval_spread = _std(valid_intervals)
    if interval_spread / median_interval > 0.20:
        return None
    return 60.0 / median_interval


def _eda_peak_features(reading: SensorReading) -> dict[str, float]:
    """Count separated EDA maxima above a median/MAD-derived robust threshold."""
    values = reading.eda
    baseline = _median(values)
    mad = _median([abs(value - baseline) for value in values])
    # The small absolute floor avoids treating quantisation noise as a response.
    threshold = baseline + max(0.05, 3.0 * mad)
    rate = _sample_rate(reading, "eda") or 1.0
    peaks = _local_maxima(values, threshold, max(1, round(rate)))
    amplitudes = [values[index] - baseline for index in peaks]
    return {
        "eda_peak_count": float(len(peaks)),
        "eda_peak_mean_amplitude": _mean(amplitudes) if amplitudes else 0.0,
        "eda_peak_max_amplitude": max(amplitudes) if amplitudes else 0.0,
    }


def extract_window_features(reading: SensorReading) -> dict[str, float]:
    """Extract deterministic baseline stress-model features from one raw window."""
    features: dict[str, float] = {}
    features.update(_signal_features("bvp", reading.bvp))
    features["bvp_rms"] = _rms(reading.bvp)
    heart_rate = _estimated_heart_rate(reading)
    if heart_rate is not None:
        features["bvp_estimated_heart_rate_bpm"] = heart_rate

    features.update(_signal_features("eda", reading.eda))
    features.update(_eda_peak_features(reading))
    features.update(_signal_features("skin_temperature", reading.skin_temperature))
    for axis, values in (("acc_x", reading.acc_x), ("acc_y", reading.acc_y), ("acc_z", reading.acc_z)):
        features.update(_signal_features(axis, values))
    magnitude = _magnitude(reading)
    features.update(_signal_features("acc_magnitude", magnitude))
    features["acc_magnitude_rms"] = _rms(magnitude)
    features["acc_magnitude_mean_abs_diff"] = _mean_abs_diff(magnitude)
    return features


DEFAULT_FEATURE_ORDER: tuple[str, ...] = (
    "bvp_mean", "bvp_std", "bvp_range", "bvp_slope", "bvp_rms",
    "eda_mean", "eda_std", "eda_range", "eda_slope",
    "eda_peak_count", "eda_peak_mean_amplitude", "eda_peak_max_amplitude",
    "skin_temperature_mean", "skin_temperature_std", "skin_temperature_range", "skin_temperature_slope",
    "acc_x_mean", "acc_x_std", "acc_y_mean", "acc_y_std", "acc_z_mean", "acc_z_std",
    "acc_magnitude_mean", "acc_magnitude_std", "acc_magnitude_range",
    "acc_magnitude_rms", "acc_magnitude_mean_abs_diff",
)

FEATURE_EXTRACTORS: dict[str, Callable[[SensorReading], float]] = {
    name: (lambda reading, feature_name=name: extract_window_features(reading)[feature_name])
    for name in (
        "bvp_mean", "bvp_std", "bvp_range", "bvp_slope",
        "bvp_rms", "bvp_estimated_heart_rate_bpm",
        "eda_mean", "eda_std", "eda_range", "eda_slope",
        "eda_peak_count", "eda_peak_mean_amplitude", "eda_peak_max_amplitude",
        "skin_temperature_mean", "skin_temperature_std", "skin_temperature_range", "skin_temperature_slope",
        "acc_x_mean", "acc_x_std", "acc_x_range", "acc_x_slope",
        "acc_y_mean", "acc_y_std", "acc_y_range", "acc_y_slope",
        "acc_z_mean", "acc_z_std", "acc_z_range", "acc_z_slope",
        "acc_magnitude_mean", "acc_magnitude_std", "acc_magnitude_range", "acc_magnitude_slope",
        "acc_magnitude_rms", "acc_magnitude_mean_abs_diff",
    )
}


@dataclass(frozen=True)
class FeatureVector:
    names: tuple[str, ...]
    values: tuple[float, ...]
    raw: dict[str, float]
    imputed: tuple[str, ...] = ()
