
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from statistics import median


FEATURE_ORDER = (
    "bvp_mean", "bvp_std", "bvp_range", "bvp_slope", "bvp_rms",
    "eda_mean", "eda_std", "eda_range", "eda_slope",
    "eda_peak_count", "eda_peak_mean_amplitude", "eda_peak_max_amplitude",
    "skin_temperature_mean", "skin_temperature_std",
    "skin_temperature_range", "skin_temperature_slope",
    "acc_x_mean", "acc_x_std", "acc_y_mean", "acc_y_std",
    "acc_z_mean", "acc_z_std",
    "acc_magnitude_mean", "acc_magnitude_std",
    "acc_magnitude_range", "acc_magnitude_rms",
    "acc_magnitude_mean_abs_diff",
)

REQUIRED_SIGNALS = (
    "bvp", "eda", "skin_temperature", "acc_x", "acc_y", "acc_z",
)

# These are the exact sample counts in the model's 60-second training windows.
EXPECTED_SAMPLE_COUNTS = {
    "bvp": 64 * 60,
    "eda": 4 * 60,
    "skin_temperature": 4 * 60,
    "acc_x": 32 * 60,
    "acc_y": 32 * 60,
    "acc_z": 32 * 60,
}


def _mean(values):
    return sum(values) / len(values)


def _std(values):
    avg = _mean(values)
    return math.sqrt(sum((v - avg) ** 2 for v in values) / len(values))


def _slope(values):
    if len(values) < 2:
        return 0.0

    x_mean = (len(values) - 1) / 2
    y_mean = _mean(values)

    denominator = sum((i - x_mean) ** 2 for i in range(len(values)))
    numerator = sum(
        (i - x_mean) * (v - y_mean)
        for i, v in enumerate(values)
    )
    return numerator / denominator


def _rms(values):
    return math.sqrt(sum(v * v for v in values) / len(values))


def _signal_features(prefix, values):
    return {
        f"{prefix}_mean": _mean(values),
        f"{prefix}_std": _std(values),
        f"{prefix}_range": max(values) - min(values),
        f"{prefix}_slope": _slope(values),
    }


def _mean_abs_diff(values):
    if len(values) < 2:
        return 0.0

    differences = [
        abs(current - previous)
        for previous, current in zip(values, values[1:])
    ]
    return _mean(differences)


def _local_maxima(values, threshold, minimum_distance):
    candidates = [
        i for i in range(1, len(values) - 1)
        if (
            values[i] > values[i - 1]
            and values[i] >= values[i + 1]
            and values[i] > threshold
        )
    ]

    selected = []

    for index in candidates:
        if not selected or index - selected[-1] >= minimum_distance:
            selected.append(index)
        elif values[index] > values[selected[-1]]:
            selected[-1] = index

    return selected


def _eda_peak_features(values, sample_rate=4):
    baseline = median(values)
    mad = median([abs(v - baseline) for v in values])

    threshold = baseline + max(0.05, 3.0 * mad)

    peaks = _local_maxima(
        values,
        threshold,
        max(1, round(sample_rate))
    )

    amplitudes = [values[i] - baseline for i in peaks]

    return {
        "eda_peak_count": float(len(peaks)),
        "eda_peak_mean_amplitude": (
            _mean(amplitudes) if amplitudes else 0.0
        ),
        "eda_peak_max_amplitude": (
            max(amplitudes) if amplitudes else 0.0
        ),
    }


def extract_window_features(window: Mapping[str, object]):
    """Extract the ordered model features from a completed WindowManager handoff.

    The ``reading`` member of WindowManager's completed-window payload is also
    accepted for existing callers.  In either form every signal must have the
    exact sample count used to train the deployed 60-second model.
    """

    if not isinstance(window, Mapping):
        raise ValueError("sensor window must be a mapping")

    if "reading" in window:
        if window.get("window_duration_seconds") != 60:
            raise ValueError("completed window must be 60 seconds")
        reading = window["reading"]
        if not isinstance(reading, Mapping):
            raise ValueError("completed window reading must be a mapping")
    else:
        reading = window

    missing = [key for key in REQUIRED_SIGNALS if key not in reading]
    if missing:
        raise ValueError(f"Missing sensor signals: {missing}")

    signals = {}
    for name in REQUIRED_SIGNALS:
        samples = reading[name]
        if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
            raise ValueError(f"{name} must be a sequence of numeric samples")
        if len(samples) != EXPECTED_SAMPLE_COUNTS[name]:
            raise ValueError(
                f"{name} must contain exactly {EXPECTED_SAMPLE_COUNTS[name]} samples"
            )
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in samples):
            raise ValueError(f"{name} contains a non-numeric sample")
        values = [float(value) for value in samples]

        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"{name} contains NaN or infinite values")
        signals[name] = values

    acc_lengths = {
        len(signals["acc_x"]),
        len(signals["acc_y"]),
        len(signals["acc_z"]),
    }

    if len(acc_lengths) != 1:
        raise ValueError("ACC axes must have equal sample counts")

    features = {}

    # BVP: 5 features
    bvp = signals["bvp"]
    features.update(_signal_features("bvp", bvp))
    features["bvp_rms"] = _rms(bvp)

    # EDA: 7 features
    eda = signals["eda"]
    features.update(_signal_features("eda", eda))
    features.update(_eda_peak_features(eda))

    # Skin temperature: 4 features
    temperature = signals["skin_temperature"]
    features.update(
        _signal_features("skin_temperature", temperature)
    )

    # ACC axes: 6 features
    for axis in ("acc_x", "acc_y", "acc_z"):
        features.update(
            _signal_features(axis, signals[axis])
        )

    # ACC magnitude: 5 features
    magnitude = [
        math.sqrt(x*x + y*y + z*z)
        for x, y, z in zip(
            signals["acc_x"],
            signals["acc_y"],
            signals["acc_z"]
        )
    ]

    features.update(_signal_features("acc_magnitude", magnitude))
    features["acc_magnitude_rms"] = _rms(magnitude)
    features["acc_magnitude_mean_abs_diff"] = _mean_abs_diff(magnitude)

    # Preserve the exact order expected by the trained model.
    ordered_features = {
        name: float(features[name])
        for name in FEATURE_ORDER
    }

    if not all(math.isfinite(v) for v in ordered_features.values()):
        raise ValueError("Extracted features contain invalid values")

    return ordered_features
