"""Raw wearable windows -> ordered vectors and subject-aware tabular ML data."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from app.core.errors import PreprocessingError
from app.preprocessing.features import DEFAULT_FEATURE_ORDER, FeatureVector, extract_window_features
from app.preprocessing.spec import ModelSpec
from app.schemas.sensor import SensorReading


@dataclass(frozen=True)
class TabularRow:
    """One labelled ML example with its subject retained as a group identifier."""

    subject_id: str
    timestamp: datetime
    stress_label: str
    wesad_label: int | None
    values: tuple[float, ...]
    raw_features: dict[str, float]


@dataclass(frozen=True)
class TabularDataset:
    """In-memory tabular dataset; ``X``, ``y``, and ``groups`` stay aligned by row."""

    feature_names: tuple[str, ...]
    rows: tuple[TabularRow, ...]

    @property
    def X(self) -> tuple[tuple[float, ...], ...]:
        return tuple(row.values for row in self.rows)

    @property
    def y(self) -> tuple[str, ...]:
        return tuple(row.stress_label for row in self.rows)

    @property
    def groups(self) -> tuple[str, ...]:
        return tuple(row.subject_id for row in self.rows)


class Preprocessor:
    def __init__(self, spec: ModelSpec, missing_feature_policy: str = "reject"):
        self.spec = spec
        self.missing_feature_policy = missing_feature_policy

    def transform(self, reading: SensorReading) -> FeatureVector:
        extracted = extract_window_features(reading)
        raw: dict[str, float] = {}
        imputed: list[str] = []
        missing: list[str] = []

        for name in self.spec.feature_order:
            value = extracted.get(name)
            if value is None and self.missing_feature_policy == "impute" and name in self.spec.impute:
                value = self.spec.impute[name]
                imputed.append(name)
            if value is None:
                missing.append(name)
            else:
                raw[name] = float(value)

        if missing:
            raise PreprocessingError(f"required extracted features are unavailable: {', '.join(missing)}")

        values = tuple(raw[name] for name in self.spec.feature_order)
        if self.spec.normalization:
            values = tuple(
                (value - self.spec.normalization.mean[name]) / self.spec.normalization.std[name]
                for name, value in zip(self.spec.feature_order, values)
            )
        return FeatureVector(tuple(self.spec.feature_order), values, raw, tuple(imputed))


def build_tabular_dataset(
    windows: Iterable[SensorReading],
    spec: ModelSpec | None = None,
) -> TabularDataset:
    """Convert labelled WESAD windows into deterministic, group-preserving ML rows.

    Unlabelled windows are unsuitable for the initial supervised binary task and are
    excluded. No random split, shuffle, or subject aggregation occurs here.
    """
    spec = spec or ModelSpec(model_name="wesad-tabular-baseline", feature_order=list(DEFAULT_FEATURE_ORDER))
    preprocessor = Preprocessor(spec)
    rows: list[TabularRow] = []
    for window in windows:
        if window.stress_label is None:
            continue
        if window.subject_id is None:
            raise PreprocessingError("labelled ML windows require subject_id")
        vector = preprocessor.transform(window)
        rows.append(
            TabularRow(
                subject_id=window.subject_id,
                timestamp=window.timestamp,
                stress_label=window.stress_label,
                wesad_label=window.wesad_label,
                values=vector.values,
                raw_features=vector.raw,
            )
        )
    return TabularDataset(tuple(spec.feature_order), tuple(rows))
