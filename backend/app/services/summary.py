"""Dashboard aggregation over the active Repository interface."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.database.base import Repository
from app.schemas.records import Summary, ValueStats


def build_summary(repository: Repository, window_minutes: int, device_id: str | None) -> Summary:
    now = datetime.now(timezone.utc)
    since = now - timedelta(minutes=window_minutes)
    data = repository.aggregate(since, device_id)
    stats = data["value_stats"]
    return Summary(
        generated_at=now, window_minutes=window_minutes, window_start=since, device_id=device_id,
        readings_total=data["readings_total"], predictions_total=data["predictions_total"],
        readings_in_window=data["readings_in_window"], predictions_in_window=data["predictions_in_window"],
        first_reading_at=data["first_reading_at"], last_reading_at=data["last_reading_at"], devices=data["devices"],
        prediction_status_counts=data["prediction_status_counts"], label_counts=data["label_counts"],
        reading_sources=data["reading_sources"], includes_mock_predictions=data["includes_mock_predictions"],
        feature_averages=data["feature_averages"],
        prediction_value_stats=ValueStats(**stats) if stats else None,
        latest_prediction=repository.latest_prediction(device_id),
    )
