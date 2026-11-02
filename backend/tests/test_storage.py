from datetime import datetime, timedelta, timezone

from app.core.timeutil import utcnow
from app.database.sqlite_repo import SQLiteRepository
from app.preprocessing.features import FeatureVector
from app.schemas.prediction import PredictionResult
from app.schemas.sensor import SensorReading


def make_reading(device="d1", ts=None, **kw):
    ts = ts or utcnow()
    return SensorReading(device_id=device, transport="test", timestamp=ts, received_at=utcnow(), heart_rate=70.0, **kw)


PRED = PredictionResult(label="class_1", class_index=1, model_name="m", is_mock=True)
FV = FeatureVector(names=("heart_rate",), values=(70.0,), raw={"heart_rate": 70.0})


def test_roundtrip_and_ordering():
    repo = SQLiteRepository(":memory:")
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(3):
        repo.save(make_reading(ts=t0 + timedelta(minutes=i)), FV, PRED, "ok", None)
    assert repo.latest_reading().timestamp == t0 + timedelta(minutes=2)
    items, total = repo.list_predictions(limit=2, offset=0)
    assert total == 3 and [p.timestamp for p in items] == [t0 + timedelta(minutes=2), t0 + timedelta(minutes=1)]
    assert items[0].features == {"heart_rate": 70.0} and items[0].is_mock is True


def test_reading_without_prediction_is_stored():
    repo = SQLiteRepository(":memory:")
    rd, pd = repo.save(make_reading(), None, None, "model_unavailable", "no model loaded")
    assert pd is None and repo.latest_prediction() is None
    assert repo.latest_reading().prediction_status == "model_unavailable"


def test_filters_and_pagination():
    repo = SQLiteRepository(":memory:")
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(4):
        repo.save(make_reading(device="a" if i % 2 else "b", ts=t0 + timedelta(hours=i)), FV, PRED, "ok", None)
    assert repo.list_predictions(limit=10, offset=0, device_id="a")[1] == 2
    assert repo.list_predictions(limit=10, offset=0, since=t0 + timedelta(hours=2))[1] == 2
    assert repo.list_predictions(limit=10, offset=0, until=t0 + timedelta(hours=1))[1] == 2
    assert len(repo.list_predictions(limit=1, offset=3)[0]) == 1
