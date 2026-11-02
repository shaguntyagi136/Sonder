"""SQLite implementation of Storage (stdlib only, development-friendly)."""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from app.core.errors import StorageError
from app.database.base import Storage
from app.models.prediction import PredictionResult
from app.models.sensor import FEATURE_CATALOG, SensorReading

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sensor_readings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  device_id TEXT NOT NULL, user_id TEXT, session_id TEXT,
  timestamp TEXT NOT NULL, received_at TEXT NOT NULL, source TEXT NOT NULL,
  heart_rate REAL, hrv REAL, spo2 REAL, skin_temp REAL,
  accel_x REAL, accel_y REAL, accel_z REAL, accel_magnitude REAL
);
CREATE INDEX IF NOT EXISTS idx_readings_ts ON sensor_readings(timestamp);
CREATE INDEX IF NOT EXISTS idx_readings_dev_ts ON sensor_readings(device_id, timestamp);

CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  reading_id INTEGER NOT NULL REFERENCES sensor_readings(id) ON DELETE CASCADE,
  device_id TEXT NOT NULL, user_id TEXT, session_id TEXT,
  timestamp TEXT NOT NULL, created_at TEXT NOT NULL,
  value TEXT NOT NULL, label TEXT NOT NULL, probabilities TEXT, confidence REAL,
  model_name TEXT NOT NULL, model_version TEXT, is_mock INTEGER NOT NULL,
  latency_ms REAL, feature_names TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pred_ts ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_pred_dev_ts ON predictions(device_id, timestamp);
"""

_TS_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"  # fixed-width UTC -> lexicographic order == time order


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime(_TS_FMT)


def _parse(s: str) -> datetime:
    return datetime.strptime(s, _TS_FMT).replace(tzinfo=timezone.utc)


def _where(device_id, since, until=None):
    clauses, params = [], []
    if device_id:
        clauses.append("device_id = ?")
        params.append(device_id)
    if since:
        clauses.append("timestamp >= ?")
        params.append(_iso(since))
    if until:
        clauses.append("timestamp <= ?")
        params.append(_iso(until))
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


class SQLiteStorage(Storage):
    def __init__(self, path: str = "data/stress.db"):
        self._path = path
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None

    # -- lifecycle --------------------------------------------------------
    def init(self) -> None:
        try:
            in_memory = self._path == ":memory:"
            if not in_memory:
                Path(self._path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self._path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            if not in_memory:
                conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
            conn.commit()
            self._conn = conn
            logger.info("SQLite storage ready (%s)", "in-memory" if in_memory else self._path)
        except (sqlite3.Error, OSError) as exc:
            logger.error("Database initialisation failed: %s", exc)
            raise StorageError("could not initialise database") from exc

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    @contextmanager
    def _use(self):
        if self._conn is None:
            raise StorageError("storage not initialised")
        with self._lock:
            try:
                with self._conn as conn:  # commit on success, rollback on error
                    yield conn
            except sqlite3.Error as exc:
                logger.error("Database error: %s", exc)
                raise StorageError("database operation failed") from exc

    def ping(self) -> bool:
        try:
            with self._use() as c:
                c.execute("SELECT 1")
            return True
        except StorageError:
            return False

    # -- writes -----------------------------------------------------------
    def save_record(self, reading, prediction=None):
        with self._use() as c:
            cur = c.execute(
                "INSERT INTO sensor_readings (device_id,user_id,session_id,timestamp,received_at,source,"
                "heart_rate,hrv,spo2,skin_temp,accel_x,accel_y,accel_z,accel_magnitude) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (reading.device_id, reading.user_id, reading.session_id, _iso(reading.timestamp),
                 _iso(reading.received_at), reading.source, reading.heart_rate, reading.hrv,
                 reading.spo2, reading.skin_temp, reading.accel_x, reading.accel_y, reading.accel_z,
                 reading.accel_magnitude),
            )
            rid = cur.lastrowid
            saved_pred = None
            if prediction is not None:
                cur = c.execute(
                    "INSERT INTO predictions (reading_id,device_id,user_id,session_id,timestamp,created_at,"
                    "value,label,probabilities,confidence,model_name,model_version,is_mock,latency_ms,"
                    "feature_names) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (rid, prediction.device_id, prediction.user_id, prediction.session_id,
                     _iso(prediction.timestamp), _iso(prediction.created_at), prediction.value,
                     prediction.label,
                     json.dumps(prediction.probabilities) if prediction.probabilities else None,
                     prediction.confidence, prediction.model_name, prediction.model_version,
                     int(prediction.is_mock), prediction.latency_ms,
                     json.dumps(list(prediction.feature_names))),
                )
                saved_pred = replace(prediction, id=cur.lastrowid, reading_id=rid)
        return replace(reading, id=rid), saved_pred

    # -- row mapping ------------------------------------------------------
    @staticmethod
    def _reading(row) -> SensorReading:
        return SensorReading(
            id=row["id"], device_id=row["device_id"], user_id=row["user_id"],
            session_id=row["session_id"], timestamp=_parse(row["timestamp"]),
            received_at=_parse(row["received_at"]), source=row["source"],
            **{f: row[f] for f in FEATURE_CATALOG},
        )

    @staticmethod
    def _prediction(row) -> PredictionResult:
        return PredictionResult(
            id=row["id"], reading_id=row["reading_id"], device_id=row["device_id"],
            user_id=row["user_id"], session_id=row["session_id"],
            timestamp=_parse(row["timestamp"]), created_at=_parse(row["created_at"]),
            value=row["value"], label=row["label"],
            probabilities=json.loads(row["probabilities"]) if row["probabilities"] else None,
            confidence=row["confidence"], model_name=row["model_name"],
            model_version=row["model_version"], is_mock=bool(row["is_mock"]),
            latency_ms=row["latency_ms"], feature_names=tuple(json.loads(row["feature_names"])),
        )

    # -- reads ------------------------------------------------------------
    def latest_reading(self, device_id=None):
        where, params = _where(device_id, None)
        with self._use() as c:
            row = c.execute(f"SELECT * FROM sensor_readings{where} "
                            "ORDER BY timestamp DESC, id DESC LIMIT 1", params).fetchone()
        return self._reading(row) if row else None

    def latest_prediction(self, device_id=None):
        where, params = _where(device_id, None)
        with self._use() as c:
            row = c.execute(f"SELECT * FROM predictions{where} "
                            "ORDER BY timestamp DESC, id DESC LIMIT 1", params).fetchone()
        return self._prediction(row) if row else None

    def list_predictions(self, *, limit=50, offset=0, device_id=None, since=None, until=None):
        where, params = _where(device_id, since, until)
        with self._use() as c:
            total = c.execute(f"SELECT COUNT(*) FROM predictions{where}", params).fetchone()[0]
            rows = c.execute(
                f"SELECT * FROM predictions{where} ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?",
                [*params, limit, offset]).fetchall()
        return [self._prediction(r) for r in rows], total

    def count_readings(self, since=None, device_id=None):
        where, params = _where(device_id, since)
        with self._use() as c:
            return c.execute(f"SELECT COUNT(*) FROM sensor_readings{where}", params).fetchone()[0]

    def count_predictions(self, since=None, device_id=None):
        where, params = _where(device_id, since)
        with self._use() as c:
            return c.execute(f"SELECT COUNT(*) FROM predictions{where}", params).fetchone()[0]

    def prediction_label_counts(self, since=None, device_id=None):
        where, params = _where(device_id, since)
        with self._use() as c:
            rows = c.execute(f"SELECT label, COUNT(*) AS n FROM predictions{where} "
                             "GROUP BY label ORDER BY n DESC, label", params).fetchall()
        return [(r["label"], r["n"]) for r in rows]

    def feature_averages(self, since=None, device_id=None):
        where, params = _where(device_id, since)
        cols = ", ".join(f"AVG({f}) AS {f}" for f in FEATURE_CATALOG)  # constants, not user input
        with self._use() as c:
            row = c.execute(f"SELECT {cols} FROM sensor_readings{where}", params).fetchone()
        return {f: float(row[f]) for f in FEATURE_CATALOG if row[f] is not None}

    def list_devices(self):
        with self._use() as c:
            rows = c.execute("SELECT DISTINCT device_id FROM sensor_readings ORDER BY device_id").fetchall()
        return [r["device_id"] for r in rows]
