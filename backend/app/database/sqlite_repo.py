"""SQLite repository for raw wearable windows and their derived predictions."""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from app.core.errors import StorageError
from app.core.timeutil import from_iso, to_iso, utcnow
from app.database.base import Repository
from app.preprocessing.features import FeatureVector
from app.schemas.prediction import PredictionResult
from app.schemas.records import StoredPrediction, StoredReading
from app.schemas.sensor import SensorReading

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sensor_readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT NOT NULL, subject_id TEXT, user_id TEXT, session_id TEXT,
    source TEXT NOT NULL, transport TEXT NOT NULL, timestamp TEXT NOT NULL, received_at TEXT NOT NULL,
    window_seconds REAL NOT NULL, sample_rates_json TEXT NOT NULL,
    bvp_json TEXT NOT NULL, eda_json TEXT NOT NULL, skin_temperature_json TEXT NOT NULL,
    acc_x_json TEXT NOT NULL, acc_y_json TEXT NOT NULL, acc_z_json TEXT NOT NULL,
    wesad_label INTEGER, stress_label TEXT, prediction_status TEXT NOT NULL, prediction_error TEXT
);
CREATE INDEX IF NOT EXISTS idx_readings_device_ts ON sensor_readings(device_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_readings_ts ON sensor_readings(timestamp);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reading_id INTEGER NOT NULL UNIQUE REFERENCES sensor_readings(id) ON DELETE CASCADE,
    device_id TEXT NOT NULL, user_id TEXT, session_id TEXT, timestamp TEXT NOT NULL,
    task TEXT NOT NULL, label TEXT, class_index INTEGER, value REAL, confidence REAL,
    probabilities_json TEXT, model_name TEXT NOT NULL, model_version TEXT NOT NULL,
    is_mock INTEGER NOT NULL, features_json TEXT NOT NULL, imputed_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pred_device_ts ON predictions(device_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_pred_ts ON predictions(timestamp);
"""

_READING_COLUMNS = (
    "device_id", "subject_id", "user_id", "session_id", "source", "transport", "timestamp", "received_at",
    "window_seconds", "sample_rates_json", "bvp_json", "eda_json", "skin_temperature_json",
    "acc_x_json", "acc_y_json", "acc_z_json", "wesad_label", "stress_label", "prediction_status", "prediction_error",
)
_REQUIRED_READING_COLUMNS = {
    "subject_id": "TEXT", "window_seconds": "REAL", "sample_rates_json": "TEXT",
    "bvp_json": "TEXT", "eda_json": "TEXT", "skin_temperature_json": "TEXT",
    "acc_x_json": "TEXT", "acc_y_json": "TEXT", "acc_z_json": "TEXT",
    "wesad_label": "INTEGER", "stress_label": "TEXT", "prediction_status": "TEXT", "prediction_error": "TEXT",
}


class SQLiteRepository(Repository):
    def __init__(self, target: str):
        self._lock = threading.RLock()
        try:
            if target != ":memory:":
                Path(target).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(target, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys=ON")
            if target != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._migrate_sensor_readings()
            self._conn.commit()
        except (sqlite3.Error, OSError) as exc:
            log.error("Database initialisation failed: %s", exc)
            raise StorageError("database initialisation failed") from None

    def _migrate_sensor_readings(self) -> None:
        """Add raw-window columns when opening a database created by the scalar prototype."""
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(sensor_readings)")}
        for name, sql_type in _REQUIRED_READING_COLUMNS.items():
            if name not in existing:
                self._conn.execute(f"ALTER TABLE sensor_readings ADD COLUMN {name} {sql_type}")

    def _fail(self, operation: str, exc: Exception) -> StorageError:
        log.error("Database error during %s: %s", operation, exc)
        return StorageError(f"database error during {operation}")

    @staticmethod
    def _reading_from_row(row: sqlite3.Row) -> StoredReading:
        data = dict(row)
        try:
            data.update({
                "timestamp": from_iso(data["timestamp"]), "received_at": from_iso(data["received_at"]),
                "sample_rates_hz": json.loads(data.pop("sample_rates_json")),
                "bvp": json.loads(data.pop("bvp_json")), "eda": json.loads(data.pop("eda_json")),
                "skin_temperature": json.loads(data.pop("skin_temperature_json")),
                "acc_x": json.loads(data.pop("acc_x_json")), "acc_y": json.loads(data.pop("acc_y_json")),
                "acc_z": json.loads(data.pop("acc_z_json")),
            })
        except (TypeError, json.JSONDecodeError) as exc:
            raise StorageError("stored raw window is unreadable") from exc
        return StoredReading(**data)

    @staticmethod
    def _prediction_from_row(row: sqlite3.Row) -> StoredPrediction:
        data = dict(row)
        return StoredPrediction(
            id=data["id"], reading_id=data["reading_id"], device_id=data["device_id"], user_id=data["user_id"],
            session_id=data["session_id"], timestamp=from_iso(data["timestamp"]), task=data["task"],
            label=data["label"], class_index=data["class_index"], value=data["value"], confidence=data["confidence"],
            probabilities=json.loads(data["probabilities_json"]) if data["probabilities_json"] else None,
            model_name=data["model_name"], model_version=data["model_version"], is_mock=bool(data["is_mock"]),
            features=json.loads(data["features_json"]), imputed_features=json.loads(data["imputed_json"]),
            created_at=from_iso(data["created_at"]),
        )

    @staticmethod
    def _where(conditions: Iterable[str]) -> str:
        conditions = list(conditions)
        return f" WHERE {' AND '.join(conditions)}" if conditions else ""

    def save(self, reading: SensorReading, features: Optional[FeatureVector], prediction: Optional[PredictionResult], status: str, error: Optional[str]) -> tuple[StoredReading, Optional[StoredPrediction]]:
        if not reading.device_id:
            raise StorageError("device_id is required")
        values: dict[str, Any] = {
            "device_id": reading.device_id, "subject_id": reading.subject_id, "user_id": reading.user_id,
            "session_id": reading.session_id, "source": reading.source, "transport": reading.transport,
            "timestamp": to_iso(reading.timestamp), "received_at": to_iso(reading.received_at),
            "window_seconds": reading.window_seconds, "sample_rates_json": json.dumps(reading.sample_rates_hz),
            "bvp_json": json.dumps(reading.bvp), "eda_json": json.dumps(reading.eda),
            "skin_temperature_json": json.dumps(reading.skin_temperature), "acc_x_json": json.dumps(reading.acc_x),
            "acc_y_json": json.dumps(reading.acc_y), "acc_z_json": json.dumps(reading.acc_z),
            "wesad_label": reading.wesad_label, "stress_label": reading.stress_label,
            "prediction_status": status, "prediction_error": error,
        }
        with self._lock:
            try:
                with self._conn:
                    placeholders = ", ".join("?" for _ in _READING_COLUMNS)
                    cursor = self._conn.execute(
                        f"INSERT INTO sensor_readings ({', '.join(_READING_COLUMNS)}) VALUES ({placeholders})",
                        [values[column] for column in _READING_COLUMNS],
                    )
                    reading_id = cursor.lastrowid
                    stored_reading = StoredReading(
                        id=reading_id,
                        **reading.model_dump(exclude={"id"}),
                        prediction_status=status,
                        prediction_error=error,
                    )
                    stored_prediction = None
                    if prediction is not None and features is not None:
                        created_at = utcnow()
                        cursor = self._conn.execute(
                            "INSERT INTO predictions (reading_id,device_id,user_id,session_id,timestamp,task,label,class_index,value,confidence,probabilities_json,model_name,model_version,is_mock,features_json,imputed_json,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (reading_id, reading.device_id, reading.user_id, reading.session_id, to_iso(reading.timestamp), prediction.task,
                             prediction.label, prediction.class_index, prediction.value, prediction.confidence,
                             json.dumps(prediction.probabilities) if prediction.probabilities else None, prediction.model_name,
                             prediction.model_version, int(prediction.is_mock), json.dumps(features.raw), json.dumps(list(features.imputed)), to_iso(created_at)),
                        )
                        stored_prediction = StoredPrediction(id=cursor.lastrowid, reading_id=reading_id, device_id=reading.device_id,
                            user_id=reading.user_id, session_id=reading.session_id, timestamp=reading.timestamp, features=features.raw,
                            imputed_features=list(features.imputed), created_at=created_at, **prediction.model_dump())
                return stored_reading, stored_prediction
            except sqlite3.Error as exc:
                raise self._fail("save", exc) from None

    def latest_reading(self, device_id: Optional[str] = None) -> Optional[StoredReading]:
        conditions, parameters = (["device_id = ?"], [device_id]) if device_id else ([], [])
        with self._lock:
            try:
                row = self._conn.execute("SELECT * FROM sensor_readings" + self._where(conditions) + " ORDER BY timestamp DESC, id DESC LIMIT 1", parameters).fetchone()
            except sqlite3.Error as exc:
                raise self._fail("latest_reading", exc) from None
        return self._reading_from_row(row) if row else None

    def latest_prediction(self, device_id: Optional[str] = None) -> Optional[StoredPrediction]:
        predictions, _ = self.list_predictions(limit=1, offset=0, device_id=device_id)
        return predictions[0] if predictions else None

    def list_predictions(self, *, limit: int, offset: int, device_id: Optional[str] = None, since: Optional[datetime] = None, until: Optional[datetime] = None) -> tuple[list[StoredPrediction], int]:
        conditions: list[str] = []; parameters: list[Any] = []
        if device_id: conditions.append("device_id = ?"); parameters.append(device_id)
        if since: conditions.append("timestamp >= ?"); parameters.append(to_iso(since))
        if until: conditions.append("timestamp <= ?"); parameters.append(to_iso(until))
        where = self._where(conditions)
        with self._lock:
            try:
                total = self._conn.execute(f"SELECT COUNT(*) FROM predictions{where}", parameters).fetchone()[0]
                rows = self._conn.execute(f"SELECT * FROM predictions{where} ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?", [*parameters, limit, offset]).fetchall()
            except sqlite3.Error as exc:
                raise self._fail("list_predictions", exc) from None
        return [self._prediction_from_row(row) for row in rows], total

    def aggregate(self, since: datetime, device_id: Optional[str] = None) -> dict[str, Any]:
        conditions, params = (["device_id = ?"], [device_id]) if device_id else ([], [])
        window_conditions, window_params = [*conditions, "timestamp >= ?"], [*params, to_iso(since)]
        with self._lock:
            try:
                query = self._conn.execute
                rw, pw = self._where(window_conditions), self._where(window_conditions)
                reading_window = query(f"SELECT COUNT(*) n, MIN(timestamp) first_ts, MAX(timestamp) last_ts FROM sensor_readings{rw}", window_params).fetchone()
                prediction_window = query(f"SELECT COUNT(*) n, COALESCE(SUM(is_mock), 0) mock_n FROM predictions{pw}", window_params).fetchone()
                return {
                    "readings_total": query(f"SELECT COUNT(*) FROM sensor_readings{self._where(conditions)}", params).fetchone()[0],
                    "predictions_total": query(f"SELECT COUNT(*) FROM predictions{self._where(conditions)}", params).fetchone()[0],
                    "readings_in_window": reading_window["n"], "predictions_in_window": prediction_window["n"],
                    "first_reading_at": from_iso(reading_window["first_ts"]) if reading_window["first_ts"] else None,
                    "last_reading_at": from_iso(reading_window["last_ts"]) if reading_window["last_ts"] else None,
                    "devices": [row[0] for row in query(f"SELECT DISTINCT device_id FROM sensor_readings{self._where(conditions)} ORDER BY 1", params)],
                    "prediction_status_counts": dict(query(f"SELECT prediction_status, COUNT(*) FROM sensor_readings{rw} GROUP BY 1", window_params).fetchall()),
                    "label_counts": dict(query(f"SELECT label, COUNT(*) FROM predictions{self._where([*window_conditions, 'label IS NOT NULL'])} GROUP BY 1", window_params).fetchall()),
                    "reading_sources": dict(query(f"SELECT source, COUNT(*) FROM sensor_readings{rw} GROUP BY 1", window_params).fetchall()),
                    "includes_mock_predictions": prediction_window["mock_n"] > 0,
                    "feature_averages": {}, "value_stats": None,
                }
            except sqlite3.Error as exc:
                raise self._fail("aggregate", exc) from None

    def ping(self) -> bool:
        try:
            with self._lock: self._conn.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False

    def close(self) -> None:
        with self._lock:
            try: self._conn.close()
            except sqlite3.Error: pass
