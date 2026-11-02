"""HTTP layer only: parse request -> call service/repository -> shape response."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException, Query, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from app import __version__
from app.container import Container
from app.core.errors import SensorValidationError
from app.core.timeutil import utcnow
from app.schemas.records import LiveTelemetry, PredictionPage, ProcessingResult, StoredPrediction, StoredReading, Summary
from app.services.summary import build_summary

router = APIRouter()

_EXAMPLE = {
    "device_id": "wearable-001",
    "subject_id": "S2",
    "user_id": "user-1",
    "session_id": "session-1",
    "timestamp": "2026-01-01T12:00:00Z",
    "window_seconds": 60,
    "sample_rates_hz": {"bvp": 64, "eda": 4, "skin_temperature": 4, "acc_x": 32, "acc_y": 32, "acc_z": 32},
    "bvp": [0.0], "eda": [0.0], "skin_temperature": [33.5],
    "acc_x": [0.01], "acc_y": [0.02], "acc_z": [0.98],
}


def _c(request: Request) -> Container:
    return request.app.state.container


@router.get("/health", tags=["system"])
def health(request: Request):
    c = _c(request)
    db_ok = c.repo.ping()
    mqtt_state = {
        "enabled": c.mqtt is not None,
        "connected": bool(c.mqtt and c.mqtt.connected),
        "broker": f"{c.settings.mqtt_broker_host}:{c.settings.mqtt_broker_port}" if c.mqtt else None,
        "topics": c.settings.mqtt_topic_list if c.mqtt else [],
        "subscribed_topics": list(c.mqtt.subscribed_topics) if c.mqtt else [],
        "subscription_acknowledged": bool(c.mqtt and c.mqtt.subscription_acknowledged),
        "received_messages": c.mqtt.received_messages if c.mqtt else 0,
        "callback": "pipeline.ingest_mqtt_message" if c.mqtt else None,
    }
    inference = c.inference.status()
    degraded = (not inference["model_loaded"]) or (mqtt_state["enabled"] and not mqtt_state["connected"])
    body = {
        "status": "unavailable" if not db_ok else ("degraded" if degraded else "ok"),
        "version": __version__,
        "environment": c.settings.app_env,
        "time": utcnow(),
        "uptime_seconds": round((utcnow() - c.started_at).total_seconds(), 1),
        "database": {"ok": db_ok},
        "mqtt": mqtt_state,
        "inference": inference,
        "pipeline": c.pipeline.diagnostics(),
    }
    return JSONResponse(jsonable_encoder(body), status_code=200 if db_ok else 503)


@router.get("/api/sensor/latest", response_model=StoredReading, tags=["sensor"])
def latest_sensor(request: Request, device_id: Optional[str] = None):
    reading = _c(request).repo.latest_reading(device_id)
    if reading is None:
        raise HTTPException(404, "No sensor readings available")
    return reading


@router.get("/api/sensor/live", response_model=LiveTelemetry, tags=["sensor"])
def live_sensor(request: Request, device_id: Optional[str] = None):
    """Latest accepted MQTT sample. Independent of the completed 60-second analysis window."""
    return _c(request).pipeline.live_telemetry(device_id)


@router.post(
    "/api/sensor/data",
    response_model=ProcessingResult,
    status_code=status.HTTP_201_CREATED,
    tags=["sensor"],
    summary="Submit a sensor reading over HTTP (same pipeline as MQTT; for testing)",
)
def post_sensor_data(request: Request, payload: dict[str, Any] = Body(..., examples=[_EXAMPLE])):
    try:
        outcome = _c(request).pipeline.ingest(payload, transport="http")
    except SensorValidationError as exc:
        return JSONResponse(status_code=422, content={"detail": "Invalid sensor payload", "errors": exc.errors})
    return ProcessingResult(
        status=outcome.status, error=outcome.error, reading=outcome.reading, prediction=outcome.prediction
    )


@router.get("/api/predictions/latest", response_model=StoredPrediction, tags=["predictions"])
def latest_prediction(request: Request, device_id: Optional[str] = None):
    pred = _c(request).repo.latest_prediction(device_id)
    if pred is None:
        raise HTTPException(404, "No predictions available")
    return pred


@router.get("/api/predictions", response_model=PredictionPage, tags=["predictions"])
def list_predictions(
    request: Request,
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    device_id: Optional[str] = None,
    since: Optional[datetime] = Query(None, description="ISO-8601, inclusive"),
    until: Optional[datetime] = Query(None, description="ISO-8601, inclusive"),
):
    items, total = _c(request).repo.list_predictions(
        limit=limit, offset=offset, device_id=device_id, since=since, until=until
    )
    return PredictionPage(items=items, total=total, limit=limit, offset=offset)


@router.get("/api/summary", response_model=Summary, tags=["dashboard"])
def summary(
    request: Request,
    window_minutes: int = Query(60, ge=1, le=60 * 24 * 30),
    device_id: Optional[str] = None,
):
    return build_summary(_c(request).repo, window_minutes, device_id)
