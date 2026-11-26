"""StressSense MQTT backend and live event API."""

from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .mqtt_handler import MQTTConfig, MQTTHandler

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class EventBroadcaster:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        async with self._lock:
            self._subscribers.add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue) -> None:
        async with self._lock:
            self._subscribers.discard(queue)

    def publish_from_thread(self, event_type: str, payload: dict) -> None:
        event = {
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device_id": payload.get("device_id"),
            "payload": payload,
        }
        loop = _event_loop
        if loop and loop.is_running():
            loop.call_soon_threadsafe(self._enqueue, event)

    def _enqueue(self, event: dict) -> None:
        for queue in tuple(self._subscribers):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(event)


broadcaster = EventBroadcaster()
_event_loop: asyncio.AbstractEventLoop | None = None
mqtt_handler: MQTTHandler | None = None
_latest_prediction: dict | None = None
_prediction_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _event_loop, mqtt_handler, _latest_prediction
    _event_loop = asyncio.get_running_loop()
    with _prediction_lock:
        _latest_prediction = None

    def on_status(event: dict) -> None:
        broadcaster.publish_from_thread("processing_status", event)
        if mqtt_handler:
            mqtt_handler.publish_status(event)

    def on_prediction(event: dict) -> None:
        global _latest_prediction
        with _prediction_lock:
            _latest_prediction = event.copy()
        broadcaster.publish_from_thread("prediction_update", event)
        if mqtt_handler:
            mqtt_handler.publish_prediction(event)

    try:
        mqtt_handler = MQTTHandler(
            MQTTConfig.from_env(),
            status_callback=on_status,
            prediction_callback=on_prediction,
            sensor_callback=lambda event: broadcaster.publish_from_thread("sensor_update", event),
        )
        mqtt_handler.start()
    except Exception:
        LOGGER.exception("Backend startup could not connect to MQTT")
        mqtt_handler = None
    try:
        yield
    finally:
        if mqtt_handler:
            mqtt_handler.stop()
            mqtt_handler = None
        with _prediction_lock:
            _latest_prediction = None


app = FastAPI(title="StressSense Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "https://sonder-frontend-psi.vercel.app",
    ],
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok" if mqtt_handler and mqtt_handler.connected else "degraded",
        "mqtt_connected": bool(mqtt_handler and mqtt_handler.connected),
        "mqtt_error": mqtt_handler.last_error if mqtt_handler else "not initialized",
    }


@app.get("/api/predictions/latest")
async def latest_prediction() -> dict:
    with _prediction_lock:
        prediction = _latest_prediction.copy() if _latest_prediction else None
    if prediction is None:
        return {"prediction": None}
    return prediction


@app.websocket("/ws/live")
async def live(websocket: WebSocket) -> None:
    await websocket.accept()
    queue = await broadcaster.subscribe()
    broadcaster.publish_from_thread("connection_status", {"status": "connected"})
    try:
        while True:
            await websocket.send_json(await queue.get())
    except WebSocketDisconnect:
        LOGGER.info("WebSocket client disconnected")
    finally:
        await broadcaster.unsubscribe(queue)


FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
