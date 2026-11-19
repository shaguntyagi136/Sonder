"""MQTT ingestion, validation, publication, and connection lifecycle."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import paho.mqtt.client as mqtt
from dotenv import load_dotenv
from paho.mqtt.enums import CallbackAPIVersion

from .window_manager import MODALITIES, WindowManager, parse_timestamp

LOGGER = logging.getLogger(__name__)
SENSOR_TOPIC = "stresssense/v1/devices/+/sensors/+"
# Canonical sensor message schema (JSON payload on SENSOR_TOPIC):
# {
#   "device_id": "device-01",             # 1-64: A-Z, a-z, 0-9, . _ -
#   "timestamp": "2026-01-01T00:00:00Z", # timezone-aware ISO-8601 UTC
#   "modality": "bvp|eda|temperature|accelerometer",
#   "sequence": 0,                        # non-negative per device/modality window
#   "sample_rate_hz": 64,                 # fixed rate for the modality
#   "value": 0.0                          # BVP/EDA/TEMP only
# }
# ACC messages replace ``value`` with finite ``acc_x``, ``acc_y``, and
# ``acc_z`` values. The canonical names map to WESAD BVP, EDA, TEMP, and ACC.
EXPECTED_SAMPLE_RATES = {
    "bvp": 64,
    "eda": 4,
    "temperature": 4,
    "accelerometer": 32,
}
DEVICE_TOPIC_RE = re.compile(r"^stresssense/v1/devices/([^/]+)/sensors/([^/]+)$")
DEVICE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
SUPPORTED_STAGES = frozenset(
    {
        "WINDOW_STARTED", "COLLECTING_SAMPLES", "WINDOW_COMPLETED",
        "EXTRACTING_FEATURES", "FEATURES_READY", "PREDICTING",
        "PREDICTION_READY", "WINDOW_RESET", "WINDOW_INCOMPLETE", "WINDOW_ERROR",
    }
)


@dataclass(frozen=True)
class MQTTConfig:
    host: str
    port: int
    username: str
    password: str
    keepalive: int

    @classmethod
    def from_env(cls) -> "MQTTConfig":
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        required = ("MQTT_BROKER_HOST", "MQTT_USERNAME", "MQTT_PASSWORD")
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            raise RuntimeError(f"Missing MQTT settings: {', '.join(missing)}")
        try:
            port = int(os.getenv("MQTT_BROKER_PORT", "8883"))
            keepalive = int(os.getenv("MQTT_KEEPALIVE", "60"))
        except ValueError as error:
            raise RuntimeError("MQTT_BROKER_PORT and MQTT_KEEPALIVE must be integers") from error
        return cls(
            host=os.environ["MQTT_BROKER_HOST"],
            port=port,
            username=os.environ["MQTT_USERNAME"],
            password=os.environ["MQTT_PASSWORD"],
            keepalive=keepalive,
        )


def validate_sensor_message(topic: str, payload: bytes | str) -> dict:
    match = DEVICE_TOPIC_RE.fullmatch(topic)
    if not match:
        raise ValueError("topic must match stresssense/v1/devices/{device_id}/sensors/{modality}")
    try:
        message = json.loads(payload)
    except (TypeError, ValueError) as error:
        raise ValueError("payload is not valid JSON") from error
    if not isinstance(message, dict):
        raise ValueError("payload must be a JSON object")

    topic_device, topic_modality = match.groups()
    required = {"device_id", "timestamp", "modality", "sequence", "sample_rate_hz"}
    missing = required - message.keys()
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    if (
        not isinstance(message["device_id"], str)
        or not DEVICE_ID_RE.fullmatch(message["device_id"])
    ):
        raise ValueError("device_id must contain only letters, digits, '.', '_' or '-'")
    if message["device_id"] != topic_device or message["modality"] != topic_modality:
        raise ValueError("topic device_id/modality does not match payload")
    if topic_modality not in MODALITIES:
        raise ValueError(f"unsupported modality: {topic_modality}")
    parse_timestamp(message["timestamp"])
    if (
        isinstance(message["sequence"], bool)
        or not isinstance(message["sequence"], int)
        or message["sequence"] < 0
    ):
        raise ValueError("sequence must be a non-negative integer")
    if (
        isinstance(message["sample_rate_hz"], bool)
        or not isinstance(message["sample_rate_hz"], (int, float))
        or not math.isfinite(message["sample_rate_hz"])
    ):
        raise ValueError("sample_rate_hz must be finite")
    if message["sample_rate_hz"] != EXPECTED_SAMPLE_RATES[topic_modality]:
        raise ValueError(
            f"sample_rate_hz for {topic_modality} must be "
            f"{EXPECTED_SAMPLE_RATES[topic_modality]}"
        )

    fields = ("value",) if topic_modality != "accelerometer" else ("acc_x", "acc_y", "acc_z")
    for field in fields:
        value = message.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"{field} must be a finite number")
    return message


class MQTTHandler:
    def __init__(
        self,
        config: MQTTConfig,
        status_callback: Callable[[dict], None] | None = None,
        prediction_callback: Callable[[dict], None] | None = None,
        sensor_callback: Callable[[dict], None] | None = None,
        window_manager: WindowManager | None = None,
    ) -> None:
        self.config = config
        self.status_callback = status_callback
        self.prediction_callback = prediction_callback
        self.sensor_callback = sensor_callback
        self.window_manager = window_manager or WindowManager(
            status_callback=status_callback,
            prediction_callback=prediction_callback,
        )
        self.connected = False
        self.last_error: str | None = None
        self.client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id=f"stresssense-backend-{uuid.uuid4().hex[:8]}",
        )
        self.client.username_pw_set(config.username, config.password)
        self.client.tls_set()
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    def start(self) -> None:
        try:
            # Keep FastAPI startup independent of broker reachability. Paho performs
            # DNS/connect/reconnect work on its network thread and reports the
            # result through _on_connect/_on_disconnect.
            self.client.connect_async(
                self.config.host,
                self.config.port,
                self.config.keepalive,
            )
            self.client.loop_start()
        except Exception as error:
            self.last_error = str(error)
            LOGGER.exception("MQTT connection failed")
            raise

    def stop(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()
        self.window_manager.close()
        self.connected = False

    def publish_status(self, event: dict) -> None:
        if event.get("stage") not in SUPPORTED_STAGES:
            LOGGER.error("Refusing unsupported status stage: %s", event.get("stage"))
            return
        self.client.publish(
            f"stresssense/v1/devices/{event['device_id']}/status",
            json.dumps(event),
            qos=1,
        )

    def publish_prediction(self, prediction: dict) -> None:
        self.client.publish(
            f"stresssense/v1/devices/{prediction['device_id']}/predictions",
            json.dumps(prediction),
            qos=1,
        )

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        if reason_code.is_failure:
            self.connected = False
            self.last_error = str(reason_code)
            LOGGER.error("MQTT connection rejected: %s", reason_code)
            return
        self.connected = True
        self.last_error = None
        client.subscribe(SENSOR_TOPIC, qos=1)
        LOGGER.info("Connected to MQTT broker and subscribed to %s", SENSOR_TOPIC)

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties) -> None:
        self.connected = False
        if reason_code != mqtt.MQTT_ERR_SUCCESS:
            self.last_error = str(reason_code)
            LOGGER.warning("MQTT disconnected; Paho will reconnect: %s", reason_code)
        else:
            LOGGER.info("MQTT disconnected")

    def _on_message(self, client, userdata, message) -> None:
        try:
            sensor = validate_sensor_message(message.topic, message.payload)
            accepted = self.window_manager.ingest(sensor)
            if accepted and self.sensor_callback:
                self.sensor_callback(sensor)
        except (ValueError, KeyError) as error:
            LOGGER.warning("Rejected sensor message on %s: %s", message.topic, error)
