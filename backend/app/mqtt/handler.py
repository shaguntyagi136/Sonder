"""MQTT message handling: bytes + topic -> pipeline. No broker/paho dependency here,
so malformed-message behaviour is testable without a broker."""
from __future__ import annotations

import json
import logging

from app.services.pipeline import PipelineResult, ProcessingPipeline

logger = logging.getLogger(__name__)


def device_id_from_topic(topic: str) -> str | None:
    """`<...>/<device_id>/data` -> device_id, else None."""
    parts = topic.split("/")
    if len(parts) >= 2 and parts[-1] == "data" and parts[-2]:
        return parts[-2]
    return None


class MqttMessageHandler:
    def __init__(self, pipeline: ProcessingPipeline, max_payload_bytes: int = 65536):
        self.pipeline = pipeline
        self.max_payload_bytes = max_payload_bytes

    def handle(self, topic: str, payload: bytes) -> PipelineResult:
        """Never raises: a bad message must not take down the MQTT network thread."""
        try:
            logger.info("MQTT message on %s (%d bytes)", topic, len(payload))
            if len(payload) > self.max_payload_bytes:
                return self.pipeline.record_rejection("payload_too_large")
            try:
                data = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return self.pipeline.record_rejection("invalid_json")
            if not isinstance(data, dict):
                return self.pipeline.record_rejection("payload_not_an_object")

            topic_device = device_id_from_topic(topic)
            if topic_device:
                if data.get("device_id") not in (None, topic_device):
                    return self.pipeline.record_rejection("device_id_topic_mismatch")
                data["device_id"] = topic_device
            return self.pipeline.process(data)
        except Exception:
            logger.exception("Unhandled error while processing MQTT message on %s", topic)
            return self.pipeline.record_rejection("internal_error")
