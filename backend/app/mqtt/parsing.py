"""Pure helpers for MQTT messages (no network, trivially testable)."""
from __future__ import annotations

import json
from typing import Any, Optional

from paho.mqtt.client import topic_matches_sub

from app.core.errors import MalformedMessageError


def decode_message(payload: bytes, max_bytes: int) -> dict[str, Any]:
    """bytes -> JSON object. Raises MalformedMessageError; error text never echoes the payload."""
    if len(payload) > max_bytes:
        raise MalformedMessageError(f"payload too large ({len(payload)} bytes, limit {max_bytes})")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        raise MalformedMessageError("payload is not valid UTF-8") from None
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, RecursionError):
        raise MalformedMessageError("payload is not valid JSON") from None
    if not isinstance(data, dict):
        raise MalformedMessageError("JSON payload must be an object")
    return data


def device_id_from_topic(topic: str, subscriptions: list[str]) -> Optional[str]:
    """For 'stress/sensor/+/data' and topic 'stress/sensor/watch-1/data' return 'watch-1'."""
    for pattern in subscriptions:
        if not topic_matches_sub(pattern, topic):
            continue
        for p_seg, t_seg in zip(pattern.split("/"), topic.split("/")):
            if p_seg == "+":
                return t_seg or None
    return None
