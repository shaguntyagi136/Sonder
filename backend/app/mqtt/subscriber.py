"""paho-mqtt (>=2.0) subscriber. Only concerns: connection, reconnection, subscription,
and handing raw (topic, bytes) to a callback. No parsing, no API, no ML."""
from __future__ import annotations

import logging
import json
import threading
from typing import Callable

import paho.mqtt.client as mqtt

from app.core.config import Settings

logger = logging.getLogger(__name__)


def _configure_client(client: mqtt.Client, settings: Settings) -> None:
    if settings.mqtt_username:
        password = settings.mqtt_password.get_secret_value() if settings.mqtt_password else None
        client.username_pw_set(settings.mqtt_username, password)
    if settings.mqtt_use_tls:
        client.tls_set()
    client.reconnect_delay_set(min_delay=1, max_delay=60)


def publish_topic_for_device(topics: list[str], device_id: str) -> str:
    """Resolve the first configured singletopic-level device topic for a publisher."""
    for topic in topics:
        if "+" in topic and "#" not in topic:
            return topic.replace("+", device_id, 1)
    raise ValueError("MQTT_TOPICS must contain a publishable topic with a '+' device segment")


class MqttSubscriber:
    def __init__(self, settings: Settings, on_message: Callable[[str, bytes], object]):
        self._s = settings
        self._callback = on_message
        self._connected = False
        self._subscription_acknowledged = False
        self._subscribed_topics: tuple[str, ...] = ()
        self._received_messages = 0
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=settings.mqtt_client_id)
        _configure_client(self._client, settings)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_subscribe = self._on_subscribe
        self._client.on_message = self._on_message

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def subscription_acknowledged(self) -> bool:
        return self._subscription_acknowledged

    @property
    def subscribed_topics(self) -> tuple[str, ...]:
        return self._subscribed_topics

    @property
    def received_messages(self) -> int:
        return self._received_messages

    def start(self) -> None:
        """Non-blocking. If the broker is down the app still starts; paho keeps retrying."""
        logger.info("MQTT connecting to %s:%s (tls=%s, auth=%s)", self._s.mqtt_broker_host, self._s.mqtt_broker_port,
                    self._s.mqtt_use_tls, bool(self._s.mqtt_username))
        self._client.connect_async(self._s.mqtt_broker_host, self._s.mqtt_broker_port, self._s.mqtt_keepalive)
        self._client.loop_start()

    def stop(self) -> None:
        try:
            self._client.disconnect()
            self._client.loop_stop()
        finally:
            self._connected = False
            logger.info("MQTT subscriber stopped")

    # -- callbacks (paho network thread) ---------------------------------
    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if getattr(reason_code, "is_failure", False):
            self._connected = False
            self._subscription_acknowledged = False
            self._subscribed_topics = ()
            logger.error("MQTT connection refused: %s", reason_code)
            return
        self._connected = True
        self._subscription_acknowledged = False
        self._subscribed_topics = ()
        logger.info("MQTT connected to %s:%s", self._s.mqtt_broker_host, self._s.mqtt_broker_port)
        topics = [(t, self._s.mqtt_qos) for t in self._s.mqtt_topic_list]
        if topics:
            subscribe_result = client.subscribe(topics)  # re-subscribe on every (re)connect
            result, mid = subscribe_result if isinstance(subscribe_result, tuple) else (mqtt.MQTT_ERR_SUCCESS, None)
            if result != mqtt.MQTT_ERR_SUCCESS:
                logger.error("MQTT subscribe request failed: rc=%s topics=%s", result, [t for t, _ in topics])
            else:
                self._subscribed_topics = tuple(t for t, _ in topics)
                logger.info("MQTT subscribe request sent: mid=%s topics=%s qos=%s", mid, self._subscribed_topics, self._s.mqtt_qos)

    def _on_subscribe(self, client, userdata, mid, reason_codes, properties=None):
        granted = [str(code) for code in reason_codes]
        self._subscription_acknowledged = bool(granted) and not any(
            getattr(code, "is_failure", None)
            if hasattr(code, "is_failure")
            else "failure" in str(code).lower()
            for code in reason_codes
        )
        if self._subscription_acknowledged:
            logger.info("MQTT SUBACK received: mid=%s topics=%s granted_qos=%s", mid, self._subscribed_topics, granted)
        else:
            logger.error("MQTT SUBACK rejected: mid=%s topics=%s granted_qos=%s", mid, self._subscribed_topics, granted)

    def _on_disconnect(self, client, userdata, disconnect_flags=None, reason_code=None, properties=None):
        self._connected = False
        self._subscription_acknowledged = False
        self._subscribed_topics = ()
        logger.warning("MQTT disconnected (%s); paho will retry", reason_code)

    def _on_message(self, client, userdata, msg):
        try:
            self._received_messages += 1
            logger.info(
                "MQTT MESSAGE RECEIVED: topic=%s payload_bytes=%s callback=%s message_count=%s",
                msg.topic,
                len(msg.payload),
                getattr(self._callback, "__qualname__", repr(self._callback)),
                self._received_messages,
            )
            self._callback(msg.topic, bytes(msg.payload))
        except Exception:
            logger.exception("MQTT callback raised for topic %s", msg.topic)


class MqttPublisher:
    """Reusable synchronous JSON publisher using the project's MQTT settings."""

    def __init__(self, settings: Settings, client_id_suffix: str = "publisher"):
        self._settings = settings
        self._connected = threading.Event()
        self._disconnected = threading.Event()
        self._client_id = f"{settings.mqtt_client_id}-{client_id_suffix}"
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=self._client_id)
        _configure_client(self._client, settings)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    @property
    def client_id(self) -> str:
        return self._client_id

    def connect(self, timeout_seconds: float = 10.0) -> None:
        self._disconnected.clear()
        self._client.connect(self._settings.mqtt_broker_host, self._settings.mqtt_broker_port, self._settings.mqtt_keepalive)
        self._client.loop_start()
        if not self._connected.wait(timeout_seconds):
            self.close()
            raise ConnectionError(
                f"MQTT connection timed out: {self._settings.mqtt_broker_host}:{self._settings.mqtt_broker_port}"
            )

    def publish_json(self, topic: str, payload: dict) -> None:
        if not self.connected:
            raise ConnectionError("MQTT publisher is not connected")
        result = self._client.publish(topic, json.dumps(payload, separators=(",", ":")), qos=self._settings.mqtt_qos)
        result.wait_for_publish()
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            raise ConnectionError(f"MQTT publish failed with result code {result.rc}")
        logger.info(
            "MQTT publish acknowledged: broker=%s:%s client_id=%s topic=%s qos=%s mid=%s",
            self._settings.mqtt_broker_host,
            self._settings.mqtt_broker_port,
            self._client_id,
            topic,
            self._settings.mqtt_qos,
            result.mid,
        )

    def close(self, timeout_seconds: float = 5.0) -> None:
        """Gracefully flush acknowledged publishes before stopping Paho's network loop."""
        try:
            if self.connected:
                # publish_json waits for each QoS acknowledgement.  Keep the loop
                # alive long enough to transmit DISCONNECT and finish its normal
                # callback cycle instead of stopping it immediately.
                self._client.disconnect()
                self._disconnected.wait(timeout_seconds)
        finally:
            self._client.loop_stop()
            self._connected.clear()

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if getattr(reason_code, "is_failure", False):
            logger.error("MQTT publisher connection refused: %s", reason_code)
            return
        self._connected.set()
        logger.info("MQTT publisher connected to %s:%s", self._settings.mqtt_broker_host, self._settings.mqtt_broker_port)

    def _on_disconnect(self, client, userdata, disconnect_flags=None, reason_code=None, properties=None):
        self._connected.clear()
        self._disconnected.set()
        logger.warning("MQTT publisher disconnected (%s)", reason_code)
