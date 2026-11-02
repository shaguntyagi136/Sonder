"""Publish real WESAD E4 samples progressively as wearable MQTT messages."""
from __future__ import annotations

import argparse
import heapq
import sys
import time
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

from app.core.config import get_settings
from app.mqtt.subscriber import MqttPublisher, publish_topic_for_device
from app.preprocessing.wesad import load_wesad_participant
from app.schemas.sensor import SensorReading

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = BACKEND_ROOT / "data" / "archive (8).zip"


def _raw_messages(window: SensorReading) -> Iterator[tuple[object, dict]]:
    """Merge actual per-modality samples into ascending simulated timestamp order."""
    streams = (
        ("bvp", window.bvp, window.sample_rates_hz["bvp"]),
        ("eda", window.eda, window.sample_rates_hz["eda"]),
        ("skin_temperature", window.skin_temperature, window.sample_rates_hz["skin_temperature"]),
        ("accelerometer", window.acc_x, window.sample_rates_hz["acc_x"]),
    )
    queue: list[tuple[float, int, str, int, float]] = []
    for order, (modality, values, rate) in enumerate(streams):
        if values:
            heapq.heappush(queue, (0.0, order, modality, 0, rate))

    while queue:
        offset_seconds, order, modality, index, rate = heapq.heappop(queue)
        timestamp = window.timestamp + timedelta(seconds=offset_seconds)
        payload = {
            "device_id": window.device_id,
            "subject_id": window.subject_id,
            "timestamp": timestamp.isoformat(),
            "source": "wesad_e4_simulator",
            "modality": modality,
            "sample_rate_hz": rate,
        }
        if modality == "accelerometer":
            payload["acc_x"] = window.acc_x[index]
            payload["acc_y"] = window.acc_y[index]
            payload["acc_z"] = window.acc_z[index]
        else:
            payload["value"] = window.bvp[index] if modality == "bvp" else (
                window.eda[index] if modality == "eda" else window.skin_temperature[index]
            )
        yield timestamp, payload

        next_index = index + 1
        values = window.acc_x if modality == "accelerometer" else (
            window.bvp if modality == "bvp" else window.eda if modality == "eda" else window.skin_temperature
        )
        if next_index < len(values):
            heapq.heappush(queue, (next_index / rate, order, modality, next_index, rate))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish WESAD raw wearable samples over MQTT.")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE, help="Path to the WESAD ZIP archive.")
    parser.add_argument("--subject", default="S2", help="WESAD subject ID, e.g. S2.")
    parser.add_argument("--speed", type=float, default=1.0, help="Simulated-time multiplier; must be positive.")
    parser.add_argument("--topic", default=None, help="Override the configured publish topic.")
    parser.add_argument("--max-windows", type=int, default=None, help="Optional maximum number of 60-second windows.")
    parser.add_argument("--connect-timeout", type=float, default=10.0, help="MQTT connection timeout in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.speed <= 0:
        raise ValueError("--speed must be positive")
    settings = get_settings()
    device_id = f"wesad-{args.subject.lower()}"
    topic = args.topic or publish_topic_for_device(settings.mqtt_topic_list, device_id)
    publisher = MqttPublisher(settings, client_id_suffix=f"sim-{args.subject.lower()}")
    print(
        f"WESAD simulator: subject={args.subject}, broker={settings.mqtt_broker_host}:{settings.mqtt_broker_port}, "
        f"client_id={publisher.client_id}, topic={topic}, qos={settings.mqtt_qos}, speed={args.speed}x"
    )

    previous_simulated_time = None
    messages = 0
    completed = False
    try:
        publisher.connect(args.connect_timeout)
        print("MQTT connected; publishing raw WESAD samples. Press Ctrl+C to stop.")
        windows = load_wesad_participant(
            args.archive,
            args.subject,
            window_seconds=settings.default_window_seconds,
            include_original_labels=False,
            max_windows=args.max_windows,
        )
        for window in windows:
            window = window.model_copy(update={"device_id": device_id})
            for simulated_time, payload in _raw_messages(window):
                if previous_simulated_time is not None:
                    elapsed = (simulated_time - previous_simulated_time).total_seconds()
                    if elapsed > 0:
                        time.sleep(elapsed / args.speed)
                try:
                    publisher.publish_json(topic, payload)
                except ConnectionError as exc:
                    print(
                        "MQTT publish failed: "
                        f"device_id={device_id} topic={topic} "
                        f"simulated_time={simulated_time.isoformat()} "
                        f"published_messages={messages} error={exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    raise
                messages += 1
                previous_simulated_time = simulated_time
                if messages == 1 or messages % 100 == 0:
                    print(f"simulated_time={simulated_time.isoformat()} published_messages={messages} speed={args.speed}x")
        completed = True
    except KeyboardInterrupt:
        print(f"Simulator stopped gracefully after {messages} published messages.")
    finally:
        if completed:
            print(f"All {messages} MQTT messages were acknowledged; draining publisher before disconnect.")
        publisher.close()


if __name__ == "__main__":
    main()
