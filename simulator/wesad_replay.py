"""Replay WESAD wrist recordings through StressSense or evaluate them offline.

The window and label preparation intentionally mirrors
``ml/notebook/stress-iot (1).ipynb``. This utility never retrains the model.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.feature_extractor import extract_window_features
from backend.app.predictor import StressPredictor

LOGGER = logging.getLogger("wesad-replay")
WINDOW_SECONDS = 60
RATES = {"bvp": 64, "eda": 4, "temperature": 4, "accelerometer": 32}
SAMPLES = {name: rate * WINDOW_SECONDS for name, rate in RATES.items()}
LABEL_RATE = 700


@dataclass(frozen=True)
class WESADWindow:
    subject_id: str
    window_index: int
    start_seconds: int
    label: int
    bvp: np.ndarray
    eda: np.ndarray
    temperature: np.ndarray
    accelerometer: np.ndarray


def subject_paths(dataset: Path, subject: str | None = None) -> list[Path]:
    paths = sorted(dataset.glob("S*/S*.pkl"), key=lambda path: int(path.stem[1:]))
    if subject:
        paths = [path for path in paths if path.stem == subject]
    if not paths:
        raise FileNotFoundError(f"No WESAD subject pickle files found under {dataset}")
    return paths


def iter_subject_windows(path: Path) -> Iterator[WESADWindow]:
    """Yield only pure-label, complete windows using the training notebook mapping."""
    with path.open("rb") as file:
        data = pickle.load(file, encoding="latin1")
    wrist = data["signal"]["wrist"]
    labels = np.asarray(data["label"])
    signals = {
        "bvp": np.asarray(wrist["BVP"], dtype=np.float64).reshape(-1),
        "eda": np.asarray(wrist["EDA"], dtype=np.float64).reshape(-1),
        "temperature": np.asarray(wrist["TEMP"], dtype=np.float64).reshape(-1),
        "accelerometer": np.asarray(wrist["ACC"], dtype=np.float64) / 64.0,
    }
    count = min(
        len(signals[name]) // samples for name, samples in SAMPLES.items()
    )
    count = min(count, len(labels) // (LABEL_RATE * WINDOW_SECONDS))
    for index in range(count):
        label_start = index * LABEL_RATE * WINDOW_SECONDS
        label_end = label_start + LABEL_RATE * WINDOW_SECONDS
        label_window = labels[label_start:label_end]
        if np.all(label_window == 1):
            target = 0
        elif np.all(label_window == 2):
            target = 1
        else:
            continue
        yield WESADWindow(
            subject_id=path.stem,
            window_index=index,
            start_seconds=index * WINDOW_SECONDS,
            label=target,
            bvp=signals["bvp"][index * SAMPLES["bvp"] : (index + 1) * SAMPLES["bvp"]],
            eda=signals["eda"][index * SAMPLES["eda"] : (index + 1) * SAMPLES["eda"]],
            temperature=signals["temperature"][
                index * SAMPLES["temperature"] : (index + 1) * SAMPLES["temperature"]
            ],
            accelerometer=signals["accelerometer"][
                index * SAMPLES["accelerometer"] : (index + 1) * SAMPLES["accelerometer"]
            ],
        )


def reading_for(window: WESADWindow) -> dict[str, list[float]]:
    return {
        "bvp": window.bvp.tolist(),
        "eda": window.eda.tolist(),
        "skin_temperature": window.temperature.tolist(),
        "acc_x": window.accelerometer[:, 0].tolist(),
        "acc_y": window.accelerometer[:, 1].tolist(),
        "acc_z": window.accelerometer[:, 2].tolist(),
    }


def evaluate(dataset: Path, subject: str | None, limit: int | None) -> None:
    predictor = StressPredictor()
    truth: list[int] = []
    predictions: list[int] = []
    processed = 0
    for path in subject_paths(dataset, subject):
        for window in iter_subject_windows(path):
            result = predictor.predict(extract_window_features(reading_for(window)))
            truth.append(window.label)
            predictions.append(result["prediction"])
            processed += 1
            if limit and processed >= limit:
                break
        if limit and processed >= limit:
            break
    if not truth:
        raise RuntimeError("No pure-label complete WESAD windows were available")
    print(json.dumps({
        "windows": processed,
        "confusion_matrix_labels_0_1": confusion_matrix(truth, predictions, labels=[0, 1]).tolist(),
        "accuracy": accuracy_score(truth, predictions),
        "precision": precision_score(truth, predictions, labels=[0, 1], zero_division=0),
        "recall": recall_score(truth, predictions, labels=[0, 1], zero_division=0),
        "macro_f1": f1_score(truth, predictions, labels=[0, 1], average="macro", zero_division=0),
    }, indent=2))


def _mqtt_client(client_id: str):
    import paho.mqtt.client as mqtt
    from paho.mqtt.enums import CallbackAPIVersion

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    client = mqtt.Client(
        callback_api_version=CallbackAPIVersion.VERSION2,
        client_id=client_id,
    )
    client.username_pw_set(os.environ["MQTT_USERNAME"], os.environ["MQTT_PASSWORD"])
    client.tls_set()
    return client


def replay(
    dataset: Path,
    subject: str,
    limit: int | None,
    speed: float,
    device_id: str,
) -> None:
    if not 0.75 <= speed <= 1.0:
        raise ValueError("--speed must be between 0.75 and 1.0 for backend duration validation")
    paths = subject_paths(dataset, subject)
    windows = []
    for window in iter_subject_windows(paths[0]):
        windows.append(window)
        if limit and len(windows) >= limit:
            break
    if not windows:
        raise RuntimeError("No pure-label complete WESAD windows available for replay")

    from paho.mqtt.client import MQTTMessage

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    host = os.environ["MQTT_BROKER_HOST"]
    port = int(os.getenv("MQTT_BROKER_PORT", "8883"))
    keepalive = int(os.getenv("MQTT_KEEPALIVE", "60"))
    status: list[dict] = []
    predictions: list[dict] = []
    connected = threading.Event()
    observer = _mqtt_client(f"stresssense-wesad-observer-{os.getpid()}")

    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            raise RuntimeError(f"MQTT observer connection failed: {reason_code}")
        client.subscribe(f"stresssense/v1/devices/{device_id}/#", qos=1)
        connected.set()

    def on_message(client, userdata, message):
        payload = json.loads(message.payload)
        if message.topic.endswith("/status"):
            status.append(payload)
        elif message.topic.endswith("/predictions"):
            predictions.append(payload)

    observer.on_connect = on_connect
    observer.on_message = on_message
    observer.connect(host, port, keepalive)
    observer.loop_start()
    if not connected.wait(10):
        raise RuntimeError("MQTT observer did not connect")

    publisher = _mqtt_client(f"stresssense-wesad-publisher-{os.getpid()}")
    publisher.connect(host, port, keepalive)
    publisher.loop_start()
    start_timestamp = datetime.now(timezone.utc).replace(
        second=0, microsecond=0
    ) + timedelta(minutes=2)
    try:
        for window in windows:
            base = start_timestamp + timedelta(seconds=window.start_seconds)
            samples = []
            for modality, values in (
                ("bvp", window.bvp),
                ("eda", window.eda),
                ("temperature", window.temperature),
            ):
                for sequence, value in enumerate(values):
                    samples.append((sequence / RATES[modality], modality, sequence, value))
            for sequence, value in enumerate(window.accelerometer):
                samples.append((sequence / RATES["accelerometer"], "accelerometer", sequence, value))
            samples.sort(key=lambda item: item[0])
            real_start = time.monotonic()
            for relative, modality, sequence, value in samples:
                target = real_start + relative / speed
                if target > time.monotonic():
                    time.sleep(target - time.monotonic())
                payload = {
                    "device_id": device_id,
                    "timestamp": (base + timedelta(seconds=relative)).isoformat().replace("+00:00", "Z"),
                    "source": "wesad-replay",
                    "modality": modality,
                    "sequence": sequence,
                    "sample_rate_hz": RATES[modality],
                }
                if modality == "accelerometer":
                    payload.update({"acc_x": float(value[0]), "acc_y": float(value[1]), "acc_z": float(value[2])})
                else:
                    payload["value"] = float(value)
                publisher.publish(
                    f"stresssense/v1/devices/{device_id}/sensors/{modality}",
                    json.dumps(payload),
                    qos=1,
                )
            publisher.loop_write()
            time.sleep(3)
    finally:
        publisher.loop_stop()
        publisher.disconnect()
        time.sleep(2)
        observer.loop_stop()
        observer.disconnect()
    expected = {window.window_index: window.label for window in windows}
    print(json.dumps({
        "subject": subject,
        "windows_sent": len(windows),
        "predictions_received": len(predictions),
        "prediction_labels": [
            {"window_id": prediction.get("window_id"), "prediction": prediction.get("prediction")}
            for prediction in predictions
        ],
        "status_stages": dict(Counter(item.get("stage") for item in status)),
        "ground_truth_by_window_index": expected,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("WESAD"))
    parser.add_argument("--subject", help="Subject ID such as S2")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--mode", choices=("evaluate", "replay"), default="evaluate")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--device-id", default="wesad-replay")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.mode == "evaluate":
        evaluate(args.dataset, args.subject, args.limit)
    else:
        if not args.subject:
            parser.error("--subject is required in replay mode")
        replay(args.dataset, args.subject, args.limit, args.speed, args.device_id)


if __name__ == "__main__":
    main()
