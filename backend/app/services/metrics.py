"""Tiny thread-safe counters exposed via /health."""
import threading


class Metrics:
    NAMES = (
        "received", "rejected", "predicted", "stored_without_prediction",
        "inference_errors", "storage_errors", "unexpected_errors",
        "raw_mqtt_received", "raw_samples_accepted", "raw_samples_rejected",
        "raw_samples_accepted_bvp", "raw_samples_accepted_eda",
        "raw_samples_accepted_skin_temperature", "raw_samples_accepted_accelerometer",
        "raw_windows_started", "raw_windows_rolled_over", "raw_windows_incomplete",
        "raw_windows_completed", "inference_started", "inference_succeeded",
        "inference_failed", "persistence_succeeded", "persistence_failed",
    )

    def __init__(self):
        self._lock = threading.Lock()
        self._c = {n: 0 for n in self.NAMES}

    def incr(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._c[name] += n

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._c)
