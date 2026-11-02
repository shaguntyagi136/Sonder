"""Composition root: the one place where the layers are wired together."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from app.core.config import Settings
from app.core.timeutil import utcnow
from app.database import create_repository
from app.database.base import Repository
from app.ml.service import InferenceService
from app.mqtt.subscriber import MqttSubscriber
from app.preprocessing.preprocessor import Preprocessor
from app.services.pipeline import IngestionPipeline

log = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    repo: Repository
    inference: InferenceService
    pipeline: IngestionPipeline
    mqtt: Optional[MqttSubscriber] = None
    started_at: datetime = field(default_factory=utcnow)

    def start(self) -> None:
        if self.mqtt:
            self.mqtt.start()
        else:
            log.info("MQTT disabled (MQTT_ENABLED=false)")

    def stop(self) -> None:
        if self.mqtt:
            self.mqtt.stop()
        self.repo.close()
        log.info("Backend stopped")


def build_container(settings: Settings) -> Container:
    repo = create_repository(settings)
    inference = InferenceService.from_settings(settings)
    preprocessor = Preprocessor(inference.spec, settings.missing_feature_policy) if inference.spec else None
    pipeline = IngestionPipeline(
        repo, inference, preprocessor,
        subscription_patterns=settings.mqtt_topic_list,
        max_payload_bytes=settings.mqtt_max_payload_bytes,
    )
    mqtt = MqttSubscriber(settings, pipeline.ingest_mqtt_message) if settings.mqtt_enabled else None
    return Container(settings, repo, inference, pipeline, mqtt)
