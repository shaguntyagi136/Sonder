"""Composition root: builds and wires components from Settings."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.core.config import Settings
from app.database.base import Storage
from app.database.sqlite import SQLiteStorage
from app.ml.predictor import Predictor, build_predictor
from app.mqtt.handler import MqttMessageHandler
from app.mqtt.subscriber import MqttSubscriber
from app.preprocessing.features import FeatureSpec, load_feature_spec
from app.preprocessing.preprocessor import Preprocessor
from app.services.pipeline import ProcessingPipeline

logger = logging.getLogger(__name__)


@dataclass
class AppContext:
    settings: Settings
    storage: Storage
    spec: FeatureSpec
    preprocessor: Preprocessor
    predictor: Predictor
    pipeline: ProcessingPipeline
    mqtt: MqttSubscriber | None = None

    def start(self) -> None:
        if self.mqtt:
            self.mqtt.start()

    def stop(self) -> None:
        if self.mqtt:
            self.mqtt.stop()
        self.storage.close()


def build_context(settings: Settings) -> AppContext:
    storage = SQLiteStorage(settings.resolved_database_path)
    storage.init()
    spec = load_feature_spec(settings.resolved_model_meta_path)
    preprocessor = Preprocessor(spec)
    predictor = build_predictor(settings, spec)
    pipeline = ProcessingPipeline(preprocessor, predictor, storage)

    subscriber = None
    if settings.mqtt_enabled:
        handler = MqttMessageHandler(pipeline, settings.max_payload_bytes)
        subscriber = MqttSubscriber(settings, handler.handle)
    else:
        logger.info("MQTT disabled (MQTT_ENABLED=false)")
    return AppContext(settings, storage, spec, preprocessor, predictor, pipeline, subscriber)
