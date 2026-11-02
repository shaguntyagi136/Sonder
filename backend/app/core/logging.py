from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def setup_logging(level: str = "INFO", debug: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if debug else getattr(logging, level, logging.INFO), format=_FORMAT)
    # paho is chatty at DEBUG; our own MQTT logs carry what matters.
    logging.getLogger("paho").setLevel(logging.WARNING)
