"""Application logging: console + rotating file. Ticket text is never logged (PII); ids and metrics are."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

import config

_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    global _configured
    if _configured:
        return
    config.APP_LOG.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    for handler in (logging.StreamHandler(), RotatingFileHandler(config.APP_LOG, maxBytes=1_000_000, backupCount=3)):
        handler.setFormatter(fmt)
        root.addHandler(handler)
    _configured = True
