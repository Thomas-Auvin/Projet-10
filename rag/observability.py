# rag/observability.py
from __future__ import annotations

import logging
import os

import logfire

logger = logging.getLogger(__name__)

_LOGFIRE_CONFIGURED = False


def setup_logfire() -> None:
    """Configure Logfire une seule fois pour l'application."""
    global _LOGFIRE_CONFIGURED

    if _LOGFIRE_CONFIGURED:
        return

    enabled = os.getenv("LOGFIRE_ENABLED", "1") == "1"
    if not enabled:
        logger.info("Logfire désactivé via LOGFIRE_ENABLED=0")
        _LOGFIRE_CONFIGURED = True
        return

    logfire.configure()
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx(capture_all=True)

    logger.info("Logfire configuré avec succès.")
    _LOGFIRE_CONFIGURED = True
