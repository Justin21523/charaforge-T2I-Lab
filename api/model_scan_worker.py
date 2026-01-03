"""Standalone Redis-backed model scan job worker (no HTTP server)."""

from __future__ import annotations

import logging
import os
import signal
import time

from core.config import bootstrap_config, get_settings

logger = logging.getLogger(__name__)


def _redis_url() -> str:
    settings = get_settings()
    return (
        os.getenv("REDIS_URL")
        or os.getenv("CELERY_BROKER_URL")
        or settings.redis_url
        or settings.celery.broker_url
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    bootstrap_config(verbose=True)

    settings = get_settings()
    from api.model_scan_jobs import ModelScanJobManager

    manager = ModelScanJobManager(
        redis_url=_redis_url(),
        worker_enabled=True,
        job_ttl_seconds=int(settings.api.models_scan_job_ttl_seconds or 0),
    )

    stop = {"value": False}

    def _handle_signal(signum: int, frame) -> None:  # type: ignore[no-untyped-def]
        stop["value"] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info("Model scan worker started (Redis queue)")
    while not stop["value"]:
        time.sleep(0.5)

    logger.info("Model scan worker shutting down")
    manager.shutdown()


if __name__ == "__main__":
    main()

