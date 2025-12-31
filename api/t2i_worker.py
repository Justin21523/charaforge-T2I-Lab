"""Standalone Redis-backed T2I job worker (no HTTP server)."""

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
    from api.t2i_jobs import T2IJobManager

    manager = T2IJobManager(
        redis_url=_redis_url(),
        worker_enabled=True,
        job_ttl_seconds=int(settings.api.t2i_job_ttl_seconds or 0),
        stale_seconds=int(settings.api.t2i_job_stale_seconds or 0),
        max_attempts=int(settings.api.t2i_job_max_attempts or 1),
        max_concurrent_per_owner=int(settings.api.t2i_max_concurrent or 1),
        max_global_concurrent=int(settings.api.t2i_max_global_concurrent or 0),
    )

    stop = {"value": False}

    def _handle_signal(signum: int, frame) -> None:  # type: ignore[no-untyped-def]
        stop["value"] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info("T2I worker started (Redis queue)")
    while not stop["value"]:
        time.sleep(0.5)

    logger.info("T2I worker shutting down")
    manager.shutdown()


if __name__ == "__main__":
    main()
