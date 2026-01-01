"""Celery tasks for processing async T2I jobs."""

from __future__ import annotations

import os

from core.config import get_settings
from workers.celery_app import celery_app


def _redis_url() -> str:
    settings = get_settings()
    return (
        os.getenv("REDIS_URL")
        or os.getenv("CELERY_BROKER_URL")
        or settings.redis_url
        or settings.celery.broker_url
    )


@celery_app.task(bind=True, name="workers.tasks.t2i.process_t2i_job")
def process_t2i_job(self, job_id: str):
    """Process a single T2I job id.

    In `API_T2I_DISPATCH_MODE=celery`, submits enqueue this task instead of using the
    Redis list-based worker loop.
    """

    from api.t2i_jobs import T2IJobManager

    settings = get_settings()
    manager = T2IJobManager(
        redis_url=_redis_url(),
        worker_enabled=False,
        dispatch_mode="celery",
        job_ttl_seconds=int(settings.api.t2i_job_ttl_seconds or 0),
        stale_seconds=int(settings.api.t2i_job_stale_seconds or 0),
        max_attempts=int(settings.api.t2i_job_max_attempts or 1),
        max_concurrent_per_owner=int(settings.api.t2i_max_concurrent or 1),
        max_global_concurrent=int(settings.api.t2i_max_global_concurrent or 0),
    )

    worker_id = getattr(self.request, "id", None) or ""
    ok = manager.process_job(str(job_id), worker_id=str(worker_id))
    if not ok:
        raise self.retry(countdown=1, max_retries=86400)

    return {"status": "ok", "job_id": str(job_id)}

