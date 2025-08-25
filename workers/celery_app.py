# workers/celery_app.py - Celery configuration and setup
from celery import Celery
import os
from core.config import get_settings

# Load settings
settings = get_settings()

# Create Celery app
celery_app = Celery(
    "charaforge-t2i",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "workers.tasks.training",
        "workers.tasks.generation",
        "workers.tasks.batch",
    ],
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Task routing
    task_routes={
        "workers.tasks.training.*": {"queue": "training"},
        "workers.tasks.generation.*": {"queue": "generation"},
        "workers.tasks.batch.*": {"queue": "batch"},
    },
    # Worker configuration
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=10,
    # Result backend settings
    result_expires=3600 * 24,  # 24 hours
    result_compression="gzip",
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

# Auto-discover tasks
celery_app.autodiscover_tasks()
