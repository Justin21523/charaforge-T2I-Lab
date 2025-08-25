# backend/jobs/worker.py - Celery Worker Configuration
from celery import Celery
import os
import redis
from kombu import Queue

# Shared Cache Bootstrap
import pathlib, torch

AI_CACHE_ROOT = os.getenv("AI_CACHE_ROOT", "/mnt/ai_warehouse/cache")
for k, v in {
    "HF_HOME": f"{AI_CACHE_ROOT}/hf",
    "TRANSFORMERS_CACHE": f"{AI_CACHE_ROOT}/hf/transformers",
    "HF_DATASETS_CACHE": f"{AI_CACHE_ROOT}/hf/datasets",
    "HUGGINGFACE_HUB_CACHE": f"{AI_CACHE_ROOT}/hf/hub",
    "TORCH_HOME": f"{AI_CACHE_ROOT}/torch",
}.items():
    os.environ[k] = v
    pathlib.Path(v).mkdir(parents=True, exist_ok=True)

# App directories
for p in [
    f"{AI_CACHE_ROOT}/models/{name}"
    for name in ["lora", "blip2", "qwen", "llava", "embeddings"]
] + [f"{AI_CACHE_ROOT}/outputs/multi-modal-lab/batch"]:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

# Celery app configuration
celery_app = Celery(
    "multi_modal_lab",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0"),
    include=["backend.jobs.batch_tasks"],
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Taipei",
    enable_utc=True,
    task_track_started=True,
    task_routes={
        "backend.jobs.batch_tasks.process_caption_batch": {"queue": "caption"},
        "backend.jobs.batch_tasks.process_vqa_batch": {"queue": "vqa"},
        "backend.jobs.batch_tasks.process_t2i_batch": {"queue": "t2i"},
    },
    task_default_queue="default",
    task_queues=(
        Queue("default", routing_key="default"),
        Queue("caption", routing_key="caption"),
        Queue("vqa", routing_key="vqa"),
        Queue("t2i", routing_key="t2i"),
    ),
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=50,
    task_soft_time_limit=300,  # 5 minutes
    task_time_limit=600,  # 10 minutes
)
