# workers/celery_app.py - Celery Application Configuration
"""
SagaForge T2I Lab Celery 配置
處理訓練任務、批次生成、模型管理等後台作業
"""

import os
import sys
from pathlib import Path
from celery import Celery
from datetime import datetime
from kombu import Queue

# 確保可以導入專案模組
ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 導入配置
try:
    from core.config import get_settings, bootstrap_config

    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    print("⚠️ Core modules not available, using fallback config")

# ===== Celery Configuration =====


def get_celery_config():
    """取得 Celery 配置"""
    if CORE_AVAILABLE:
        try:
            settings = get_settings()
            return {
                "broker_url": settings.celery.broker_url,
                "result_backend": settings.celery.result_backend,
                "worker_concurrency": settings.celery.worker_concurrency,
            }
        except Exception as e:
            print(f"⚠️ Failed to load settings: {e}")

    # Fallback configuration
    return {
        "broker_url": os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
        "result_backend": os.getenv(
            "CELERY_RESULT_BACKEND", "redis://localhost:6379/0"
        ),
        "worker_concurrency": int(os.getenv("CELERY_WORKER_CONCURRENCY", "1")),
    }


config = get_celery_config()

# Create Celery app
celery_app = Celery("sagaforge_workers")

# Basic configuration
celery_app.conf.update(
    broker_url=config["broker_url"],
    result_backend=config["result_backend"],
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
    # Queues
    task_default_queue="default",
    task_queues=(
        Queue("default", routing_key="default"),
        Queue("training", routing_key="training"),
        Queue("generation", routing_key="generation"),
        Queue("batch", routing_key="batch"),
    ),
    # Worker settings
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=10,  # Restart worker after N tasks to prevent memory leaks
    # Task timeouts
    task_soft_time_limit=3600,  # 1 hour soft limit
    task_time_limit=7200,  # 2 hour hard limit
    # Results
    result_expires=86400,  # 24 hours
    result_persistent=True,
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

# ===== Task Discovery =====

# Import tasks to register them
task_modules = [
    "workers.tasks.training",
    "workers.tasks.generation",
    "workers.tasks.batch",
]

for module in task_modules:
    try:
        celery_app.autodiscover_tasks([module])
        print(f"✅ Loaded tasks from {module}")
    except ImportError as e:
        print(f"⚠️ Failed to load tasks from {module}: {e}")

# ===== Celery Events and Signals =====


@celery_app.task(bind=True, name="workers.health_check")
def health_check(self):
    """Worker 健康檢查任務"""
    import torch
    from datetime import datetime

    return {
        "worker_id": self.request.id,
        "timestamp": datetime.now().isoformat(),
        "hostname": self.request.hostname,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "memory_info": (
            {
                "allocated": (
                    torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                ),
                "cached": (
                    torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
                ),
            }
            if torch.cuda.is_available()
            else None
        ),
        "status": "healthy",
    }


@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """設定週期性任務"""
    # 每30秒執行健康檢查
    sender.add_periodic_task(30.0, health_check.s(), name="worker health check")


@celery_app.task(bind=True, name="workers.cleanup_cache")
def cleanup_cache(self):
    """清理快取任務"""
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "task_id": self.request.id,
        "timestamp": datetime.now().isoformat(),
        "action": "cache_cleaned",
        "memory_freed": True,
    }


# ===== Worker Bootstrap =====


@celery_app.on_worker_init.connect
def setup_worker(sender=None, **kwargs):
    """Worker 初始化設定"""
    print("🔧 Initializing Celery worker...")

    if CORE_AVAILABLE:
        try:
            # Bootstrap configuration and cache
            settings, cache_paths, app_paths = bootstrap_config(verbose=True)
            print(f"✅ Worker configuration initialized")
            print(f"   Cache root: {cache_paths.root}")
            print(f"   Environment: {settings.environment}")

        except Exception as e:
            print(f"⚠️ Worker bootstrap failed: {e}")

    # Set up logging
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("🚀 Celery worker ready")


@celery_app.on_worker_shutdown.connect
def cleanup_worker(sender=None, **kwargs):
    """Worker 關閉清理"""
    print("🛑 Shutting down Celery worker...")

    # Clear GPU memory
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("✅ GPU memory cleared")
    except ImportError:
        pass

    print("✅ Worker shutdown complete")


# ===== Task Progress Tracking =====


class TaskProgress:
    """任務進度追蹤輔助類"""

    def __init__(self, task, total_steps: int = 100):
        self.task = task
        self.total_steps = total_steps
        self.current_step = 0

    def update(self, step: int, message: str = "", **extra_data):
        """更新任務進度"""
        self.current_step = step
        progress_percent = (step / self.total_steps) * 100

        state_data = {
            "current": step,
            "total": self.total_steps,
            "percent": round(progress_percent, 1),
            "message": message,
            **extra_data,
        }

        self.task.update_state(state="PROGRESS", meta=state_data)

        return state_data

    def complete(self, result: dict):
        """標記任務完成"""
        final_data = {
            "current": self.total_steps,
            "total": self.total_steps,
            "percent": 100.0,
            "message": "Complete",
            **result,
        }

        self.task.update_state(state="SUCCESS", meta=final_data)

        return final_data

    def fail(self, error_message: str, **extra_data):
        """標記任務失敗"""
        error_data = {
            "current": self.current_step,
            "total": self.total_steps,
            "percent": (self.current_step / self.total_steps) * 100,
            "message": f"Failed: {error_message}",
            "error": error_message,
            **extra_data,
        }

        self.task.update_state(state="FAILURE", meta=error_data)

        return error_data


# ===== CLI Entry Points =====


def start_worker():
    """啟動 Celery Worker"""
    print("🔥 Starting Celery worker...")

    # Worker arguments
    worker_args = [
        "worker",
        "--loglevel=info",
        f"--concurrency={config['worker_concurrency']}",
        "--queues=default,training,generation,batch",
        "--pool=solo" if os.name == "nt" else "--pool=prefork",  # Windows compatibility
    ]

    celery_app.start(worker_args)


def start_beat():
    """啟動 Celery Beat (排程器)"""
    print("⏰ Starting Celery beat scheduler...")

    beat_args = [
        "beat",
        "--loglevel=info",
    ]

    celery_app.start(beat_args)


def start_flower():
    """啟動 Flower (監控介面)"""
    print("🌸 Starting Flower monitoring...")

    flower_args = [
        "flower",
        "--port=5555",
        "--basic_auth=admin:admin123",  # Change in production
    ]

    celery_app.start(flower_args)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "worker":
            start_worker()
        elif command == "beat":
            start_beat()
        elif command == "flower":
            start_flower()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: worker, beat, flower")
    else:
        print("Usage: python celery_app.py [worker|beat|flower]")
        print("  worker: Start task worker")
        print("  beat: Start periodic task scheduler")
        print("  flower: Start web monitoring interface")
