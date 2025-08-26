# workers/celery_app.py - Celery configuration and setup
import os
import logging
from datetime import datetime
from typing import List, Dict, Union, Optional, Any
import torch
from celery import Celery
from celery.signals import (
    worker_ready,
    worker_shutdown,
    task_prerun,
    task_postrun,
    task_failure,
)

from workers.utils.job_tracker import JobTracker
from core.config import get_settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        "workers.tasks.batch"
    ]
)

# Enhanced Celery configuration
celery_app.conf.update(
    # Serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',

    # Timezone
    timezone='UTC',
    enable_utc=True,

    # Task routing and queues
    task_routes={
        'workers.tasks.training.*': {'queue': 'training'},
        'workers.tasks.generation.*': {'queue': 'generation'},
        'workers.tasks.batch.*': {'queue': 'batch'},
    },

    # Worker configuration
    worker_prefetch_multiplier=1,  # Important for memory-intensive tasks
    task_acks_late=True,
    worker_max_tasks_per_child=3,  # Prevent memory leaks
    worker_disable_rate_limits=True,

    # Task execution
    task_soft_time_limit=3600,  # 1 hour soft limit
    task_time_limit=7200,       # 2 hour hard limit
    task_reject_on_worker_lost=True,

    # Result backend settings
    result_expires=3600 * 24 * 7,  # 7 days
    result_compression='gzip',
    result_chord_join_timeout=60,

    # Monitoring and events
    worker_send_task_events=True,
    task_send_sent_event=True,
    task_track_started=True,

    # Beat schedule (for periodic tasks)
    beat_schedule={
        'cleanup-old-jobs': {
            'task': 'workers.tasks.maintenance.cleanup_old_jobs',
            'schedule': 3600 * 24,  # Daily
        },
        'update-system-stats': {
            'task': 'workers.tasks.maintenance.update_system_stats',
            'schedule': 300,  # Every 5 minutes
        },
    },
)

# Auto-discover tasks
celery_app.autodiscover_tasks()


# Celery signal handlers
@worker_ready.connect
def worker_ready_handler(sender, **kwargs):
    """Handle worker ready signal"""
    logger.info(f"Worker {sender} is ready")

    # Initialize shared cache and check GPU
    try:
        from core.config import get_cache_paths
        cache_paths = get_cache_paths()
        logger.info(f"Cache paths initialized: {cache_paths.root}")

        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
            logger.info(f"GPU available: {gpu_name} ({gpu_memory}GB)")
        else:
            logger.warning("No GPU available, running on CPU")

    except Exception as e:
        logger.error(f"Worker initialization failed: {e}")


@worker_shutdown.connect
def worker_shutdown_handler(sender, **kwargs):
    """Handle worker shutdown signal"""
    logger.info(f"Worker {sender} shutting down")

    # Clean up GPU memory
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cleaned up")
    except Exception as e:
        logger.warning(f"GPU cleanup failed: {e}")


@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
    """Handle task pre-run"""
    logger.info(f"Starting task {task.name} [{task_id}]") # type: ignore

    # Update job status
    try:
        tracker = JobTracker()
        tracker.set_job_status(task_id, "running", {
            "task_name": task.name, # type: ignore
            "started_at": datetime.now().isoformat()
        })
    except Exception as e:
        logger.warning(f"Failed to update job status: {e}")


@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **kwds):
    """Handle task post-run"""
    logger.info(f"Completed task {task.name} [{task_id}] with state: {state}") # type: ignore

    # Update job status
    try:
        tracker = JobTracker()

        status = "completed" if state == "SUCCESS" else "failed"
        metadata = {
            "task_name": task.name, # type: ignore
            "completed_at": datetime.now().isoformat(),
            "final_state": state
        }

        if retval and isinstance(retval, dict):
            metadata.update(retval)

        tracker.set_job_status(task_id, status, metadata)

    except Exception as e:
        logger.warning(f"Failed to update job completion status: {e}")


@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwargs):
    """Handle task failure"""
    logger.error(f"Task {sender.name} [{task_id}] failed: {exception}") # type: ignore

    # Update job status with error
    try:
        tracker = JobTracker()
        tracker.set_job_status(task_id, "failed", {
            "task_name": sender.name, # type: ignore
            "error": str(exception),
            "failed_at": datetime.now().isoformat()
        })
    except Exception as e:
        logger.warning(f"Failed to update job failure status: {e}")


if __name__ == '__main__':
    celery_app.start()# 續完 core/train/lora_trainer.py
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer for training"""
        # Get trainable parameters
        params_to_optimize = self.unet.parameters()

        # Choose optimizer
        optimizer_name = self.config.get("optimizer", "adamw")
        learning_rate = self.config.get("learning_rate", 1e-4)

        if optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                params_to_optimize,
                lr=learning_rate,
                betas=self.config.get("adam_beta1", 0.9), self.config.get("adam_beta2", 0.999)), # type: ignore
                weight_decay=self.config.get("adam_weight_decay", 1e-2),
                eps=self.config.get("adam_epsilon", 1e-08)
            )
        elif optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(params_to_optimize, lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        return optimizer

    def _setup_lr_scheduler(self, optimizer: torch.optim.Optimizer, num_training_steps_per_epoch: int):
        """Setup learning rate scheduler"""
        lr_scheduler_name = self.config.get("lr_scheduler", "constant")

        if lr_scheduler_name == "constant":
            lr_scheduler = get_scheduler(
                "constant",
                optimizer=optimizer,
                num_warmup_steps=self.config.get("lr_warmup_steps", 0),
                num_training_steps=self.config.get("max_train_steps", 1000)
            )
        elif lr_scheduler_name == "cosine":
            lr_scheduler = get_scheduler(
                "cosine",
                optimizer=optimizer,
                num_warmup_steps=self.config.get("lr_warmup_steps", 0),
                num_training_steps=self.config.get("max_train_steps", 1000)
            )
        else:
            lr_scheduler = get_scheduler(
                lr_scheduler_name,
                optimizer=optimizer,
                num_warmup_steps=self.config.get("lr_warmup_steps", 0),
                num_training_steps=self.config.get("max_train_steps", 1000)
            )

        return lr_scheduler

    def _compute_snr(self, timesteps):
        """Compute signal-to-noise ratio for timesteps"""
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod)**0.5

        # Expand the tensors
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]

