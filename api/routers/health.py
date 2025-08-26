# backend/api/health.py
"""Health check endpoints"""
import torch
import psutil
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any
import torch
from datetime import datetime
import redis

from core.config import get_settings, get_cache_paths
from workers.utils.queue_monitor import QueueMonitor

router = APIRouter()
settings = get_settings()


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_count: int
    memory_usage: Dict[str, Any]
    cache_root: str


@router.get("/healthz")
async def health_check() -> Dict[str, Any]:
    """Comprehensive health check endpoint"""

    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "0.1.0",
        "components": {},
    }

    # Check GPU
    try:
        if torch.cuda.is_available():
            gpu_info = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated": torch.cuda.memory_allocated(0),
                "memory_reserved": torch.cuda.memory_reserved(0),
                "memory_total": torch.cuda.get_device_properties(0).total_memory,
            }
        else:
            gpu_info = {"available": False, "reason": "CUDA not available"}

        health_status["components"]["gpu"] = gpu_info

    except Exception as e:
        health_status["components"]["gpu"] = {"available": False, "error": str(e)}

    # Check Redis
    try:
        redis_client = redis.from_url(settings.redis_url)
        redis_client.ping()
        redis_info = redis_client.info()

        health_status["components"]["redis"] = {
            "status": "connected",
            "version": redis_info.get("redis_version"),  # type: ignore
            "used_memory": redis_info.get("used_memory_human"),  # type: ignore
            "connected_clients": redis_info.get("connected_clients"),  # type: ignore
        }

    except Exception as e:
        health_status["components"]["redis"] = {"status": "error", "error": str(e)}
        health_status["status"] = "degraded"

    # Check system resources
    try:
        health_status["components"]["system"] = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage("/").percent,
            "load_average": (
                psutil.getloadavg() if hasattr(psutil, "getloadavg") else None
            ),
        }
    except Exception as e:
        health_status["components"]["system"] = {"error": str(e)}

    # Check cache directories
    try:
        cache_paths = get_cache_paths()
        cache_status = {
            "root_exists": cache_paths.root.exists(),
            "models_dir": cache_paths.models.exists(),
            "datasets_dir": cache_paths.datasets.exists(),
            "outputs_dir": cache_paths.outputs.exists(),
        }
        health_status["components"]["cache"] = cache_status

    except Exception as e:
        health_status["components"]["cache"] = {"error": str(e)}

    # Check worker queues
    try:
        monitor = QueueMonitor()
        queue_stats = monitor.get_queue_stats()
        health_status["components"]["workers"] = queue_stats

        if queue_stats.get("workers_online", 0) == 0:
            health_status["status"] = "degraded"

    except Exception as e:
        health_status["components"]["workers"] = {"error": str(e)}
        health_status["status"] = "degraded"

    # Overall status determination
    failed_components = [
        name
        for name, component in health_status["components"].items()
        if "error" in component or component.get("status") == "error"
    ]

    if failed_components:
        if len(failed_components) >= len(health_status["components"]) // 2:
            health_status["status"] = "unhealthy"
        else:
            health_status["status"] = "degraded"

        health_status["failed_components"] = failed_components

    return health_status


@router.get("/readiness")
async def readiness_check() -> Dict[str, Any]:
    """Kubernetes readiness probe"""
    try:
        # Quick checks for essential services
        cache_paths = get_cache_paths()

        # Check if cache root exists
        if not cache_paths.root.exists():
            return {"ready": False, "reason": "Cache root not initialized"}

        # Check Redis connection
        redis_client = redis.from_url(settings.redis_url)
        redis_client.ping()

        return {"ready": True, "timestamp": datetime.now().isoformat()}

    except Exception as e:
        return {
            "ready": False,
            "reason": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@router.get("/liveness")
async def liveness_check() -> Dict[str, Any]:
    """Kubernetes liveness probe"""
    return {
        "alive": True,
        "timestamp": datetime.now().isoformat(),
        "uptime": "running",  # Could calculate actual uptime
    }
