"""Health check endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import psutil

try:
    import redis  # type: ignore
except ImportError:  # pragma: no cover
    redis = None  # type: ignore
import torch
from fastapi import APIRouter

from core.config import get_cache_paths, get_settings

router = APIRouter(tags=["health"])


def _gpu_info() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}

    try:
        props = torch.cuda.get_device_properties(0)
        return {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "device_name": props.name,
            "memory_total": props.total_memory,
            "memory_allocated": torch.cuda.memory_allocated(0),
            "memory_reserved": torch.cuda.memory_reserved(0),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def _redis_info(redis_url: str) -> Dict[str, Any]:
    if redis is None:
        return {"status": "unavailable", "error": "redis package not installed"}
    try:
        client = redis.from_url(redis_url, socket_connect_timeout=0.2, socket_timeout=0.2)
        client.ping()
        info = client.info()
        return {
            "status": "connected",
            "version": info.get("redis_version"),
            "used_memory": info.get("used_memory_human"),
            "connected_clients": info.get("connected_clients"),
        }
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc)}


@router.get("/health")
@router.get("/healthz")
async def health() -> Dict[str, Any]:
    """Basic health endpoint used by the React UI."""
    settings = get_settings()
    cache_paths = get_cache_paths()

    system = {}
    try:
        system = {
            "cpu_percent": psutil.cpu_percent(interval=0.2),
            "memory_percent": psutil.virtual_memory().percent,
        }
    except Exception as exc:
        system = {"error": str(exc)}

    components: Dict[str, Any] = {
        "gpu": _gpu_info(),
        "redis": _redis_info(settings.redis_url),
        "cache": {
            "cache_root": str(cache_paths.root),
            "models_root": str(cache_paths.models),
            "datasets_root": str(cache_paths.datasets),
            "outputs_root": str(cache_paths.outputs),
        },
        "system": system,
    }

    status = "ok"
    if components["redis"].get("status") != "connected":
        status = "degraded"

    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "version": "0.2.0",
        "environment": settings.environment,
        "cache_root": str(cache_paths.root),
        "gpu_available": bool(components["gpu"].get("available")),
        "gpu_count": components["gpu"].get("device_count", 0),
        "components": components,
    }


@router.get("/readiness")
async def readiness() -> Dict[str, Any]:
    settings = get_settings()
    cache_paths = get_cache_paths()
    ready = cache_paths.root.exists()
    redis_state = _redis_info(settings.redis_url)
    ready = ready and redis_state.get("status") == "connected"
    return {"ready": bool(ready), "timestamp": datetime.now().isoformat()}


@router.get("/liveness")
async def liveness() -> Dict[str, Any]:
    return {"alive": True, "timestamp": datetime.now().isoformat()}
