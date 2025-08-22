# backend/api/health.py
"""Health check endpoints"""
import torch
import psutil
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_count: int
    memory_usage: Dict[str, Any]
    cache_root: str


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health and GPU availability"""
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0

    # Memory info
    memory = psutil.virtual_memory()
    memory_info = {
        "total_gb": round(memory.total / (1024**3), 2),
        "available_gb": round(memory.available / (1024**3), 2),
        "usage_percent": memory.percent,
    }

    # GPU memory if available
    if gpu_available:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        memory_info["gpu_total_gb"] = round(gpu_memory / (1024**3), 2)

    from backend.core.config import settings

    return HealthResponse(
        status="healthy",
        gpu_available=gpu_available,
        gpu_count=gpu_count,
        memory_usage=memory_info,
        cache_root=settings.AI_CACHE_ROOT,
    )
