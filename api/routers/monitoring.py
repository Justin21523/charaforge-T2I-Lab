# api/routers/monitoring.py - System monitoring endpoints
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import psutil
import torch
from datetime import datetime, timedelta
from pathlib import Path
from core.config import get_settings, get_cache_paths
from workers.utils.queue_monitor import QueueMonitor
from workers.utils.job_tracker import JobTracker

router = APIRouter()
settings = get_settings()


class SystemResourceStatus(BaseModel):
    cpu_percent: float
    memory_percent: float
    memory_total_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_total_gb: float
    disk_free_gb: float
    gpu_info: Optional[Dict[str, Any]] = None
    load_average: Optional[List[float]] = None


class QueueStatus(BaseModel):
    active_tasks: int
    reserved_tasks: int
    scheduled_tasks: int
    workers_online: int
    queue_details: Dict[str, Any]


class JobStats(BaseModel):
    total_jobs_today: int
    completed_jobs_today: int
    failed_jobs_today: int
    success_rate_today: float
    daily_stats: Dict[str, Dict[str, int]]


@router.get("/resources", response_model=SystemResourceStatus)
async def get_system_resources():
    """Get current system resource usage"""

    try:
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # GPU information
        gpu_info = None
        if torch.cuda.is_available():
            gpu_info = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "devices": [],
            }

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_reserved = torch.cuda.memory_reserved(i)

                gpu_info["devices"].append(
                    {
                        "device_id": i,
                        "name": props.name,
                        "total_memory": props.total_memory,
                        "memory_allocated": memory_allocated,
                        "memory_reserved": memory_reserved,
                        "memory_free": props.total_memory - memory_reserved,
                        "utilization_percent": (memory_allocated / props.total_memory)
                        * 100,
                    }
                )
        else:
            gpu_info = {"available": False}

        # Load average (Linux/macOS only)
        load_avg = None
        if hasattr(psutil, "getloadavg"):
            load_avg = list(psutil.getloadavg())

        return SystemResourceStatus(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_total_gb=memory.total / (1024**3),
            memory_available_gb=memory.available / (1024**3),
            disk_usage_percent=disk.percent,
            disk_total_gb=disk.total / (1024**3),
            disk_free_gb=disk.free / (1024**3),
            gpu_info=gpu_info,
            load_average=load_avg,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get system resources: {str(e)}"
        )


@router.get("/queue", response_model=QueueStatus)
async def get_queue_status():
    """Get Celery queue status"""

    try:
        monitor = QueueMonitor()
        stats = monitor.get_queue_stats()

        return QueueStatus(
            active_tasks=stats.get("active_tasks", 0),
            reserved_tasks=stats.get("reserved_tasks", 0),
            scheduled_tasks=stats.get("scheduled_tasks", 0),
            workers_online=stats.get("workers_online", 0),
            queue_details=stats.get("queue_details", {}),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get queue status: {str(e)}"
        )


@router.get("/jobs/stats", response_model=JobStats)
async def get_job_statistics(days: int = 7):
    """Get job statistics for the last N days"""

    if days < 1 or days > 30:
        raise HTTPException(status_code=400, detail="Days must be between 1 and 30")

    try:
        tracker = JobTracker()
        stats = tracker.get_job_stats(days)

        # Calculate today's stats
        today = datetime.now().strftime("%Y-%m-%d")
        today_stats = stats.get("daily_stats", {}).get(today, {})

        completed_today = today_stats.get("completed_count", 0)
        failed_today = today_stats.get("failed_count", 0)
        total_today = today_stats.get("total_count", 0)

        success_rate = (completed_today / total_today * 100) if total_today > 0 else 0

        return JobStats(
            total_jobs_today=total_today,
            completed_jobs_today=completed_today,
            failed_jobs_today=failed_today,
            success_rate_today=success_rate,
            daily_stats=stats.get("daily_stats", {}),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get job statistics: {str(e)}"
        )


@router.get("/jobs/active")
async def get_active_jobs():
    """Get currently active jobs"""

    try:
        monitor = QueueMonitor()
        active_tasks = monitor.get_active_tasks_details()

        return {"active_tasks": active_tasks}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get active jobs: {str(e)}"
        )


@router.get("/workers/health")
async def get_worker_health():
    """Get worker health status"""

    try:
        monitor = QueueMonitor()
        health = monitor.get_worker_health()

        return health

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get worker health: {str(e)}"
        )


@router.post("/workers/restart")
async def restart_workers(worker_names: Optional[List[str]] = None):
    """Restart specific workers or all workers"""

    try:
        monitor = QueueMonitor()
        result = monitor.restart_workers(worker_names)  # type: ignore

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to restart workers: {str(e)}"
        )


@router.get("/cache/status")
async def get_cache_status():
    """Get cache directory status"""

    try:
        cache_paths = get_cache_paths()

        def get_dir_info(path: Path):
            if not path.exists():
                return {"exists": False, "size_mb": 0, "file_count": 0}

            total_size = 0
            file_count = 0

            for file in path.rglob("*"):
                if file.is_file():
                    total_size += file.stat().st_size
                    file_count += 1

            return {
                "exists": True,
                "size_mb": total_size / (1024**2),
                "file_count": file_count,
            }

        return {
            "cache_root": str(cache_paths.root),
            "directories": {
                "models": get_dir_info(cache_paths.models),
                "datasets": get_dir_info(cache_paths.datasets),
                "outputs": get_dir_info(cache_paths.outputs),
                "runs": get_dir_info(cache_paths.runs),
                "hf_cache": get_dir_info(cache_paths.cache / "hf"),
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get cache status: {str(e)}"
        )


@router.post("/cleanup")
async def cleanup_old_data():
    """Clean up old temporary data"""

    try:
        # Clean up old job records
        tracker = JobTracker()
        tracker.cleanup_old_jobs(days=7)

        # Clean up old exports
        cache_paths = get_cache_paths()
        export_dir = cache_paths.outputs / "exports"

        if export_dir.exists():
            cutoff = datetime.now() - timedelta(days=1)
            cleaned_count = 0

            for export_file in export_dir.glob("*.zip"):
                try:
                    if datetime.fromtimestamp(export_file.stat().st_mtime) < cutoff:
                        export_file.unlink()
                        cleaned_count += 1
                except Exception:
                    continue

        return {"message": "Cleanup completed", "cleaned_exports": cleaned_count}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@router.get("/alerts")
async def get_system_alerts():
    """Get system alerts and warnings"""

    alerts = []

    try:
        # Check system resources
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            alerts.append(
                {
                    "level": "critical",
                    "message": f"High memory usage: {memory.percent:.1f}%",
                    "timestamp": datetime.now().isoformat(),
                }
            )
        elif memory.percent > 80:
            alerts.append(
                {
                    "level": "warning",
                    "message": f"Elevated memory usage: {memory.percent:.1f}%",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # Check disk space
        disk = psutil.disk_usage("/")
        if disk.percent > 95:
            alerts.append(
                {
                    "level": "critical",
                    "message": f"Low disk space: {disk.percent:.1f}% used",
                    "timestamp": datetime.now().isoformat(),
                }
            )
        elif disk.percent > 85:
            alerts.append(
                {
                    "level": "warning",
                    "message": f"Disk space running low: {disk.percent:.1f}% used",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # Check GPU memory
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_reserved = torch.cuda.memory_reserved(i)
                utilization = (memory_reserved / props.total_memory) * 100

                if utilization > 95:
                    alerts.append(
                        {
                            "level": "critical",
                            "message": f"GPU {i} memory critical: {utilization:.1f}%",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                elif utilization > 85:
                    alerts.append(
                        {
                            "level": "warning",
                            "message": f"GPU {i} memory high: {utilization:.1f}%",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

        # Check workers
        try:
            monitor = QueueMonitor()
            health = monitor.get_worker_health()

            if health.get("total_workers", 0) == 0:
                alerts.append(
                    {
                        "level": "critical",
                        "message": "No workers online",
                        "timestamp": datetime.now().isoformat(),
                    }
                )
        except Exception:
            alerts.append(
                {
                    "level": "error",
                    "message": "Cannot check worker status",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return {"alerts": alerts}

    except Exception as e:
        return {
            "alerts": [
                {
                    "level": "error",
                    "message": f"Failed to check system alerts: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                }
            ]
        }
