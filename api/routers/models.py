"""Model registry + scanning endpoints."""

from __future__ import annotations

import asyncio
import weakref
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from api.model_scan_jobs import ModelScanJobManager
from core.config import get_settings
from core.train.registry import get_model_registry

router = APIRouter(prefix="/models", tags=["models"])

_SCAN_LOCKS: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = (
    weakref.WeakKeyDictionary()
)


def _scan_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _SCAN_LOCKS.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _SCAN_LOCKS[loop] = lock
    return lock


def _scan_job_manager(request: Request) -> ModelScanJobManager:
    manager = getattr(request.app.state, "model_scan_job_manager", None)
    if isinstance(manager, ModelScanJobManager):
        return manager

    settings = get_settings()
    manager = ModelScanJobManager(
        redis_url=getattr(request.app.state, "redis_url", None),
        worker_enabled=bool(settings.api.models_scan_worker_enabled),
        job_ttl_seconds=int(settings.api.models_scan_job_ttl_seconds or 0),
    )
    request.app.state.model_scan_job_manager = manager
    return manager


def _is_admin(request: Request) -> bool:
    return getattr(request.state, "auth_role", "anonymous") == "admin"


def _require_job_access(
    request: Request, manager: ModelScanJobManager, job_id: str
) -> str:
    owner = manager.get_owner(job_id)
    if not owner:
        raise HTTPException(status_code=404, detail="Job not found")
    if _is_admin(request):
        return owner
    client_key = getattr(request.state, "client_key", "ip:unknown")
    if client_key != owner:
        raise HTTPException(status_code=403, detail="Forbidden")
    return owner


class ScanRequest(BaseModel):
    replace: bool = Field(default=False, description="Rebuild registry from disk")


@router.get("")
async def list_models(
    model_type: Optional[str] = None,
    q: Optional[str] = None,
) -> Dict[str, Any]:
    registry = get_model_registry()
    if q:
        models = registry.search_models(q)
    else:
        models = registry.list_models(model_type=model_type)

    return {
        "count": len(models),
        "models": [m.to_dict() for m in models],
        "registry_path": str(registry.registry_path),
    }


@router.post("/scan")
async def scan_models(req: ScanRequest) -> Dict[str, Any]:
    registry = get_model_registry()
    async with _scan_lock():
        # NOTE: This endpoint is admin-only + rate-limited; we run the scan
        # synchronously to avoid executor/loop shutdown edge cases observed in
        # our runtime when using thread offloading.
        result = registry.scan_filesystem(req.replace)
        if str(result.get("status") or "") == "busy":
            raise HTTPException(
                status_code=409,
                detail={
                    "error": str(result.get("error") or "SCAN_IN_PROGRESS"),
                    "message": str(result.get("message") or "Model scan already in progress"),
                },
            )
        return result


@router.post("/scan/submit")
async def submit_scan(req: ScanRequest, request: Request) -> Dict[str, Any]:
    manager = _scan_job_manager(request)
    owner = getattr(request.state, "client_key", "ip:unknown")
    job_id, created = manager.submit(owner=owner, replace=req.replace)
    if not created:
        details: Dict[str, Any] = {}
        if job_id:
            details["job_id"] = job_id
        raise HTTPException(
            status_code=409,
            detail={
                "error": "MODELS_SCAN_JOB_IN_PROGRESS",
                "message": "Model scan job already queued or running",
                "details": details,
            },
        )

    base = str(request.base_url).rstrip("/")
    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"{base}/api/v1/models/scan/status/{job_id}",
        "cancel_url": f"{base}/api/v1/models/scan/cancel/{job_id}",
    }


@router.get("/scan/status/{job_id}")
async def scan_status(job_id: str, request: Request) -> Dict[str, Any]:
    manager = _scan_job_manager(request)
    _require_job_access(request, manager, job_id)
    snapshot = manager.get(job_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return snapshot


@router.post("/scan/cancel/{job_id}")
async def cancel_scan(job_id: str, request: Request) -> Dict[str, Any]:
    manager = _scan_job_manager(request)
    _require_job_access(request, manager, job_id)
    result = manager.cancel(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if not result.canceled:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "MODELS_SCAN_JOB_NOT_CANCELLABLE",
                "message": "Model scan job cannot be canceled once running",
            },
        )
    return result.snapshot
