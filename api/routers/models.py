"""Model registry + scanning endpoints."""

from __future__ import annotations

import asyncio
import weakref
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

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
