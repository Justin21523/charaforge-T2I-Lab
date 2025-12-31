"""Model registry + scanning endpoints."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from core.train.registry import get_model_registry

router = APIRouter(prefix="/models", tags=["models"])

_SCAN_LOCK = asyncio.Lock()


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
    async with _SCAN_LOCK:
        return registry.scan_filesystem(replace=req.replace)
