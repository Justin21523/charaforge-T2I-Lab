"""LoRA management endpoints."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.config import get_settings
from core.t2i.pipeline import PIPELINE_LOCK, get_pipeline_manager
from core.train.registry import get_model_registry

router = APIRouter(prefix="/lora", tags=["lora"])


class LoadLoRARequest(BaseModel):
    lora_id: str = Field(..., min_length=1, max_length=500)
    weight: float = Field(default=1.0, ge=0.0, le=2.0)


class UnloadLoRARequest(BaseModel):
    lora_id: str = Field(..., min_length=1, max_length=500)


@router.get("/list")
async def list_loras() -> List[Dict[str, Any]]:
    registry = get_model_registry()
    loras = registry.list_models(model_type="lora")
    return [
        {
            "id": entry.name,
            "name": entry.name.split("/", 1)[-1],
            "type": "lora",
            "size_mb": entry.size_mb,
            "description": entry.description,
            "tags": entry.tags,
            "path": entry.path,
            "created_at": entry.created_at,
            "last_used": entry.last_used,
        }
        for entry in loras
    ]


@router.get("/status")
async def lora_status() -> Dict[str, Any]:
    manager = get_pipeline_manager()
    return {
        "pipeline_loaded": bool(manager.pipeline_loaded),
        "base_model": manager.current_model,
        "loaded_loras": manager.lora_manager.list_loaded_loras(),
        "lora_stack": manager.lora_manager.lora_stack.to_dict(),
    }


@router.post("/load")
async def load_lora(req: LoadLoRARequest) -> Dict[str, Any]:
    settings = get_settings()
    registry = get_model_registry()
    if not registry.get_model(req.lora_id):
        raise HTTPException(status_code=404, detail=f"LoRA not found: {req.lora_id}")

    manager = get_pipeline_manager()
    with PIPELINE_LOCK:
        if not manager.pipeline_loaded:
            if not manager.load_model(settings.model.default_sdxl_model):
                raise HTTPException(status_code=503, detail="Base model not available")

        existing = [
            {"name": lora.name, "scale": lora.scale, "enabled": lora.enabled}
            for lora in manager.lora_manager.lora_stack.loras
            if lora.name != req.lora_id
        ]
        existing.append({"name": req.lora_id, "scale": req.weight, "enabled": True})

        if not manager.load_lora_stack(existing):
            raise HTTPException(status_code=400, detail="Failed to apply LoRA stack")

        return {
            "status": "success",
            "loaded_loras": manager.lora_manager.list_loaded_loras(),
            "lora_stack": manager.lora_manager.lora_stack.to_dict(),
        }


@router.post("/unload")
async def unload_lora(req: UnloadLoRARequest) -> Dict[str, Any]:
    manager = get_pipeline_manager()
    with PIPELINE_LOCK:
        if not manager.pipeline_loaded:
            return {"status": "success", "loaded_loras": [], "lora_stack": {"loras": []}}

        remaining = [
            {"name": lora.name, "scale": lora.scale, "enabled": lora.enabled}
            for lora in manager.lora_manager.lora_stack.loras
            if lora.name != req.lora_id
        ]

        if not remaining:
            manager.unload_loras()
        else:
            if not manager.load_lora_stack(remaining):
                raise HTTPException(status_code=400, detail="Failed to re-apply LoRA stack")

        return {
            "status": "success",
            "loaded_loras": manager.lora_manager.list_loaded_loras(),
            "lora_stack": manager.lora_manager.lora_stack.to_dict(),
        }
