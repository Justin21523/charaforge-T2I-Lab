"""Fine-tuning (training) endpoints.

This router focuses on LoRA training submission + status tracking. It uses Celery
when available, but keeps the API contract stable even when workers/Redis are
offline.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.config import get_cache_paths, get_settings

router = APIRouter(prefix="/finetune", tags=["finetune"])


def _get_celery_app():
    try:
        from workers.celery_app import celery_app

        return celery_app
    except Exception:
        return None


def _resolve_dataset_path(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="dataset_path is required")

    cache_paths = get_cache_paths()
    base_root = cache_paths.datasets.resolve()

    candidate: Path
    p = Path(raw)
    if p.is_absolute():
        candidate = p
    else:
        # Allow shorthand dataset name (maps to `<datasets_root>/raw/<name>`),
        # or allow relative paths under the project dataset root.
        if "/" not in raw and "\\" not in raw:
            candidate = base_root / "raw" / raw
        else:
            candidate = base_root / raw

    resolved = candidate.resolve()
    if base_root not in resolved.parents and resolved != base_root:
        raise HTTPException(
            status_code=400,
            detail=f"dataset_path must be under {base_root}",
        )
    if not resolved.exists():
        raise HTTPException(status_code=400, detail=f"dataset_path not found: {resolved}")
    if not resolved.is_dir():
        raise HTTPException(status_code=400, detail="dataset_path must be a directory")

    return str(resolved)


class LoRATrainRequest(BaseModel):
    project_name: str = Field(..., min_length=1, max_length=100)
    dataset_path: str = Field(..., min_length=1, max_length=2000)
    instance_prompt: str = Field(..., min_length=1, max_length=2000)

    # Base model selection
    base_model: Optional[str] = Field(default=None, max_length=2000)
    model_type: str = Field(default="sdxl")  # "sd15" | "sdxl"

    # LoRA params
    lora_rank: int = Field(default=16, ge=1, le=128)
    lora_alpha: int = Field(default=32, ge=1, le=256)
    lora_dropout: float = Field(default=0.1, ge=0.0, le=1.0)

    # Training params
    num_train_epochs: int = Field(default=10, ge=1, le=200)
    train_batch_size: int = Field(default=1, ge=1, le=8)
    learning_rate: float = Field(default=1e-4, gt=0.0, le=1e-2)
    gradient_checkpointing: bool = Field(default=True)
    gradient_accumulation_steps: int = Field(default=1, ge=1, le=64)
    mixed_precision: str = Field(default="fp16")
    resolution: int = Field(default=768, ge=256, le=2048)
    max_train_steps: Optional[int] = Field(default=None, ge=1)
    save_steps: int = Field(default=500, ge=1)
    validation_steps: int = Field(default=100, ge=1)
    max_samples: Optional[int] = Field(default=None, ge=1)


@router.post("/lora/train")
async def submit_lora_train(req: LoRATrainRequest) -> Dict[str, Any]:
    settings = get_settings()

    base_model = req.base_model
    if not base_model:
        base_model = (
            settings.model.default_sdxl_model
            if (req.model_type or "").lower() == "sdxl"
            else settings.model.default_sd15_model
        )

    job_id = f"lora_{req.project_name}_{uuid.uuid4().hex[:8]}"
    celery_app = _get_celery_app()

    resolved_dataset_path = _resolve_dataset_path(req.dataset_path)

    task_config = {
        "project_name": req.project_name,
        "base_model": base_model,
        "dataset_path": resolved_dataset_path,
        "instance_prompt": req.instance_prompt,
        "lora_rank": req.lora_rank,
        "lora_alpha": req.lora_alpha,
        "lora_dropout": req.lora_dropout,
        "num_train_epochs": req.num_train_epochs,
        "train_batch_size": req.train_batch_size,
        "learning_rate": req.learning_rate,
        "gradient_checkpointing": req.gradient_checkpointing,
        "gradient_accumulation_steps": req.gradient_accumulation_steps,
        "mixed_precision": req.mixed_precision,
        "resolution": req.resolution,
        "max_train_steps": req.max_train_steps,
        "save_steps": req.save_steps,
        "validation_steps": req.validation_steps,
        "max_samples": req.max_samples,
    }

    if celery_app is None:
        return {
            "job_id": job_id,
            "status": "queued",
            "submitted_at": datetime.now().isoformat(),
            "note": "Celery is not available in this process; start Redis + worker to execute training.",
        }

    try:
        celery_app.send_task(
            "workers.tasks.training.train_lora",
            args=[task_config],
            task_id=job_id,
            queue="training",
        )
    except Exception as exc:
        return {
            "job_id": job_id,
            "status": "queued",
            "submitted_at": datetime.now().isoformat(),
            "note": f"Failed to enqueue task (workers/Redis offline?): {exc}",
        }

    return {
        "job_id": job_id,
        "status": "queued",
        "submitted_at": datetime.now().isoformat(),
    }


@router.get("/lora/status/{job_id}")
async def lora_status(job_id: str) -> Dict[str, Any]:
    celery_app = _get_celery_app()
    if celery_app is None:
        return {"job_id": job_id, "status": "UNKNOWN", "note": "Celery not available"}

    try:
        result = celery_app.AsyncResult(job_id)
        payload: Dict[str, Any] = {"job_id": job_id, "status": result.status}
        if result.status == "PROGRESS":
            payload["progress"] = result.info
        elif result.status == "SUCCESS":
            payload["result"] = result.result
        elif result.status == "FAILURE":
            payload["error"] = str(result.info)
        return payload
    except Exception as exc:
        return {"job_id": job_id, "status": "ERROR", "error": str(exc)}


@router.post("/lora/cancel/{job_id}")
async def cancel_lora(job_id: str) -> Dict[str, Any]:
    celery_app = _get_celery_app()
    if celery_app is None:
        raise HTTPException(status_code=503, detail="Celery not available")

    try:
        celery_app.control.revoke(job_id, terminate=True)
        return {"job_id": job_id, "status": "cancelled", "cancelled_at": datetime.now().isoformat()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
