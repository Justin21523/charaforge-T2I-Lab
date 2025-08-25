from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from celery.result import AsyncResult

from workers.celery_app import celery_app
from core.config import get_settings

# Task name must match workers.tasks.training.train_lora_task
# Route table in celery_app maps *.training.* to queue "training"

router = APIRouter(prefix="/finetune", tags=["fine-tuning"])


class LoRATrainRequest(BaseModel):
    run_id: str = Field(..., min_length=1, max_length=100)
    dataset_name: str = Field(..., min_length=1)  # unified with T2IDataset
    base_model: str = Field(default="stabilityai/stable-diffusion-xl-base-1.0")
    rank: int = Field(default=16, ge=1, le=256)
    learning_rate: float = Field(default=1e-4, gt=0, le=1e-2)
    max_train_steps: int = Field(default=1000, ge=50, le=200000)
    train_batch_size: int = Field(default=1, ge=1, le=8)
    gradient_accumulation_steps: int = Field(default=8, ge=1, le=64)
    mixed_precision: str = Field(default="bf16")
    validation_prompts: List[str] = Field(default=[])
    notes: Optional[str] = None


class TrainResponse(BaseModel):
    job_id: str
    run_id: str
    status: str
    created_at: datetime
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    run_id: str
    status: str  # pending/running/completed/failed
    progress: float = 0.0
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    loss: Optional[float] = None
    eta_minutes: Optional[int] = None
    error_message: Optional[str] = None
    artifacts: Dict[str, Any] = {}


@router.post("/lora/train", response_model=TrainResponse)
async def train_lora(req: LoRATrainRequest):
    try:
        # Pass (run_id, config) to match workers.tasks.training.train_lora_task signature
        job = celery_app.send_task(
            name="train_lora", args=[req.run_id, req.dict()], queue="training"
        )
        return TrainResponse(
            job_id=job.id,
            run_id=req.run_id,
            status="pending",
            created_at=datetime.utcnow(),
            message=f"submitted: {job.id}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def job_status(job_id: str):
    r = AsyncResult(job_id, app=celery_app)
    meta = r.info or {}
    state = r.state  # PENDING/STARTED/PROGRESS/SUCCESS/FAILURE

    status_map = {
        "PENDING": "pending",
        "STARTED": "running",
        "PROGRESS": "running",
        "SUCCESS": "completed",
        "FAILURE": "failed",
    }

    # Normalize optional fields
    progress = float(meta.get("progress", 0.0))
    total_steps = meta.get("total_steps")
    current_step = meta.get("current_step")

    return JobStatusResponse(
        job_id=job_id,
        run_id=meta.get("run_id", ""),
        status=status_map.get(state, "pending"),
        progress=progress if progress <= 1.0 else min(1.0, progress),
        current_step=current_step,
        total_steps=total_steps,
        loss=meta.get("loss"),
        eta_minutes=meta.get("eta_minutes"),
        artifacts=meta.get("artifacts", {}),
        error_message=meta.get("error"),
    )
