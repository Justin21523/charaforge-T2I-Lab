from workers.celery_app import celery
from celery.result import AsyncResult
from workers.tasks.training import train_lora_task


@router.post("/lora/train", response_model=TrainResponse)
async def train_lora(request: LoRATrainRequest):
    job = train_lora_task.delay(request.dict())
    return TrainResponse(
        job_id=job.id,
        run_id=request.run_id,
        status="pending",
        created_at=datetime.now(),
        message=f"LoRA training submitted: {job.id}",
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    r = AsyncResult(job_id, app=celery)
    meta = r.info or {}
    state = r.state  # PENDING / STARTED / PROGRESS / SUCCESS / FAILURE
    status_map = {
        "PENDING": "pending",
        "STARTED": "running",
        "PROGRESS": "running",
        "SUCCESS": "completed",
        "FAILURE": "failed",
    }
    return JobStatusResponse(
        job_id=job_id,
        run_id=meta.get("run_id", ""),
        status=status_map.get(state, "pending"),
        progress=float(meta.get("progress", 0.0)),
        current_step=meta.get("current_step"),
        total_steps=meta.get("total_steps"),
        loss=meta.get("loss"),
        eta_minutes=meta.get("eta_minutes"),
        artifacts=meta.get("artifacts", {}),
        error_message=meta.get("error_message"),
    )
