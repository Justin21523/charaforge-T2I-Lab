"""Batch generation endpoints.

Implementation notes:
- Uses an in-process job store (dev-friendly).
- Serializes generation via the shared `PIPELINE_LOCK`.
"""

from __future__ import annotations

import asyncio
import io
import secrets
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from core.config import get_app_paths, get_settings
from core.t2i.pipeline import PIPELINE_LOCK, GenerationParams, get_pipeline_manager

router = APIRouter(prefix="/batch", tags=["batch"])

_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = asyncio.Lock()


class BatchTask(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    negative: str = Field(default="", max_length=2000)
    width: int = Field(default=768, ge=256, le=2048)
    height: int = Field(default=768, ge=256, le=2048)
    steps: int = Field(default=25, ge=1, le=150)
    cfg_scale: float = Field(default=7.5, ge=1.0, le=30.0)
    sampler: str = Field(default="DPM++ 2M Karras", max_length=100)
    seed: int = Field(default=-1, ge=-1, le=2**32 - 1)
    batch_size: int = Field(default=1, ge=1, le=8)

    model_type: str = Field(default="sdxl")
    model: Optional[str] = Field(default=None, max_length=2000)


class BatchSubmitRequest(BaseModel):
    job_name: str = Field(default="batch_job", max_length=200)
    tasks: List[BatchTask] = Field(..., min_length=1, max_length=500)


def _batch_dir(job_id: str) -> Path:
    app_paths = get_app_paths()
    return app_paths.outputs / "batch" / job_id


def _safe_resolve(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if root_resolved not in resolved.parents and resolved != root_resolved:
        raise HTTPException(status_code=400, detail="Invalid path")
    return resolved


def _sampler_to_scheduler(sampler: str) -> Tuple[str, Dict[str, Any]]:
    name = (sampler or "").strip().lower()
    if name in {"ddim"}:
        return "DDIM", {}
    if name in {"lms"}:
        return "LMS", {}
    if name in {"euler a", "euler_a", "euler-ancestral"}:
        return "EulerAncestral", {}
    if name in {"euler", "heun"}:
        return "EulerAncestral", {}
    if name.startswith("dpm++"):
        kwargs: Dict[str, Any] = {"use_karras_sigmas": "karras" in name}
        if "sde" in name:
            kwargs["algorithm_type"] = "sde-dpmsolver++"
        return "DPMSolverMultistep", kwargs
    return "DPMSolverMultistep", {}


def _pick_model_id(model_type: str, model: Optional[str]) -> str:
    settings = get_settings()
    if model:
        return model
    return (
        settings.model.default_sdxl_model
        if (model_type or "").lower() == "sdxl"
        else settings.model.default_sd15_model
    )


def _run_one_task_sync(task: BatchTask, output_dir: Path) -> Dict[str, Any]:
    settings = get_settings()

    if task.width % 8 != 0 or task.height % 8 != 0:
        raise HTTPException(status_code=400, detail="Width/height must be multiples of 8")
    if task.batch_size > settings.api.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"batch_size exceeds limit ({settings.api.max_batch_size})",
        )
    if task.steps > settings.api.max_steps:
        raise HTTPException(
            status_code=400, detail=f"steps exceeds limit ({settings.api.max_steps})"
        )

    seed = task.seed if task.seed >= 0 else secrets.randbelow(2**32)
    model_id = _pick_model_id(task.model_type, task.model)
    scheduler_name, scheduler_kwargs = _sampler_to_scheduler(task.sampler)

    manager = get_pipeline_manager()
    with PIPELINE_LOCK:
        if not manager.pipeline_loaded or manager.current_model != model_id:
            if not manager.load_model(model_id):
                raise HTTPException(status_code=503, detail=f"Failed to load model: {model_id}")

        manager.set_scheduler(scheduler_name, **scheduler_kwargs)

        params = GenerationParams(
            prompt=task.prompt,
            negative_prompt=task.negative or None,
            width=task.width,
            height=task.height,
            num_inference_steps=task.steps,
            guidance_scale=task.cfg_scale,
            num_images_per_prompt=task.batch_size,
            seed=seed,
        )
        result = manager.generate(params)
        saved = manager.save_generation_result(result, output_dir)

    return {
        "status": "success",
        "seed": seed,
        "elapsed_ms": int(result.generation_time * 1000),
        "metadata": result.metadata,
        "saved_paths": saved,
    }


async def _run_job(job_id: str, submit: BatchSubmitRequest, base_url: str) -> None:
    started_at = datetime.now().isoformat()
    async with _JOBS_LOCK:
        job = _JOBS[job_id]
        job["status"] = "running"
        job["started_at"] = started_at

    output_dir = _batch_dir(job_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, task in enumerate(submit.tasks):
        async with _JOBS_LOCK:
            if _JOBS[job_id].get("cancel_requested"):
                _JOBS[job_id]["status"] = "cancelled"
                _JOBS[job_id]["completed_at"] = datetime.now().isoformat()
                return

        try:
            item_dir = output_dir / f"task_{index:04d}"
            payload = await asyncio.to_thread(_run_one_task_sync, task, item_dir)
            urls = [
                f"{base_url}/api/v1/batch/images/{job_id}/{item_dir.name}/{p.name}"
                for p in payload["saved_paths"]
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            ]
            item_result = {
                "task_index": index,
                "status": "success",
                "image_path": urls,
                "seed": payload["seed"],
                "elapsed_ms": payload["elapsed_ms"],
                "metadata": payload["metadata"],
            }
        except Exception as exc:
            item_result = {
                "task_index": index,
                "status": "failed",
                "error": str(exc),
            }

        async with _JOBS_LOCK:
            job = _JOBS[job_id]
            job["results"].append(item_result)
            job["completed"] += 1

    async with _JOBS_LOCK:
        _JOBS[job_id]["status"] = "completed"
        _JOBS[job_id]["completed_at"] = datetime.now().isoformat()


@router.post("/submit")
async def submit_batch(req: BatchSubmitRequest, request: Request) -> Dict[str, Any]:
    job_id = str(uuid.uuid4())
    base_url = str(request.base_url).rstrip("/")

    async with _JOBS_LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "job_name": req.job_name,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "total": len(req.tasks),
            "completed": 0,
            "results": [],
            "cancel_requested": False,
        }

    asyncio.create_task(_run_job(job_id, req, base_url))

    return {
        "job_id": job_id,
        "status": "pending",
        "total": len(req.tasks),
        "completed": 0,
    }


@router.get("/status/{job_id}")
async def get_batch_status(job_id: str) -> Dict[str, Any]:
    async with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job


@router.post("/cancel/{job_id}")
async def cancel_batch(job_id: str) -> Dict[str, Any]:
    async with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        job["cancel_requested"] = True
        if job["status"] in {"completed", "failed", "cancelled"}:
            return {"job_id": job_id, "status": job["status"]}
        job["status"] = "cancelling"
        return {"job_id": job_id, "status": "cancelling"}


@router.get("/list")
async def list_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    if limit < 1:
        limit = 1
    if limit > 200:
        limit = 200
    async with _JOBS_LOCK:
        jobs = list(_JOBS.values())[-limit:]
    return [
        {
            "job_id": j["job_id"],
            "job_name": j["job_name"],
            "status": j["status"],
            "created_at": j["created_at"],
            "total": j["total"],
            "completed": j["completed"],
        }
        for j in reversed(jobs)
    ]


@router.get("/images/{job_id}/{task_dir}/{filename}")
async def get_batch_image(job_id: str, task_dir: str, filename: str):
    if any(sep in filename for sep in ["/", "\\"]) or any(sep in task_dir for sep in ["/", "\\"]):
        raise HTTPException(status_code=400, detail="Invalid filename")

    root = _batch_dir(job_id)
    path = _safe_resolve(root / task_dir / filename, root)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    media_type = "image/png"
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        media_type = "image/jpeg"
    elif path.suffix.lower() == ".webp":
        media_type = "image/webp"

    return FileResponse(path=str(path), media_type=media_type, filename=path.name)


@router.get("/download/{job_id}")
async def download_batch_zip(job_id: str):
    async with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] != "completed":
            raise HTTPException(status_code=400, detail="Job not completed")

    root = _batch_dir(job_id)
    if not root.exists():
        raise HTTPException(status_code=404, detail="Job output not found")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in root.rglob("*"):
            if file.is_file() and file.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                zf.write(file, arcname=str(file.relative_to(root)))

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="batch_{job_id}.zip"'},
    )

