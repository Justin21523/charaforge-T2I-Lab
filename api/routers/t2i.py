"""Text-to-Image (T2I) API router.

This router is intentionally thin: it adapts the React UI contract to the
core `T2IPipelineManager` implementation.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from api.schemas.t2i import GenerateRequest, GenerateResponse, JobStatusResponse, SubmitResponse
from api.t2i_jobs import T2IJobManager, job_dir, run_generate_sync, validate_generate_request

router = APIRouter(prefix="/t2i", tags=["t2i"])


def _safe_resolve(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if root_resolved not in resolved.parents and resolved != root_resolved:
        raise HTTPException(status_code=400, detail="Invalid path")
    return resolved


def _job_manager(request: Request) -> T2IJobManager:
    manager = getattr(request.app.state, "t2i_job_manager", None)
    if isinstance(manager, T2IJobManager):
        return manager
    manager = T2IJobManager(redis_url=getattr(request.app.state, "redis_url", None))
    request.app.state.t2i_job_manager = manager
    return manager


@router.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, request: Request):
    job_id = str(uuid.uuid4())

    try:
        payload = await asyncio.to_thread(run_generate_sync, req, job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    base = str(request.base_url).rstrip("/")
    image_urls = [
        f"{base}/api/v1/t2i/images/{job_id}/{name}"
        for name in payload["saved_paths"]
        if Path(name).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]

    return GenerateResponse(
        status="success",
        job_id=job_id,
        image_path=image_urls,
        seed=payload["seed"],
        elapsed_ms=payload["elapsed_ms"],
        metadata=payload["metadata"] or {},
    )


@router.post("/submit", response_model=SubmitResponse)
async def submit(req: GenerateRequest, request: Request):
    try:
        validate_generate_request(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    manager = _job_manager(request)
    owner = getattr(request.state, "client_key", "ip:unknown")
    job_id = manager.submit(req, owner=owner)
    base = str(request.base_url).rstrip("/")
    return SubmitResponse(
        status="queued",
        job_id=job_id,
        status_url=f"{base}/api/v1/t2i/status/{job_id}",
        cancel_url=f"{base}/api/v1/t2i/cancel/{job_id}",
    )


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def status(job_id: str, request: Request):
    manager = _job_manager(request)
    snapshot = manager.get(job_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Job not found")

    base = str(request.base_url).rstrip("/")
    image_urls = [
        f"{base}/api/v1/t2i/images/{job_id}/{name}"
        for name in snapshot.get("saved_paths") or []
        if Path(name).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]

    return JobStatusResponse(
        status=str(snapshot["status"]),
        job_id=job_id,
        image_path=image_urls,
        seed=snapshot.get("seed"),
        elapsed_ms=snapshot.get("elapsed_ms"),
        metadata=snapshot.get("metadata") or {},
        progress=snapshot.get("progress"),
        cancel_requested=bool(snapshot.get("cancel_requested")),
        error=snapshot.get("error"),
    )


@router.post("/cancel/{job_id}", response_model=JobStatusResponse)
async def cancel(job_id: str, request: Request):
    manager = _job_manager(request)
    snapshot = manager.cancel(job_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Job not found")

    base = str(request.base_url).rstrip("/")
    image_urls = [
        f"{base}/api/v1/t2i/images/{job_id}/{name}"
        for name in snapshot.get("saved_paths") or []
        if Path(name).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]

    return JobStatusResponse(
        status=str(snapshot["status"]),
        job_id=job_id,
        image_path=image_urls,
        seed=snapshot.get("seed"),
        elapsed_ms=snapshot.get("elapsed_ms"),
        metadata=snapshot.get("metadata") or {},
        progress=snapshot.get("progress"),
        cancel_requested=bool(snapshot.get("cancel_requested")),
        error=snapshot.get("error"),
    )


@router.get("/images/{job_id}/{filename}")
async def get_image(job_id: str, filename: str):
    if "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    root = job_dir(job_id)
    path = _safe_resolve(root / filename, root)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    media_type = "image/png"
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        media_type = "image/jpeg"
    elif path.suffix.lower() == ".webp":
        media_type = "image/webp"

    return FileResponse(path=str(path), media_type=media_type, filename=path.name)
