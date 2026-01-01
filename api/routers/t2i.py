"""Text-to-Image (T2I) API router.

This router is intentionally thin: it adapts the React UI contract to the
core `T2IPipelineManager` implementation.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from api.file_stream import stream_file
from api.schemas.t2i import GenerateRequest, GenerateResponse, JobStatusResponse, SubmitResponse
from api.t2i_cost import estimate_t2i_cost
from api.t2i_jobs import (
    T2IJobManager,
    job_dir,
    read_access_owner,
    run_generate_sync,
    validate_generate_request,
    write_access_meta,
)
from api.t2i_tokens import make_image_token, verify_image_token
from core.config import get_app_paths, get_settings

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
    settings = get_settings()
    dispatch_mode = str(settings.api.t2i_dispatch_mode or "redis").lower()
    worker_enabled = bool(settings.api.t2i_worker_enabled) and dispatch_mode != "celery"
    manager = T2IJobManager(
        redis_url=getattr(request.app.state, "redis_url", None),
        worker_enabled=worker_enabled,
        dispatch_mode=dispatch_mode,
        job_ttl_seconds=int(settings.api.t2i_job_ttl_seconds or 0),
        stale_seconds=int(settings.api.t2i_job_stale_seconds or 0),
        max_attempts=int(settings.api.t2i_job_max_attempts or 1),
        max_concurrent_per_owner=int(settings.api.t2i_max_concurrent or 1),
        max_global_concurrent=int(settings.api.t2i_max_global_concurrent or 0),
    )
    request.app.state.t2i_job_manager = manager
    return manager


def _is_admin(request: Request) -> bool:
    return getattr(request.state, "auth_role", "anonymous") == "admin"


def _job_owner(manager: T2IJobManager, job_id: str) -> str | None:
    owner = manager.get_owner(job_id)
    if owner:
        return owner
    return read_access_owner(job_id)


def _require_job_access(request: Request, manager: T2IJobManager, job_id: str) -> str:
    owner = _job_owner(manager, job_id)
    if owner is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if _is_admin(request):
        return owner
    client_key = getattr(request.state, "client_key", "ip:unknown")
    if client_key != owner:
        raise HTTPException(status_code=403, detail="Forbidden")
    return owner


def _image_url(request: Request, *, job_id: str, filename: str, owner: str) -> str:
    base = str(request.base_url).rstrip("/")
    url = f"{base}/api/v1/t2i/images/{job_id}/{filename}"
    if getattr(request.app.state, "auth_enabled", False):
        ttl = int(get_settings().api.t2i_image_token_ttl_seconds or 0)
        token = make_image_token(job_id=job_id, filename=filename, owner=owner, ttl_seconds=ttl)
        if token:
            url = f"{url}?img_token={token}"
    return url


def _enforce_t2i_cost_limit(
    request: Request, *, width: int, height: int, steps: int, batch_size: int
) -> None:
    settings = get_settings()
    limit = int(settings.api.t2i_cost_rate_limit or 0)
    if limit <= 0:
        return

    cost = estimate_t2i_cost(width=width, height=height, steps=steps, batch_size=batch_size)
    client_key = getattr(request.state, "client_key", "ip:unknown")
    rate_limiter = getattr(request.app.state, "rate_limiter", None)
    if rate_limiter is None:
        return
    result = rate_limiter.check(
        key=f"t2i_cost:{client_key}",
        limit=limit,
        cost=cost,
        redis_url=getattr(request.app.state, "redis_url", None),
    )
    if result.allowed:
        return

    retry_after = max(0, result.reset_epoch - int(time.time()))
    metrics = getattr(request.app.state, "metrics", None)
    if metrics is not None:
        try:
            metrics.inc_rate_limited(bucket="t2i_cost")
        except Exception:
            pass
    raise HTTPException(
        status_code=429,
        detail={
            "error": "T2I_COST_LIMITED",
            "message": "T2I cost rate limit exceeded",
            "details": {
                "bucket": "t2i_cost",
                "cost": cost,
                "limit": result.limit,
                "remaining": result.remaining,
                "reset": result.reset_epoch,
            },
        },
        headers={
            "Retry-After": str(retry_after),
            "X-RateLimit-Bucket": "t2i_cost",
            "X-RateLimit-Bucket-Limit": str(result.limit),
            "X-RateLimit-Bucket-Remaining": str(result.remaining),
            "X-RateLimit-Bucket-Reset": str(result.reset_epoch),
        },
    )


@router.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, request: Request):
    job_id = str(uuid.uuid4())
    owner = getattr(request.state, "client_key", "ip:unknown")

    _enforce_t2i_cost_limit(
        request,
        width=req.width,
        height=req.height,
        steps=req.steps,
        batch_size=req.batch_size,
    )

    try:
        payload = await asyncio.to_thread(run_generate_sync, req, job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    write_access_meta(job_id, owner=owner)
    image_urls = [
        _image_url(request, job_id=job_id, filename=name, owner=owner)
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

    _enforce_t2i_cost_limit(
        request,
        width=req.width,
        height=req.height,
        steps=req.steps,
        batch_size=req.batch_size,
    )

    manager = _job_manager(request)
    owner = getattr(request.state, "client_key", "ip:unknown")

    max_queue = int(get_settings().api.t2i_max_queue or 0)
    if max_queue > 0:
        counts = manager.owner_counts(owner)
        active = int(counts.get("queued", 0)) + int(counts.get("running", 0))
        if active >= max_queue:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "T2I_QUEUE_FULL",
                    "message": "Too many queued jobs",
                    "details": {
                        "max_queue": max_queue,
                        "queued": int(counts.get("queued", 0)),
                        "running": int(counts.get("running", 0)),
                    },
                },
            )

    max_global_queue = int(get_settings().api.t2i_max_global_queue or 0)
    if max_global_queue > 0:
        counts = manager.global_counts()
        active = int(counts.get("queued", 0)) + int(counts.get("running", 0))
        if active >= max_global_queue:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "T2I_GLOBAL_QUEUE_FULL",
                    "message": "T2I service is busy",
                    "details": {
                        "max_global_queue": max_global_queue,
                        "queued": int(counts.get("queued", 0)),
                        "running": int(counts.get("running", 0)),
                    },
                },
            )

    job_id = manager.submit(req, owner=owner)
    write_access_meta(job_id, owner=owner)
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
    owner = _require_job_access(request, manager, job_id)
    snapshot = manager.get(job_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Job not found")

    image_urls = [
        _image_url(request, job_id=job_id, filename=name, owner=owner)
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
    owner = _require_job_access(request, manager, job_id)
    snapshot = manager.cancel(job_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Job not found")

    image_urls = [
        _image_url(request, job_id=job_id, filename=name, owner=owner)
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
async def get_image(
    job_id: str, filename: str, request: Request, img_token: str | None = None
):
    if "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    manager = _job_manager(request)
    owner = _job_owner(manager, job_id) or ""
    if not owner:
        raise HTTPException(status_code=404, detail="Job not found")

    if not _is_admin(request):
        client_key = getattr(request.state, "client_key", "ip:unknown")
        if client_key != owner and not (
            img_token
            and verify_image_token(
                img_token, job_id=job_id, filename=filename, owner=owner
            )
        ):
            raise HTTPException(status_code=403, detail="Forbidden")

    root = job_dir(job_id)
    path = _safe_resolve(root / filename, root)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    media_type = "image/png"
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        media_type = "image/jpeg"
    elif path.suffix.lower() == ".webp":
        media_type = "image/webp"

    return stream_file(path, media_type=media_type, filename=path.name, disposition="inline")


@router.get("/jobs")
async def list_jobs(
    request: Request,
    limit: int = 50,
    status: str | None = None,
    all: bool = False,
):
    manager = _job_manager(request)
    is_admin = _is_admin(request)
    if all and not is_admin:
        raise HTTPException(status_code=403, detail="Forbidden")

    owner = None if (all and is_admin) else getattr(request.state, "client_key", "ip:unknown")
    jobs = manager.list_jobs(owner=owner, limit=limit, status=status)
    if not is_admin:
        for job in jobs:
            job.pop("owner", None)
    return {"count": len(jobs), "jobs": jobs}


@router.delete("/jobs/{job_id}")
async def delete_job(
    job_id: str,
    request: Request,
    delete_outputs: bool = True,
):
    manager = _job_manager(request)
    _require_job_access(request, manager, job_id)

    snapshot = manager.get(job_id)
    if snapshot and str(snapshot.get("status") or "") == "running":
        raise HTTPException(status_code=409, detail="Job is running; cancel first")

    if snapshot and str(snapshot.get("status") or "") == "queued":
        manager.cancel(job_id)

    deleted_record = manager.delete_job(job_id)
    deleted_outputs = False
    if delete_outputs:
        root = job_dir(job_id)
        try:
            shutil.rmtree(root)
            deleted_outputs = True
        except FileNotFoundError:
            deleted_outputs = False
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "status": "deleted",
        "job_id": job_id,
        "deleted_record": bool(deleted_record),
        "deleted_outputs": bool(deleted_outputs),
    }


@router.post("/jobs/cleanup")
async def cleanup_jobs(
    request: Request,
    ttl_seconds: int | None = None,
    dry_run: bool = True,
    delete_records: bool = False,
    all: bool = False,
    only_terminal: bool = True,
    limit: int = 200,
):
    manager = _job_manager(request)
    is_admin = _is_admin(request)
    if all and not is_admin:
        raise HTTPException(status_code=403, detail="Forbidden")

    settings = get_settings()
    ttl = int(ttl_seconds if ttl_seconds is not None else (settings.api.t2i_output_ttl_seconds or 0))
    if ttl <= 0:
        raise HTTPException(status_code=400, detail="ttl_seconds must be > 0 (or set API_T2I_OUTPUT_TTL_SECONDS)")

    limit = max(1, min(int(limit or 200), 2000))
    owner = None if (all and is_admin) else getattr(request.state, "client_key", "ip:unknown")

    root = get_app_paths().outputs / "t2i"
    now = time.time()
    scanned = 0
    candidates = 0
    skipped_active = 0
    deleted_outputs = 0
    deleted_records = 0

    if not root.exists():
        return {
            "dry_run": bool(dry_run),
            "ttl_seconds": ttl,
            "owner": owner,
            "scanned": 0,
            "candidates": 0,
            "skipped_active": 0,
            "deleted_outputs": 0,
            "deleted_records": 0,
        }

    for entry in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if candidates >= limit:
            break
        if not entry.is_dir():
            continue
        job_id = entry.name
        scanned += 1

        meta_path = entry / "_access.json"
        meta_owner = None
        created_at = 0.0
        try:
            raw = meta_path.read_text(encoding="utf-8")
            meta = json.loads(raw)
            if isinstance(meta, dict):
                meta_owner = meta.get("owner")
                created_at = float(meta.get("created_at") or 0.0)
        except Exception:
            meta_owner = None
            created_at = 0.0

        if owner and str(meta_owner or "") != owner:
            continue

        age = now - (created_at or entry.stat().st_mtime)
        if age < ttl:
            continue

        snapshot = manager.get(job_id)
        if only_terminal and snapshot and str(snapshot.get("status") or "") in {"queued", "running"}:
            skipped_active += 1
            continue

        candidates += 1
        if dry_run:
            continue

        try:
            shutil.rmtree(entry)
            deleted_outputs += 1
        except FileNotFoundError:
            pass
        except Exception:
            continue

        if delete_records:
            if manager.delete_job(job_id):
                deleted_records += 1

    return {
        "dry_run": bool(dry_run),
        "ttl_seconds": ttl,
        "owner": owner,
        "scanned": scanned,
        "candidates": candidates,
        "skipped_active": skipped_active,
        "deleted_outputs": deleted_outputs,
        "deleted_records": deleted_records,
    }
