"""ControlNet generation endpoints.

The React UI posts to:
  POST /api/v1/controlnet/{type}

Where `{type}` is one of: pose, depth, canny, lineart
"""

from __future__ import annotations

import asyncio
import base64
import json
import secrets
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Request
from PIL import Image
from pydantic import BaseModel, Field

from api.file_stream import stream_file
from api.t2i_cost import estimate_t2i_cost
from api.t2i_tokens import make_image_token, verify_image_token
from core.config import get_app_paths, get_settings
from core.t2i.pipeline import PIPELINE_LOCK, GenerationParams, get_pipeline_manager

router = APIRouter(prefix="/controlnet", tags=["controlnet"])

ALLOWED_CONTROL_TYPES = {"pose", "depth", "canny", "lineart"}

DEFAULT_CONTROLNET_MODELS: Dict[str, Dict[str, str]] = {
    "pose": {
        "sd15": "lllyasviel/sd-controlnet-openpose",
        "sdxl": "thibaud/controlnet-openpose-sdxl-1.0",
    },
    "depth": {
        "sd15": "lllyasviel/sd-controlnet-depth",
        "sdxl": "diffusers/controlnet-depth-sdxl-1.0",
    },
    "canny": {
        "sd15": "lllyasviel/sd-controlnet-canny",
        "sdxl": "diffusers/controlnet-canny-sdxl-1.0",
    },
    "lineart": {
        "sd15": "lllyasviel/sd-controlnet-lineart",
        "sdxl": "diffusers/controlnet-lineart-sdxl-1.0",
    },
}


def _enforce_t2i_cost_limit(request: Request, req: "ControlNetGenerateRequest") -> None:
    settings = get_settings()
    limit = int(settings.api.t2i_cost_rate_limit or 0)
    if limit <= 0:
        return

    cost = estimate_t2i_cost(
        width=req.width, height=req.height, steps=req.steps, batch_size=req.batch_size
    )
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


class ControlNetGenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    negative: str = Field(default="", max_length=2000)

    width: int = Field(default=768, ge=256, le=2048)
    height: int = Field(default=768, ge=256, le=2048)

    steps: int = Field(default=25, ge=1, le=150)
    cfg_scale: float = Field(default=7.5, ge=1.0, le=30.0)
    seed: int = Field(default=-1, ge=-1, le=2**32 - 1)
    sampler: str = Field(default="DPM++ 2M Karras", max_length=100)
    batch_size: int = Field(default=1, ge=1, le=8)

    model_type: str = Field(default="sdxl")
    model: Optional[str] = Field(default=None, max_length=2000)

    loras: List[Dict[str, Any]] = Field(default_factory=list)
    clip_skip: Optional[int] = Field(default=None, ge=1, le=12)

    # ControlNet
    control_image: str = Field(..., min_length=1, max_length=10_000_000)
    weight: float = Field(default=1.0, ge=0.0, le=2.0)
    controlnet_model: Optional[str] = Field(
        default=None, description="Registry id, HF id, or absolute path"
    )
    preprocess: bool = Field(default=True)

    canny_low_threshold: int = Field(default=100, ge=0, le=255)
    canny_high_threshold: int = Field(default=200, ge=0, le=255)


class ControlNetGenerateResponse(BaseModel):
    status: str
    job_id: str
    image_path: List[str]
    seed: int
    elapsed_ms: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


def _pick_model_id(req: ControlNetGenerateRequest) -> str:
    settings = get_settings()
    model_type = (req.model_type or "").lower()
    if model_type not in {"sd15", "sdxl"}:
        model_type = "sdxl"

    if req.model:
        return req.model

    return (
        settings.model.default_sdxl_model
        if model_type == "sdxl"
        else settings.model.default_sd15_model
    )


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


def _job_dir(job_id: str) -> Path:
    app_paths = get_app_paths()
    return app_paths.outputs / "controlnet" / job_id


def _access_meta_path(job_id: str) -> Path:
    return _job_dir(job_id) / "_access.json"


def _write_access_meta(job_id: str, *, owner: str) -> None:
    if not job_id or not owner:
        return
    path = _access_meta_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"job_id": job_id, "owner": owner, "created_at": time.time()}
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        return


def _read_access_owner(job_id: str) -> str | None:
    path = _access_meta_path(job_id)
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    owner = data.get("owner")
    return str(owner) if owner else None


def _is_admin(request: Request) -> bool:
    return getattr(request.state, "auth_role", "anonymous") == "admin"


def _image_url(request: Request, *, job_id: str, filename: str, owner: str) -> str:
    base = str(request.base_url).rstrip("/")
    url = f"{base}/api/v1/controlnet/images/{job_id}/{filename}"
    if getattr(request.app.state, "auth_enabled", False):
        ttl = int(get_settings().api.t2i_image_token_ttl_seconds or 0)
        token = make_image_token(job_id=job_id, filename=filename, owner=owner, ttl_seconds=ttl)
        if token:
            url = f"{url}?img_token={token}"
    return url


def _safe_resolve(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if root_resolved not in resolved.parents and resolved != root_resolved:
        raise HTTPException(status_code=400, detail="Invalid path")
    return resolved


def _decode_data_url(data_url: str) -> Image.Image:
    if not data_url:
        raise ValueError("Missing control_image")

    payload = data_url.strip()
    if payload.startswith("data:"):
        try:
            _, payload = payload.split(",", 1)
        except ValueError as exc:
            raise ValueError("Invalid data URL") from exc

    raw = base64.b64decode(payload, validate=False)
    img = Image.open(BytesIO(raw))
    return img.convert("RGB")


def _default_controlnet_model(control_type: str, model_id: str) -> str:
    is_sdxl = "xl" in (model_id or "").lower()
    family = "sdxl" if is_sdxl else "sd15"
    return DEFAULT_CONTROLNET_MODELS[control_type][family]


def _run_generate_controlnet_sync(
    req: ControlNetGenerateRequest,
    job_id: str,
    control_type: str,
    control_image: Image.Image,
) -> Dict[str, Any]:
    settings = get_settings()

    if control_type not in ALLOWED_CONTROL_TYPES:
        raise HTTPException(status_code=404, detail="Unknown control type")

    if req.width % 8 != 0 or req.height % 8 != 0:
        raise HTTPException(status_code=400, detail="Width/height must be multiples of 8")
    if req.batch_size > settings.api.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"batch_size exceeds limit ({settings.api.max_batch_size})",
        )
    if req.steps > settings.api.max_steps:
        raise HTTPException(status_code=400, detail=f"steps exceeds limit ({settings.api.max_steps})")

    seed = req.seed if req.seed >= 0 else secrets.randbelow(2**32)
    model_id = _pick_model_id(req)
    scheduler_name, scheduler_kwargs = _sampler_to_scheduler(req.sampler)

    controlnet_model = req.controlnet_model or _default_controlnet_model(control_type, model_id)
    output_dir = _job_dir(job_id)

    manager = get_pipeline_manager()

    with PIPELINE_LOCK:
        if not manager.pipeline_loaded or manager.current_model != model_id:
            if not manager.load_model(model_id):
                raise HTTPException(status_code=503, detail=f"Failed to load model: {model_id}")

        manager.set_scheduler(scheduler_name, **scheduler_kwargs)

        if req.loras:
            lora_configs = []
            for lora in req.loras:
                if not isinstance(lora, dict) or "name" not in lora:
                    continue
                lora_configs.append(
                    {
                        "name": str(lora["name"]),
                        "scale": float(lora.get("scale", 1.0)),
                        "enabled": bool(lora.get("enabled", True)),
                    }
                )
            manager.load_lora_stack(lora_configs)
        else:
            manager.unload_loras()

        params = GenerationParams(
            prompt=req.prompt,
            negative_prompt=req.negative or None,
            width=req.width,
            height=req.height,
            num_inference_steps=req.steps,
            guidance_scale=req.cfg_scale,
            num_images_per_prompt=req.batch_size,
            seed=seed,
            clip_skip=req.clip_skip,
        )

        result = manager.generate_with_controlnet(
            params,
            control_image=control_image,
            control_type=control_type,
            controlnet_model=controlnet_model,
            controlnet_conditioning_scale=req.weight,
            preprocess=req.preprocess,
            canny_low_threshold=req.canny_low_threshold,
            canny_high_threshold=req.canny_high_threshold,
        )
        saved = manager.save_generation_result(result, output_dir)

    return {
        "seed": seed,
        "elapsed_ms": int(result.generation_time * 1000),
        "metadata": result.metadata,
        "saved_paths": saved,
    }


@router.get("/types")
async def list_controlnet_types() -> Dict[str, Any]:
    return {"types": sorted(ALLOWED_CONTROL_TYPES), "defaults": DEFAULT_CONTROLNET_MODELS}


@router.post("/{control_type}", response_model=ControlNetGenerateResponse)
async def generate_controlnet(
    control_type: str, req: ControlNetGenerateRequest, request: Request
):
    if control_type not in ALLOWED_CONTROL_TYPES:
        raise HTTPException(status_code=404, detail="Unknown control type")

    _enforce_t2i_cost_limit(request, req)

    try:
        control_image = _decode_data_url(req.control_image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = str(uuid.uuid4())
    owner = getattr(request.state, "client_key", "ip:unknown")
    _write_access_meta(job_id, owner=owner)
    payload = await asyncio.to_thread(
        _run_generate_controlnet_sync, req, job_id, control_type, control_image
    )

    image_urls = [
        _image_url(request, job_id=job_id, filename=p.name, owner=owner)
        for p in payload["saved_paths"]
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]

    return ControlNetGenerateResponse(
        status="success",
        job_id=job_id,
        image_path=image_urls,
        seed=payload["seed"],
        elapsed_ms=payload["elapsed_ms"],
        metadata=payload["metadata"] or {},
    )


@router.get("/images/{job_id}/{filename}")
async def get_controlnet_image(
    job_id: str, filename: str, request: Request, img_token: str | None = None
):
    if "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    owner = _read_access_owner(job_id) or ""
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

    root = _job_dir(job_id)
    path = _safe_resolve(root / filename, root)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    media_type = "image/png"
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        media_type = "image/jpeg"
    elif path.suffix.lower() == ".webp":
        media_type = "image/webp"

    return stream_file(path, media_type=media_type, filename=path.name, disposition="inline")
