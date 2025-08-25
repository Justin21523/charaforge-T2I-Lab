# api/routers/t2i.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path
import json, time, uuid

from core.config import get_cache_paths
from core.t2i.pipeline import PipelineManager
from core.t2i.lora_manager import LoRAManager
from core.t2i.safety import SafetyChecker

router = APIRouter(prefix="/t2i", tags=["text-to-image"])


class T2IRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    negative_prompt: str = ""
    width: int = Field(default=768, ge=64, le=2048)
    height: int = Field(default=768, ge=64, le=2048)
    steps: int = Field(default=25, ge=1, le=150)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=30.0)
    seed: Optional[int] = None
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    lora_ids: List[str] = []
    lora_weights: List[float] = []


class T2IResponse(BaseModel):
    image_path: str
    metadata_path: str
    seed: int
    elapsed_ms: int


@router.post("/generate", response_model=T2IResponse)
async def generate(req: T2IRequest):
    try:
        t0 = time.time()
        pipe_mgr = PipelineManager()
        pipe_mgr.load_pipeline(req.base_model, pipeline_type="sdxl")
        lora_mgr = LoRAManager(pipe_mgr)
        if req.lora_ids:
            weights = req.lora_weights or [1.0] * len(req.lora_ids)
            for lid, w in zip(req.lora_ids, weights):
                lora_mgr.load_lora(lid, w)

        result = pipe_mgr.generate(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
        )

        # Save to outputs
        paths = get_cache_paths()
        out_dir = paths.outputs / "t2i"
        out_dir.mkdir(parents=True, exist_ok=True)
        img_id = uuid.uuid4().hex[:10]
        img_path = out_dir / f"{img_id}.png"
        meta_path = out_dir / f"{img_id}.json"

        # Expect result["images"][0] to be a PIL.Image.Image
        result["images"][0].save(img_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(result["metadata"], f, indent=2)

        return T2IResponse(
            image_path=str(img_path),
            metadata_path=str(meta_path),
            seed=result["seed"],
            elapsed_ms=int((time.time() - t0) * 1000),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
