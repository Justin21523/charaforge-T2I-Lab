"""T2I request/response models shared across endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    negative: str = Field(default="", max_length=2000)

    width: int = Field(default=768, ge=256, le=2048)
    height: int = Field(default=768, ge=256, le=2048)

    steps: int = Field(default=25, ge=1, le=150)
    cfg_scale: float = Field(default=7.5, ge=1.0, le=30.0)
    seed: int = Field(default=-1, ge=-1, le=2**32 - 1)
    sampler: str = Field(default="DPM++ 2M Karras", max_length=100)
    batch_size: int = Field(default=1, ge=1, le=8)

    model_type: str = Field(default="sdxl")  # "sd15" | "sdxl"
    model: Optional[str] = Field(default=None, max_length=2000)

    loras: List[Dict[str, Any]] = Field(default_factory=list)
    clip_skip: Optional[int] = Field(default=None, ge=1, le=12)


class GenerateResponse(BaseModel):
    status: str
    job_id: str
    image_path: List[str]
    seed: int
    elapsed_ms: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SubmitResponse(BaseModel):
    status: str
    job_id: str
    status_url: str
    cancel_url: str


class JobStatusResponse(BaseModel):
    status: str
    job_id: str
    image_path: List[str] = Field(default_factory=list)
    seed: Optional[int] = None
    elapsed_ms: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    progress: Optional[Dict[str, int]] = None
    cancel_requested: bool = False
    error: Optional[Dict[str, Any]] = None

