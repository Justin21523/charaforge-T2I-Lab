# --- LoRA Training Schemas (append to the end of your file) ---
import uvicorn
import os
import logging
from typing import List, Dict, Optional, Any
import json, time
import torch
from datetime import datetime
from pydantic import Field, BaseModel


class LoRATrainRequest(BaseModel):
    run_id: str
    dataset_path: str
    base_model: ModelType = ModelType.sdxl
    rank: int = Field(16, ge=4, le=256)
    learning_rate: float = Field(1e-4, gt=0)
    train_steps: int = Field(200, ge=1, le=20000)
    batch_size: int = Field(1, ge=1, le=8)
    gradient_accumulation: int = Field(8, ge=1, le=64)
    mixed_precision: str = Field("bf16", pattern="^(fp16|bf16|fp32)$")
    seed: Optional[int] = None
    notes: Optional[str] = None


class TrainResponse(BaseModel):
    job_id: str
    run_id: str
    status: str = "pending"
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
    artifacts: Dict[str, Any] = {}
    error_message: Optional[str] = None
