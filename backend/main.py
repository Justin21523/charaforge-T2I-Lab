# backend/main.py
"""FastAPI application entry point"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import torch
import logging
import pathlib
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# MUST import cache setup first
from backend.core.cache import setup_shared_cache
from backend.api import caption
from backend.core.config import settings
from backend.api.health import router as health_router
from backend.utils.logging import setup_logging

setup_shared_cache()

# Shared Cache Bootstrap (必須在所有 import 之前)
AI_CACHE_ROOT = os.getenv("AI_CACHE_ROOT", "/mnt/ai_warehouse/cache")
for k, v in {
    "HF_HOME": f"{AI_CACHE_ROOT}/hf",
    "TRANSFORMERS_CACHE": f"{AI_CACHE_ROOT}/hf/transformers",
    "HF_DATASETS_CACHE": f"{AI_CACHE_ROOT}/hf/datasets",
    "HUGGINGFACE_HUB_CACHE": f"{AI_CACHE_ROOT}/hf/hub",
    "TORCH_HOME": f"{AI_CACHE_ROOT}/torch",
}.items():
    os.environ[k] = v
    pathlib.Path(v).mkdir(parents=True, exist_ok=True)

# App-specific directories
APP_DIRS = {
    "MODELS_BLIP2": f"{AI_CACHE_ROOT}/models/blip2",
    "MODELS_LLAVA": f"{AI_CACHE_ROOT}/models/llava",
    "MODELS_QWEN": f"{AI_CACHE_ROOT}/models/qwen",
    "MODELS_EMBEDDINGS": f"{AI_CACHE_ROOT}/models/embeddings",
    "MODELS_LORA": f"{AI_CACHE_ROOT}/models/lora",
    "OUTPUT_DIR": f"{AI_CACHE_ROOT}/outputs/multi-modal-lab",
}
for p in APP_DIRS.values():
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

print("[Cache]", AI_CACHE_ROOT, "| GPU:", torch.cuda.is_available())

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Multi-Modal Lab API",
    description="Personal AI toolkit with vision, chat, RAG, and text adventure",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Multi-Modal Lab API", "docs": "/docs"}


# Health check
@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "cache_root": AI_CACHE_ROOT,
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


# Include routers
app.include_router(health_router, prefix=settings.API_PREFIX, tags=["system"])
# Include routers
API_PREFIX = os.getenv("API_PREFIX", "/api/v1")
app.include_router(caption.router, prefix=API_PREFIX, tags=["Caption"])


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )
