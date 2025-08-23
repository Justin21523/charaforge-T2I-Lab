# backend/main.py
"""FastAPI application entry point"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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
from backend.api import caption, vqa, chat
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

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.getenv("LOG_FILE", "/tmp/multimodal-lab.log")),
        logging.StreamHandler(),
    ],
)

# Create FastAPI app
app = FastAPI(
    title="CharaForge Multi-Modal Lab API",
    version="0.2.0",
    description="Unified API for Multi-modal AI: Caption, VQA, Chat, T2I, and more",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Configuration
allowed_origins = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:7860"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Multi-Modal Lab API", "docs": "/docs"}


# Health check endpoint
@app.get("/api/v1/health")
def health_check():
    """System health check with detailed status"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated": torch.cuda.memory_allocated(0),
            "memory_total": torch.cuda.get_device_properties(0).total_memory,
        }

    # Check model loading status
    model_status = {}
    try:
        from backend.core.caption_pipeline import get_caption_pipeline

        model_status["caption"] = get_caption_pipeline().loaded
    except:
        model_status["caption"] = False

    try:
        from backend.core.vqa_pipeline import get_vqa_pipeline

        model_status["vqa"] = get_vqa_pipeline().loaded
    except:
        model_status["vqa"] = False

    try:
        from backend.core.chat_pipeline import get_chat_pipeline

        model_status["chat"] = get_chat_pipeline().loaded
    except:
        model_status["chat"] = False

    return {
        "status": "healthy",
        "version": "0.3.0",
        "cache_root": AI_CACHE_ROOT,
        "gpu_available": torch.cuda.is_available(),
        "gpu_info": gpu_info,
        "loaded_models": model_status,
        "features": ["caption", "vqa", "chat"],
        "chat_features": {
            "conversation_memory": True,
            "persona_support": True,
            "safety_filtering": True,
            "multi_language": True,
        },
    }


# Model management endpoints
@app.get("/api/v1/models/status")
def models_status():
    """Get status of all loaded models"""
    status = {}

    try:
        from backend.core.caption_pipeline import get_caption_pipeline

        pipeline = get_caption_pipeline()
        status["caption"] = {
            "loaded": pipeline.loaded,
            "model_name": pipeline.model_name,
        }
    except Exception as e:
        status["caption"] = {"loaded": False, "error": str(e)}

    try:
        from backend.core.vqa_pipeline import get_vqa_pipeline

        pipeline = get_vqa_pipeline()
        status["vqa"] = {
            "loaded": pipeline.loaded,
            "model_name": pipeline.model_name,
        }
    except Exception as e:
        status["vqa"] = {"loaded": False, "error": str(e)}

    try:
        from backend.core.chat_pipeline import get_chat_pipeline

        pipeline = get_chat_pipeline()
        status["chat"] = {
            "loaded": pipeline.loaded,
            "model_name": pipeline.model_name,
            "conversation_manager": True,
            "safety_filter": True,
        }
    except Exception as e:
        status["chat"] = {"loaded": False, "error": str(e)}

    return status


@app.post("/api/v1/models/preload")
def preload_models(models: list[str] = None):
    """Preload specified models to reduce first-request latency"""
    if models is None:
        models = ["caption", "vqa", "chat"]

    results = {}

    for model_type in models:
        try:
            if model_type == "caption":
                from backend.core.caption_pipeline import get_caption_pipeline

                pipeline = get_caption_pipeline()
                pipeline.load_model()
                results[model_type] = "loaded"
            elif model_type == "vqa":
                from backend.core.vqa_pipeline import get_vqa_pipeline

                pipeline = get_vqa_pipeline()
                pipeline.load_model()
                results[model_type] = "loaded"
            elif model_type == "chat":
                from backend.core.chat_pipeline import get_chat_pipeline

                pipeline = get_chat_pipeline()
                pipeline.load_model()
                results[model_type] = "loaded"
            else:
                results[model_type] = "unknown_model_type"
        except Exception as e:
            results[model_type] = f"error: {str(e)}"

    return {"preload_results": results}


# System info endpoint
@app.get("/api/v1/system/info")
def system_info():
    """Get detailed system information"""
    import psutil

    return {
        "system": {
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage": psutil.disk_usage("/").percent,
        },
        "cache": {
            "root": AI_CACHE_ROOT,
            "directories": list(APP_DIRS.keys()),
        },
        "environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logging.error(f"Global exception: {exc}", exc_info=True)
    return {"error": "Internal server error", "detail": str(exc)}


API_PREFIX = os.getenv("API_PREFIX", "/api/v1")
# Include routers
app.include_router(health_router, prefix=settings.API_PREFIX, tags=["system"])
app.include_router(caption.router, prefix=API_PREFIX, tags=["Caption"])
app.include_router(vqa.router, prefix=API_PREFIX, tags=["VQA"])
app.include_router(chat.router, prefix=API_PREFIX, tags=["Chat"])


# Serve React build files (production)
if os.path.exists("frontend/react_app/dist"):
    app.mount(
        "/",
        StaticFiles(directory="frontend/react_app/dist", html=True),
        name="frontend",
    )


# Request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    import time

    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time
    logging.info(
        f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s"
    )

    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logging.error(f"Global exception: {exc}", exc_info=True)
    return {"error": "Internal server error", "detail": str(exc)}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )
