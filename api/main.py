# backend/main.py
"""FastAPI application entry point"""
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import time
import logging
from contextlib import asynccontextmanager

from core.config import get_settings, get_cache_paths
from api.middleware import ErrorHandlerMiddleware, LoggingMiddleware
from api.dependencies import get_current_user, verify_api_key
from api.routers import health, t2i, finetune, batch, export, safety, monitoring

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


settings = get_settings()


# Create FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting CharaForge T2I Lab API")

    # Initialize cache paths
    cache_paths = get_cache_paths()
    logger.info(f"Cache root initialized: {cache_paths.root}")

    # Check GPU availability
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
            logger.info(f"GPU: {gpu_name} ({gpu_memory}GB VRAM)")
        else:
            logger.warning("No GPU detected - running on CPU")
    except ImportError:
        logger.warning("PyTorch not available")

    # Test Redis connection
    try:
        import redis

        redis_client = redis.from_url(settings.redis_url)
        redis_client.ping()
        logger.info("Redis connection successful")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")

    yield

    # Shutdown
    logger.info("Shutting down CharaForge T2I Lab API")


# Create FastAPI app
app = FastAPI(
    title="CharaForge T2I Lab",
    description="Text-to-Image generation and LoRA fine-tuning API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api_cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Custom middleware
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(LoggingMiddleware)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(t2i.router, prefix="/api/v1", tags=["text-to-image"])
app.include_router(finetune.router, prefix="/api/v1", tags=["fine-tuning"])
app.include_router(batch.router, prefix="/api/v1", tags=["batch"])
app.include_router(export.router, prefix="/api/v1", tags=["export"])
app.include_router(safety.router, prefix="/api/v1", tags=["safety"])
app.include_router(monitoring.router, prefix="/api/v1", tags=["monitoring"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CharaForge T2I Lab API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/healthz",
    }
