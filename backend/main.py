# backend/main.py
"""FastAPI application entry point"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# MUST import cache setup first
from backend.core.cache import setup_shared_cache

setup_shared_cache()

from backend.core.config import settings
from backend.api.health import router as health_router

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

# Include routers
app.include_router(health_router, prefix=settings.API_PREFIX, tags=["system"])


@app.get("/")
async def root():
    return {"message": "Multi-Modal Lab API", "docs": "/docs"}


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )
