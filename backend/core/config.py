# backend/core/config.py
"""Application configuration from environment variables"""
import os
from typing import List


class Settings:
    API_PREFIX: str = os.getenv("API_PREFIX", "/api/v1")
    ALLOWED_ORIGINS: List[str] = os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:7860"
    ).split(",")
    DEVICE: str = os.getenv("DEVICE", "auto")
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "2"))
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "4"))
    AI_CACHE_ROOT: str = os.getenv("AI_CACHE_ROOT", "/mnt/ai_warehouse/cache")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
