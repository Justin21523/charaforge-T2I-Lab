# backend/core/cache.py
"""Shared cache bootstrap - must be imported first in every entry point"""
import os
import pathlib
import torch
from typing import Dict

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))


def setup_shared_cache() -> Dict[str, str]:
    """Initialize shared warehouse cache directories"""
    AI_CACHE_ROOT = os.getenv("AI_CACHE_ROOT", "/mnt/ai_warehouse/cache")

    # Set HuggingFace and PyTorch cache locations
    cache_vars = {
        "HF_HOME": f"{AI_CACHE_ROOT}/hf",
        "TRANSFORMERS_CACHE": f"{AI_CACHE_ROOT}/hf/transformers",
        "HF_DATASETS_CACHE": f"{AI_CACHE_ROOT}/hf/datasets",
        "HUGGINGFACE_HUB_CACHE": f"{AI_CACHE_ROOT}/hf/hub",
        "TORCH_HOME": f"{AI_CACHE_ROOT}/torch",
    }

    for key, value in cache_vars.items():
        os.environ[key] = value
        pathlib.Path(value).mkdir(parents=True, exist_ok=True)

    # Create app-specific directories
    app_dirs = (
        [
            f"{AI_CACHE_ROOT}/models/{name}"
            for name in ["lora", "blip2", "qwen", "llava", "embeddings"]
        ]
        + [
            f"{AI_CACHE_ROOT}/datasets/{name}"
            for name in ["raw", "processed", "metadata"]
        ]
        + [f"{AI_CACHE_ROOT}/outputs/multi-modal-lab"]
    )

    for dir_path in app_dirs:
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)

    print(f"[Cache] {AI_CACHE_ROOT} | GPU: {torch.cuda.is_available()}")
    return {"AI_CACHE_ROOT": AI_CACHE_ROOT, **cache_vars}
