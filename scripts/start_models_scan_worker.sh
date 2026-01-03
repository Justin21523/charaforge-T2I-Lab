#!/bin/bash
# scripts/start_models_scan_worker.sh - Standalone models scan job worker (Redis-backed)

set -euo pipefail

echo "=== CharaForge T2I Lab - Models Scan Worker ==="

if [ -f .env ]; then
    echo "📋 Loading environment variables from .env"
    set -o allexport
    source .env
    set +o allexport
fi

export PROJECT_SLUG="${PROJECT_SLUG:-charaforge-t2i-lab}"
export AI_MODELS_ROOT="${AI_MODELS_ROOT:-/mnt/c/ai_models}"
export AI_CACHE_ROOT="${AI_CACHE_ROOT:-/mnt/c/ai_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/c/ai_cache}"
export HF_HOME="${HF_HOME:-/mnt/c/ai_cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/mnt/c/ai_cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/mnt/c/ai_cache/torch}"
export AI_DATASETS_ROOT="${AI_DATASETS_ROOT:-/mnt/data/datasets}"
export AI_TRAINING_ROOT="${AI_TRAINING_ROOT:-/mnt/data/training}"

echo "📁 Ensuring required directories..."
mkdir -p "$AI_CACHE_ROOT"/{pip,torch,huggingface}
mkdir -p "$AI_MODELS_ROOT"/{stable-diffusion,controlnet,lora,lora_sdxl,embeddings}
mkdir -p "$AI_DATASETS_ROOT/$PROJECT_SLUG"/{raw,processed}
mkdir -p "$AI_TRAINING_ROOT"/{runs,logs}
mkdir -p "$AI_TRAINING_ROOT/runs/$PROJECT_SLUG"/{outputs}

echo "🔍 Checking Redis connection..."
if command -v redis-cli >/dev/null 2>&1; then
    if redis-cli ping >/dev/null 2>&1; then
        echo "✅ Redis is running"
    else
        echo "❌ Redis is not responding. Please start Redis first."
        exit 1
    fi
else
    echo "⚠️  redis-cli not found; continuing without ping check."
fi

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONUNBUFFERED=1

echo "🚀 Starting models scan worker..."
exec python -m api.model_scan_worker

