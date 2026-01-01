#!/bin/bash
# scripts/start_t2i_celery_worker.sh - Celery T2I worker (queue=t2i)

set -euo pipefail

echo "=== CharaForge T2I Lab - T2I Celery Worker (queue=t2i) ==="

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

export API_T2I_DISPATCH_MODE="${API_T2I_DISPATCH_MODE:-celery}"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONUNBUFFERED=1

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

echo "🔍 Checking Celery dependencies..."
python -c "
import sys
required_packages = ['celery', 'redis', 'torch', 'diffusers', 'transformers', 'accelerate']
missing_packages = []
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)
if missing_packages:
    print(f'❌ Missing packages: {missing_packages}')
    print('   Install with: pip install -r requirements.txt')
    sys.exit(1)
print('✅ Celery packages available')
"

WORKER_NAME="${CELERY_T2I_WORKER_NAME:-t2i@%h}"
CONCURRENCY="${CELERY_T2I_CONCURRENCY:-1}"
LOG_LEVEL="${CELERY_T2I_LOG_LEVEL:-info}"
MAX_TASKS_PER_CHILD="${CELERY_T2I_MAX_TASKS_PER_CHILD:-1000}"

echo "🚀 Starting Celery T2I worker..."
echo "   queue=t2i, concurrency=${CONCURRENCY}, loglevel=${LOG_LEVEL}"

exec celery -A workers.celery_app worker \
    --loglevel="$LOG_LEVEL" \
    --queues="t2i" \
    --concurrency="$CONCURRENCY" \
    --hostname="$WORKER_NAME" \
    --max-tasks-per-child="$MAX_TASKS_PER_CHILD" \
    --prefetch-multiplier=1

