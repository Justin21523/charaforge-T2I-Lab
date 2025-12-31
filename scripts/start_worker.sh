#!/bin/bash
# scripts/start_worker.sh - CharaForge T2I Lab Celery Worker Startup Script

set -euo pipefail

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}👷 Starting CharaForge T2I Lab Celery Worker...${NC}"

# 檢查是否在專案根目錄
if [ ! -f "workers/celery_app.py" ]; then
    echo -e "${RED}❌ Error: workers/celery_app.py not found. Please run this script from the project root directory.${NC}"
    exit 1
fi

# 檢查虛擬環境
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}⚠️  Warning: No virtual environment detected.${NC}"
fi

# 設定環境變數
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONUNBUFFERED=1

# 載入環境變數檔案
if [ -f .env ]; then
    echo -e "${BLUE}📋 Loading environment variables from .env${NC}"
    set -o allexport
    source .env
    set +o allexport
fi

# AI_WAREHOUSE 3.0 defaults (see ~/Desktop/data_model_structure.md)
export PROJECT_SLUG="${PROJECT_SLUG:-charaforge-t2i-lab}"
export AI_MODELS_ROOT="${AI_MODELS_ROOT:-/mnt/c/ai_models}"
export AI_CACHE_ROOT="${AI_CACHE_ROOT:-/mnt/c/ai_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/c/ai_cache}"
export HF_HOME="${HF_HOME:-/mnt/c/ai_cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/mnt/c/ai_cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/mnt/c/ai_cache/torch}"
export AI_DATASETS_ROOT="${AI_DATASETS_ROOT:-/mnt/data/datasets}"
export AI_TRAINING_ROOT="${AI_TRAINING_ROOT:-/mnt/data/training}"

echo -e "${BLUE}📁 Ensuring required directories...${NC}"
mkdir -p "$AI_CACHE_ROOT"/{pip,torch,huggingface}
mkdir -p "$AI_MODELS_ROOT"/{stable-diffusion,controlnet,lora,lora_sdxl,embeddings}
mkdir -p "$AI_DATASETS_ROOT/$PROJECT_SLUG"/{raw,processed}
mkdir -p "$AI_TRAINING_ROOT"/{runs,logs}
mkdir -p "$AI_TRAINING_ROOT/runs/$PROJECT_SLUG"/{outputs}

# 檢查 Redis 連線
echo -e "${BLUE}🔍 Checking Redis connection...${NC}"
if command -v redis-cli >/dev/null 2>&1; then
    if redis-cli ping >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Redis is running${NC}"
    else
        echo -e "${RED}❌ Redis is not responding. Please start Redis first.${NC}"
        exit 1
    fi
fi

# 檢查必要的 Python 套件
echo -e "${BLUE}🔍 Checking Celery dependencies...${NC}"
python -c "
import sys
required_packages = ['celery', 'redis', 'peft', 'torch', 'diffusers', 'transformers', 'accelerate']
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
else:
    print('✅ Celery packages available')
"

if [ $? -ne 0 ]; then
    exit 1
fi

# Worker 設定
WORKER_NAME="${CELERY_WORKER_NAME:-worker@%h}"
CONCURRENCY="${CELERY_CONCURRENCY:-2}"
LOG_LEVEL="${CELERY_LOG_LEVEL:-info}"
MAX_TASKS_PER_CHILD="${CELERY_MAX_TASKS_PER_CHILD:-1000}"

echo -e "${BLUE}⚙️  Worker Configuration:${NC}"
echo -e "   Worker Name: ${WORKER_NAME}"
echo -e "   Concurrency: ${CONCURRENCY}"
echo -e "   Log Level: ${LOG_LEVEL}"
echo -e "   Max Tasks per Child: ${MAX_TASKS_PER_CHILD}"
echo -e "   Cache Root: ${AI_CACHE_ROOT}"
echo -e "   Models Root: ${AI_MODELS_ROOT}"
echo -e "   Training Root: ${AI_TRAINING_ROOT}"

# 檢查 GPU 可用性
echo -e "${BLUE}🖥️  Checking GPU availability for worker...${NC}"
python -c "
try:
    import torch
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f'✅ Worker will use GPU: {gpu_name} ({memory_gb:.1f}GB)')
    else:
        print('ℹ️  Worker will run in CPU mode')
except ImportError:
    print('ℹ️  PyTorch not available - worker will run without GPU support')
"

# 清理函數
cleanup() {
    echo -e "\n${YELLOW}🛑 Shutting down Celery worker...${NC}"
    # Celery 會自動處理清理
    exit 0
}

# 設定清理陷阱
trap cleanup SIGINT SIGTERM

echo -e "${GREEN}🚀 Starting Celery worker...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the worker${NC}"
echo ""

# 啟動 Celery worker
exec celery -A workers.celery_app worker \
    --loglevel="$LOG_LEVEL" \
    --concurrency="$CONCURRENCY" \
    --hostname="$WORKER_NAME" \
    --max-tasks-per-child="$MAX_TASKS_PER_CHILD" \
    --prefetch-multiplier=1
