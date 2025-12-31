#!/bin/bash
# scripts/start_api.sh - API 啟動腳本

set -euo pipefail

echo "=== CharaForge T2I Lab - API 啟動腳本 ==="

# 載入環境變數（如果存在）
if [ -f .env ]; then
    echo "📋 Loading environment variables from .env"
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

echo "📁 Ensuring required directories..."
mkdir -p "$AI_CACHE_ROOT"/{pip,torch,huggingface}
mkdir -p "$AI_MODELS_ROOT"/{stable-diffusion,controlnet,lora,lora_sdxl,embeddings}
mkdir -p "$AI_DATASETS_ROOT/$PROJECT_SLUG"/{raw,processed}
mkdir -p "$AI_TRAINING_ROOT"/{runs,logs}
mkdir -p "$AI_TRAINING_ROOT/runs/$PROJECT_SLUG"/{outputs}

# 檢查 Python 環境
echo "檢查 Python 環境..."
if ! python -c "import torch, diffusers, transformers, fastapi, peft, celery, redis" 2>/dev/null; then
    echo "錯誤: 缺少必要的 Python 套件"
    echo "請運行: pip install -r requirements.txt"
    exit 1
fi

# 檢查 GPU 可用性
echo "檢查 GPU 狀態..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU 可用: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB')
else:
    print('警告: 未檢測到 GPU，將使用 CPU 模式')
"

# 檢查 Redis 連接
echo "檢查 Redis 連接..."
if command -v redis-cli >/dev/null 2>&1; then
    if redis-cli ping >/dev/null 2>&1; then
        echo "Redis 連接正常"
    else
        echo "警告: Redis 連接失敗，請確保 Redis 正在運行"
        echo "啟動 Redis: redis-server"
    fi
else
    echo "警告: redis-cli 未找到"
fi

# 設置環境變數
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 啟動 API
echo "啟動 FastAPI 服務器..."
echo "API 文檔: http://localhost:8000/docs"
echo "健康檢查: http://localhost:8000/api/v1/health"
echo ""

exec uvicorn api.main:app --host "${API_HOST:-0.0.0.0}" --port "${API_PORT:-8000}" --reload
