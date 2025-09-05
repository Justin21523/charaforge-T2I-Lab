#!/bin/bash
# scripts/start_api.sh - API 啟動腳本

set -e

echo "=== CharaForge T2I Lab - API 啟動腳本 ==="

# 檢查環境變數
if [ -z "$AI_CACHE_ROOT" ]; then
    echo "警告: AI_CACHE_ROOT 未設定，使用預設值"
    export AI_CACHE_ROOT="../ai_warehouse/cache"
fi

# 檢查並創建必要目錄
echo "檢查快取目錄..."
mkdir -p "$AI_CACHE_ROOT"/{models,datasets,cache,runs,outputs}
mkdir -p "$AI_CACHE_ROOT"/models/{sd,sdxl,controlnet,lora,ipadapter}
mkdir -p "$AI_CACHE_ROOT"/cache/{hf,torch}

# 檢查 Python 環境
echo "檢查 Python 環境..."
if ! python -c "import torch, diffusers, transformers, fastapi" 2>/dev/null; then
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
echo "健康檢查: http://localhost:8000/healthz"
echo ""

exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

---
#!/bin/bash
# scripts/start_worker.sh - Worker 啟動腳本

set -e

echo "=== CharaForge T2I Lab - Worker 啟動腳本 ==="

# 檢查環境變數
if [ -z "$AI_CACHE_ROOT" ]; then
    echo "警告: AI_CACHE_ROOT 未設定，使用預設值"
    export AI_CACHE_ROOT="../ai_warehouse/cache"
fi

# 檢查 Redis 連接
echo "檢查 Redis 連接..."
if ! redis-cli ping >/dev/null 2>&1; then
    echo "錯誤: Redis 連接失敗"
    echo "請先啟動 Redis: redis-server"
    exit 1
fi

# 設置環境變數
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 選擇 worker 類型
WORKER_TYPE="${1:-all}"

case "$WORKER_TYPE" in
    "training")
        echo "啟動訓練 Worker (單並發)..."
        exec celery -A workers.celery_app worker \
            --loglevel=info \
            --queues=training \
            --concurrency=1 \
            --max-tasks-per-child=1
        ;;
    "generation")
        echo "啟動生成 Worker (多並發)..."
        exec celery -A workers.celery_app worker \
            --loglevel=info \
            --queues=generation \
            --concurrency=2 \
            --max-tasks-per-child=5
        ;;
    "batch")
        echo "啟動批次 Worker..."
        exec celery -A workers.celery_app worker \
            --loglevel=info \
            --queues=batch \
            --concurrency=1 \
            --max-tasks-per-child=3
        ;;
    "all"|*)
        echo "啟動通用 Worker (所有隊列)..."
        exec celery -A workers.celery_app worker \
            --loglevel=info \
            --concurrency=2 \
            --max-tasks-per-child=3
        ;;
esac