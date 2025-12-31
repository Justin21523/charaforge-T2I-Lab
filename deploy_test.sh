# CharaForge T2I Lab 部署與驗收指南 (AI_WAREHOUSE 3.0)
#
# NOTE: This is a checklist. Adjust paths + env vars for your workstation.

# ===== 環境準備 =====

# 1) 設定環境變數 (see ~/Desktop/data_model_structure.md)
export PROJECT_SLUG="charaforge-t2i-lab"
export AI_MODELS_ROOT="/mnt/c/ai_models"
export AI_CACHE_ROOT="/mnt/c/ai_cache"
export XDG_CACHE_HOME="/mnt/c/ai_cache"
export HF_HOME="/mnt/c/ai_cache/huggingface"
export TRANSFORMERS_CACHE="/mnt/c/ai_cache/huggingface"
export TORCH_HOME="/mnt/c/ai_cache/torch"
export AI_DATASETS_ROOT="/mnt/data/datasets"
export AI_TRAINING_ROOT="/mnt/data/training"
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# 2) 建立必要目錄
mkdir -p "$AI_CACHE_ROOT"/{pip,torch,huggingface}
mkdir -p "$AI_MODELS_ROOT"/{stable-diffusion,controlnet,lora,lora_sdxl,embeddings}
mkdir -p "$AI_DATASETS_ROOT/$PROJECT_SLUG"/{raw,processed}
mkdir -p "$AI_TRAINING_ROOT"/{runs,logs}
mkdir -p "$AI_TRAINING_ROOT/runs/$PROJECT_SLUG"/outputs

# 3) 安裝相依套件 (如果需要)
pip install -r requirements.txt

# ===== 啟動服務 =====

# 1) 啟動 Redis
redis-server --daemonize yes

# 2) 啟動 API
cd /path/to/charaforge-T2I-Lab
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 3) 啟動 Celery Worker (另一個終端)
cd /path/to/charaforge-T2I-Lab
celery -A workers.celery_app worker --loglevel=info --queues=training --concurrency=1

# ===== 驗收測試 =====

# 0) 掃描模型（建立 registry.json）
python scripts/scan_models.py --replace

# 1) 健康檢查
curl -X GET "http://localhost:8000/api/v1/health"
# Expected: {"status":"ok|degraded", ...}

curl -X GET "http://localhost:8000/"
# Expected: {"name":"CharaForge T2I Lab","version":"0.2.0", ...}

# 2) 模型列表
curl -X GET "http://localhost:8000/api/v1/models"
# Optional filter: ?model_type=sd15|sdxl|lora|controlnet|embedding

# 3) T2I 生成測試（需要已存在的本地模型資料夾）
curl -X POST "http://localhost:8000/api/v1/t2i/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful sunset over mountains",
    "model_type": "sd15",
    "width": 512,
    "height": 512,
    "steps": 10,
    "batch_size": 1,
    "seed": 42
  }'

# 4) LoRA 訓練提交測試（需要 dataset_path 存在於 datasets root 之下）
curl -X POST "http://localhost:8000/api/v1/finetune/lora/train" \
  -H "Content-Type: application/json" \
  -d '{
    "project_name": "test_anime_lora",
    "dataset_path": "test_dataset",
    "instance_prompt": "anime character",
    "model_type": "sd15",
    "lora_rank": 16,
    "num_train_epochs": 1
  }'

# 5) 任務狀態查詢
JOB_ID="上一步返回的job_id"
curl -X GET "http://localhost:8000/api/v1/finetune/lora/status/${JOB_ID}"

# ===== 進階測試 =====

# 測試資料集列表/驗證
curl -X GET "http://localhost:8000/api/v1/datasets/root"
curl -X GET "http://localhost:8000/api/v1/datasets/list"
curl -X POST "http://localhost:8000/api/v1/datasets/validate" \
  -H "Content-Type: application/json" \
  -d '{"dataset_path":"test_dataset"}'

# ===== 驗收標準 =====

# ✅ 必須通過:
# 1) GET /api/v1/health 回傳 200
# 2) GET / 回傳 200
# 3) GET /api/v1/models 回傳 200
# 6. 所有 import 路徑正確，無 ImportError (core 模組可選)
# 7. Redis 連線正常，Celery worker 可啟動

# ⚠️ 可接受的警告:
# - "Core modules not available" (如果相依套件未安裝)
# - "Pipeline not available" (如果模型未下載)
# - "Mock response" (測試階段可接受)

# ❌ 不可接受的錯誤:
# - API 啟動失敗
# - Import path 錯誤
# - Redis 連線失敗 (如果 Celery 需要)
# - JSON 格式錯誤

# ===== 清理 =====

# 停止服務
pkill -f "uvicorn api.main"
pkill -f "celery worker"
pkill -f redis-server

# 清理測試資料
rm -rf "/mnt/data/datasets/${PROJECT_SLUG}/raw/test_dataset"
