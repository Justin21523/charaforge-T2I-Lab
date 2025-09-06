# SagaForge T2I Lab 部署與驗收指南

# ===== 環境準備 =====

# 1. 設定環境變數
export AI_CACHE_ROOT="../ai_warehouse/cache"
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# 2. 建立必要目錄
mkdir -p ../ai_warehouse/cache/{models,datasets,outputs,cache,runs}
mkdir -p ../ai_warehouse/cache/cache/{hf,torch}
mkdir -p ../ai_warehouse/cache/models/{sd15,sdxl,lora}

# 3. 安裝相依套件 (如果需要)
pip install fastapi uvicorn celery redis torch diffusers transformers peft accelerate pillow

# ===== 啟動服務 =====

# 1. 啟動 Redis (在背景)
redis-server --daemonize yes

# 2. 啟動 API 服務
cd /path/to/saga-forge
python api/main.py
# 或使用 uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 3. 啟動 Celery Worker (另一個終端)
cd /path/to/saga-forge
python workers/celery_app.py worker

# 4. (可選) 啟動 Flower 監控
python workers/celery_app.py flower

# ===== 驗收測試 =====

# 1. 健康檢查
curl -X GET "http://localhost:8000/healthz"
# 預期: {"status": "healthy", "timestamp": "...", ...}

curl -X GET "http://localhost:8000/"
# 預期: {"name": "SagaForge T2I Lab", "version": "0.2.0", ...}

# 2. T2I 生成測試 (Mock 模式)
curl -X POST "http://localhost:8000/t2i/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful sunset over mountains",
    "model_type": "sd15",
    "width": 512,
    "height": 512,
    "num_inference_steps": 20,
    "num_images": 1,
    "seed": 42
  }'
# 預期: {"job_id": "...", "status": "mock_success" 或 "success", ...}

# 3. LoRA 訓練提交測試
curl -X POST "http://localhost:8000/finetune/lora/train" \
  -H "Content-Type: application/json" \
  -d '{
    "project_name": "test_anime_lora",
    "base_model": "sd15",
    "dataset_path": "/tmp/test_dataset",
    "instance_prompt": "anime character",
    "lora_rank": 16,
    "num_train_epochs": 5
  }'
# 預期: {"job_id": "...", "training_type": "lora", "status": "queued", ...}

# 4. 任務狀態查詢
JOB_ID="上一步返回的job_id"
curl -X GET "http://localhost:8000/finetune/job/${JOB_ID}"
# 預期: {"job_id": "...", "status": "PENDING|PROGRESS|SUCCESS|FAILURE", ...}

# 5. 模型列表
curl -X GET "http://localhost:8000/t2i/models"
# 預期: {"available_models": {"sd15": [...], "sdxl": [...]}, ...}

# 6. 配置列表
curl -X GET "http://localhost:8000/finetune/configs"
# 預期: [{"name": "lora_anime", "training_type": "lora", ...}, ...]

# 7. Worker 健康檢查
curl -X POST "http://localhost:8000/finetune/job_health_check" \
  -H "Content-Type: application/json" \
  -d '{}'
# 或直接檢查 Celery
python -c "
from workers.celery_app import celery_app
result = celery_app.send_task('workers.health_check')
print('Worker status:', result.get(timeout=10))
"

# ===== 進階測試 (如果實際模型可用) =====

# 測試資料集驗證
mkdir -p /tmp/test_dataset
echo "anime character, masterpiece" > /tmp/test_dataset/001.txt
curl -X POST "http://localhost:8000/finetune/dataset/validate?dataset_path=/tmp/test_dataset"

# 測試預設配置
curl -X GET "http://localhost:8000/finetune/presets/anime_style"

# 測試 T2I 狀態
curl -X GET "http://localhost:8000/t2i/status"

# ===== 驗收標準 =====

# ✅ 必須通過的檢查:
# 1. GET /healthz 回傳 200 且 status: "healthy"
# 2. GET / 回傳 200 且包含 available_routers
# 3. POST /t2i/generate 回傳 200 (mock 或實際結果)
# 4. POST /finetune/lora/train 回傳 200 且包含 job_id
# 5. GET /finetune/job/{job_id} 回傳 200 且包含狀態
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
rm -rf /tmp/test_dataset