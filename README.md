# CharaForge T2I Lab

Text-to-Image generation + LoRA fine-tuning lab with a FastAPI backend and a React frontend.

## Storage Layout (AI_WAREHOUSE 3.0)

This repo follows `~/Desktop/data_model_structure.md`:
- Code: `/mnt/c/ai_projects/charaforge-T2I-Lab`
- Models: `/mnt/c/ai_models`
- Caches (HF/Torch/pip): `/mnt/c/ai_cache` (never `~/.cache`)
- Datasets: `/mnt/data/datasets/<project_slug>/...`
- Training runs/outputs: `/mnt/data/training/runs/<project_slug>/...`

Model folders:
```
/mnt/c/ai_models/
  stable-diffusion/sd15/<name>/model_index.json
  stable-diffusion/sdxl/<name>/model_index.json
  controlnet/<name>/
  lora/*.safetensors
```

## Setup (Conda `ai_env`)

```bash
conda env create -f environment.yml
conda activate ai_env
python scripts/check_ai_env.py
python -c "import peft,redis,celery"
```

Create required directories + `.env`:
```bash
python scripts/setup.py
```

## Quick Start (Local)

1) Start Redis:
```bash
docker run -p 6379:6379 --name redis -d redis:7
```

2) Start API:
```bash
bash scripts/start_api.sh
```

3) Start worker (training queue):
```bash
bash scripts/start_worker.sh
```

4) Scan models into registry:
```bash
python scripts/scan_models.py --replace
# or: curl -X POST http://localhost:8000/api/v1/models/scan -H 'Content-Type: application/json' -d '{"replace":true}'
```

5) Start React UI:
```bash
cd frontend/react_app
npm ci
npm run dev
```

## API (Base Prefix: `/api/v1`)

- Health: `GET /api/v1/health`
- T2I: `POST /api/v1/t2i/generate`
- ControlNet: `POST /api/v1/controlnet/{pose|depth|canny|lineart}`
- LoRA: `GET /api/v1/lora/list`, `POST /api/v1/lora/load`, `POST /api/v1/lora/unload`
- Batch: `POST /api/v1/batch/submit`, `GET /api/v1/batch/status/{job_id}`, `GET /api/v1/batch/download/{job_id}`
- Train: `POST /api/v1/finetune/lora/train`, `GET /api/v1/finetune/lora/status/{job_id}`
- Datasets: `GET /api/v1/datasets/root`, `GET /api/v1/datasets/list`, `POST /api/v1/datasets/validate`
- WebSocket: `ws://<host>/api/v1/ws/train/{job_id}`

## Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```

## Dev Checks

```bash
pytest -q
ruff check api core workers tests
```

