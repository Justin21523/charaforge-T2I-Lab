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

3) Start T2I worker (async generation queue):
```bash
bash scripts/start_t2i_worker.sh
```

4) Start worker (training queue):
```bash
bash scripts/start_worker.sh
```

5) Scan models into registry:
```bash
python scripts/scan_models.py --replace
# or: curl -X POST http://localhost:8000/api/v1/models/scan -H 'Content-Type: application/json' -d '{"replace":true}'
```

6) Start React UI:
```bash
cd frontend/react_app
npm ci
npm run dev
```

## API (Base Prefix: `/api/v1`)

- Health: `GET /api/v1/health`
- Auth: `GET /api/v1/auth/me`, `POST /api/v1/auth/token`, `POST /api/v1/auth/refresh`, `POST /api/v1/auth/logout`, `GET|POST /api/v1/auth/keys/*` (admin)
- T2I: `POST /api/v1/t2i/generate`, `POST /api/v1/t2i/submit`, `GET /api/v1/t2i/status/{job_id}`, `POST /api/v1/t2i/cancel/{job_id}`, `GET /api/v1/t2i/jobs`, `DELETE /api/v1/t2i/jobs/{job_id}`
- ControlNet: `POST /api/v1/controlnet/{pose|depth|canny|lineart}`
- LoRA: `GET /api/v1/lora/list`, `POST /api/v1/lora/load`, `POST /api/v1/lora/unload`
- Batch: `POST /api/v1/batch/submit`, `GET /api/v1/batch/status/{job_id}`, `GET /api/v1/batch/download/{job_id}`
- Train: `POST /api/v1/finetune/lora/train`, `GET /api/v1/finetune/lora/status/{job_id}`
- Datasets: `GET /api/v1/datasets/root`, `GET /api/v1/datasets/list`, `POST /api/v1/datasets/validate`
- WebSocket: `ws://<host>/api/v1/ws/train/{job_id}`

### Auth + Rate Limiting (Optional)

- Set `API_ADMIN_KEYS` and/or `API_KEYS` (comma-separated) to require an API key for `/api/v1/*` (health/readiness/liveness stay open).
- `/api/v1/models/scan` requires an `admin` key when auth is enabled.
- For browser WebSockets, pass `?access_token=...` (JWT) or `?api_key=...` (API key); headers are not supported by the WebSocket API.
- Set `API_RATE_LIMIT` for global RPM and `API_SCAN_RATE_LIMIT` for `/api/v1/models/scan` (0 disables).
- Bucket limits: `API_UPLOAD_RATE_LIMIT` and `API_DATASETS_RATE_LIMIT` (requests/minute, 0 disables).
- Cost-based throttles: `API_T2I_COST_RATE_LIMIT` (cost units/minute, 0 disables).
- Output cleanup: `API_T2I_OUTPUT_TTL_SECONDS` (seconds, 0 disables), plus `POST /api/v1/t2i/jobs/cleanup`.
- T2I queue/concurrency: per-owner `API_T2I_MAX_CONCURRENT`/`API_T2I_MAX_QUEUE`, global `API_T2I_MAX_GLOBAL_CONCURRENT`/`API_T2I_MAX_GLOBAL_QUEUE`.
- Optional Celery dispatch: set `API_T2I_DISPATCH_MODE=celery` and run a worker that consumes `t2i` tasks (example below).
- Frontend: set the API key in the header UI (stored in localStorage) instead of baking it into build env vars.
- Optional JWT: exchange an API key once (`POST /api/v1/auth/token`) then use `Authorization: Bearer ...` for subsequent calls; refresh via `POST /api/v1/auth/refresh` (no API key needed).
- JWT TTL: `API_JWT_ACCESS_TTL_SECONDS` and `API_JWT_REFRESH_TTL_SECONDS`.

### Managed API Keys (`/api/v1/auth/*`)

- Managed keys are stored hashed under `$AI_CACHE_ROOT/auth/api_keys.json` and can be created/revoked/rotated by an `admin` key.
- Key format: `cfk_<key_id>.<secret>` (send via `X-API-Key` header by default).

Examples:
```bash
# Who am I?
curl -s http://localhost:8000/api/v1/auth/me -H "X-API-Key: $API_KEY"

# List keys (admin only)
curl -s http://localhost:8000/api/v1/auth/keys -H "X-API-Key: $ADMIN_KEY"

# Create a scoped key (admin only)
curl -s -X POST http://localhost:8000/api/v1/auth/keys \\
  -H "Content-Type: application/json" -H "X-API-Key: $ADMIN_KEY" \\
  -d '{"role":"user","scopes":["t2i:generate"],"label":"frontend"}'
```

### JWT Tokens (`/api/v1/auth/token|refresh|logout`)

```bash
# Exchange API key -> short-lived access token + refresh token
curl -s -X POST http://localhost:8000/api/v1/auth/token -H "X-API-Key: $API_KEY"

# Use the access token (no API key header required)
curl -s http://localhost:8000/api/v1/datasets/root -H "Authorization: Bearer $ACCESS_TOKEN"

# Refresh (rotates refresh_token)
curl -s -X POST http://localhost:8000/api/v1/auth/refresh \\
  -H "Content-Type: application/json" \\
  -d '{"refresh_token":"'$REFRESH_TOKEN'"}'
```

### Error Format + Request IDs

- All 4xx/5xx return `{ "error": "...", "message": "...", "details": { ... }, "request_id": "..." }`.
- Every response includes `X-Request-ID` (send it in the request to propagate your own id).

### Monitoring (Optional)

- JSON request logs: set `LOG_JSON=true`.
- Prometheus: set `PROMETHEUS_ENABLED=true` to expose `GET /api/v1/metrics` (requires `admin` or `metrics:read` scope when auth is enabled).
- Sentry: set `SENTRY_DSN` (requires `sentry-sdk` installed in your environment).

### Celery (Optional)

```bash
# T2I worker (Celery task mode)
celery -A workers.celery_app worker -Q t2i --concurrency 1 --loglevel=info
```

## Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```

## Dev Checks

```bash
pytest -q
ruff check api core workers tests
```
