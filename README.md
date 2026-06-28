# CharaForge T2I Lab

Portfolio-ready AI image generation and LoRA training lab.

This repository demonstrates how I structure a production-style AI workflow around
text-to-image generation, model registry scanning, async job queues, dataset validation,
LoRA fine-tuning, API security, and a React operator dashboard.

## Live Demo

The public demo is a static GitHub Pages walkthrough:

- Demo site: `https://justin21523.github.io/charaforge-T2I-Lab/`
- Source: [`portfolio-web/`](portfolio-web/)

GitHub Pages cannot run GPU inference, Redis, Celery, or private model weights. The demo
therefore uses mock/recorded states for stable portfolio review while the repo keeps the
real FastAPI backend and worker code for local execution.

## What To Look For

- FastAPI API surface under `/api/v1` with structured errors, request IDs, auth, rate limits, and CORS.
- Async T2I job handling with queued/running/succeeded/failed/canceled states.
- Redis/Celery training queue with LoRA progress reporting and WebSocket support.
- Model registry scanning for SD1.5, SDXL, ControlNet, LoRA, and embeddings.
- Dataset upload/validation safeguards for fine-tuning workflows.
- React dashboard for generation, ControlNet, batch processing, LoRA management, training monitor, gallery, and jobs.
- Static portfolio packaging that makes the project understandable without GPU hardware.

## Current Status

This is a portfolio project, not a hosted public inference service.

Working locally:

- Backend health, auth, dataset, model scan, upload, and job-management code paths.
- React app builds with Vite.
- Static portfolio demo works on GitHub Pages.
- Fast tests cover API health, model scanning, datasets, auth/security, ownership, WebSocket tickets, and observability logic.

Requires local infrastructure:

- Real T2I generation requires compatible PyTorch/Diffusers/PEFT packages, model files or Hugging Face access, and enough CPU/GPU memory.
- Training requires Redis and a Celery worker.
- The default warehouse paths may need to be overridden on machines that cannot write to `/mnt/data`.

## Architecture

```text
React Dashboard / Static Portfolio Demo
        |
        v
FastAPI /api/v1
  - auth, rate limiting, request IDs
  - model registry and dataset validation
  - generation, ControlNet, batch, LoRA, training endpoints
        |
        v
Redis / Celery / worker loops
        |
        v
Diffusers + PEFT + PyTorch
        |
        v
AI_WAREHOUSE storage
  - models
  - datasets
  - cache
  - training runs and generated outputs
```

## Tech Stack

- Frontend: React, Vite, react-router, lucide-react, axios, react-hot-toast.
- Backend: FastAPI, Pydantic settings, httpx tests, structured exception handling.
- AI: PyTorch, Diffusers, Transformers, Accelerate, PEFT, Safetensors.
- Jobs: Redis, Celery, in-process fallback queues for local development.
- Tooling: pytest, pytest-asyncio, ruff, ESLint, Vitest.
- Deployment: GitHub Pages for static demo; Docker Compose for local services.

## Local Setup

Create an environment:

```bash
conda env create -f environment.yml
conda activate ai_env
python scripts/check_ai_env.py
```

If your user cannot write to `/mnt/data`, create a local `.env` override:

```bash
cp .env.example .env
cat >> .env <<'EOF'
AI_DATASETS_ROOT=.local_ai/datasets
AI_TRAINING_ROOT=.local_ai/training
AI_CACHE_ROOT=.local_ai/cache
AI_MODELS_ROOT=.local_ai/models
XDG_CACHE_HOME=.local_ai/cache
HF_HOME=.local_ai/cache/huggingface
TRANSFORMERS_CACHE=.local_ai/cache/huggingface
TORCH_HOME=.local_ai/cache/torch
EOF
```

Then initialize folders:

```bash
python scripts/setup.py
```

## Run The Full Stack Locally

Start Redis:

```bash
docker run -p 6379:6379 --name charaforge-redis -d redis:7
```

Start the API:

```bash
bash scripts/start_api.sh
```

Start workers as needed:

```bash
bash scripts/start_t2i_worker.sh
bash scripts/start_models_scan_worker.sh
bash scripts/start_worker.sh
```

Start the React dashboard:

```bash
npm --prefix frontend/react_app ci
npm --prefix frontend/react_app run dev
```

Useful URLs:

- API docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/api/v1/health`
- React app: `http://localhost:5173`

## Validation

```bash
ruff check api core workers tests
pytest -q
npm --prefix frontend/react_app ci
npm --prefix frontend/react_app run lint
npm --prefix frontend/react_app run test -- --run
npm --prefix frontend/react_app run build
```

## API Overview

Main routes:

- `GET /api/v1/health`
- `GET /api/v1/models`
- `POST /api/v1/models/scan`
- `POST /api/v1/t2i/submit`
- `GET /api/v1/t2i/status/{job_id}`
- `POST /api/v1/controlnet/{pose|depth|canny|lineart}`
- `POST /api/v1/batch/submit`
- `POST /api/v1/finetune/lora/train`
- `GET /api/v1/datasets/list`
- `POST /api/v1/auth/token`
- `POST /api/v1/auth/ws_ticket`
- `WS /api/v1/ws/train/{job_id}`

See [`docs/api.md`](docs/api.md) for a reviewer-focused API summary.

## Repository Layout

```text
api/                 FastAPI app, routers, auth, job managers
core/                config, model registry, T2I pipeline, training helpers
workers/             Celery app and task implementations
frontend/react_app/  React dashboard
portfolio-web/       GitHub Pages static demo
configs/             model, app, train, celery config
scripts/             setup, startup, smoke, model scan scripts
tests/               pytest coverage for backend behavior
docker/              Dockerfiles and compose configs
docs/                API, deployment, training, and changelog notes
```

## Deployment

Static portfolio demo:

```bash
# Deployed by GitHub Actions from portfolio-web/
```

Local/container full stack:

```bash
docker compose -f docker/docker-compose.yml up --build
```

GitHub Pages is the recommended public demo target for this project because it is reliable,
free, and avoids exposing GPU infrastructure or model files. For a live backend demo, deploy
the API separately to a GPU-capable host or a private workstation behind a reverse proxy.

## Notes For Interviewers

This project is strongest as an engineering portfolio case study. The important parts are the
system boundaries, job state modeling, auth/ownership checks, AI warehouse path discipline,
training progress flow, and the conversion of a local AI tool into a clear portfolio demo.
