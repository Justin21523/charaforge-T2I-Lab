# CharaForge T2I Lab

**Mass-scale anime character text-to-image fine-tuning** (SD/SDXL + LoRA/ControlNet/IP-Adapter) with a **shared warehouse (`AI_CACHE_ROOT`)**, FastAPI backend, WebUI & PyQt desktop clients, **batch generation**, and **reproducible training/evaluation**.

> Core goals: **character consistency**, **pose/shot control**, **low-VRAM defaults**, **portable REST API**, and **clean Git workflow**.

---

## Table of Contents

- [CharaForge T2I Lab](#charaforge-t2i-lab)
  - [Table of Contents](#table-of-contents)
  - [Why this project](#why-this-project)
  - [Key features](#key-features)
  - [Repository layout](#repository-layout)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
    - [Create conda env](#create-conda-env)
    - [Install Python deps](#install-python-deps)
    - [(Optional) Node / React](#optional-node--react)
  - [Environment \& shared warehouse](#environment--shared-warehouse)
  - [Quick start](#quick-start)
    - [1) Start Redis (for Celery)](#1-start-redis-for-celery)
    - [2) Run the API](#2-run-the-api)
    - [3) Start a worker](#3-start-a-worker)
    - [4) Try a smoke request](#4-try-a-smoke-request)
  - [HTTP API](#http-api)
    - [`GET /health`](#get-health)
    - [`POST /txt2img`](#post-txt2img)
    - [`POST /controlnet/pose`](#post-controlnetpose)
    - [`GET /lora/list`](#get-loralist)
    - [`POST /lora/load` / `POST /lora/unload`](#post-loraload--post-loraunload)
    - [`POST /batch/submit`](#post-batchsubmit)
    - [`GET /batch/status/{job_id}` / `POST /batch/cancel/{job_id}`](#get-batchstatusjob_id--post-batchcanceljob_id)
    - [`POST /eval/report`](#post-evalreport)
  - [Dataset pipeline](#dataset-pipeline)
  - [LoRA fine-tuning](#lora-fine-tuning)
  - [Evaluation \& model selection](#evaluation--model-selection)
  - [Batch generation](#batch-generation)
  - [UI (Web \& Desktop)](#ui-web--desktop)
  - [Monitoring \& Docker](#monitoring--docker)
  - [Git workflow](#git-workflow)
  - [Roadmap (Stages 0→8)](#roadmap-stages-08)
  - [Security \& licensing](#security--licensing)
  - [Acknowledgements](#acknowledgements)

---

## Why this project

Anime character generation requires **repeatable identity + controlled poses**. This lab turns your image collection into **lightweight LoRA adapters**, adds **ControlNet/IP-Adapter** for conditioning, and ships a **production-lean API** you can call from Web, Desktop, or scripts—while keeping large assets out of the repo via a **shared warehouse**.

---

## Key features

* **Fine-tune at scale**: LoRA for SD1.5/SDXL (rank 8–32), fp16, gradient accumulation, 8-bit optimizers.
* **Control & consistency**: ControlNet (OpenPose/Depth/Canny/LineArt) + IP-Adapter (face/style).
* **Data engineering**: pHash dedupe, WD14/DeepDanbooru tagging, OpenPose/Mediapipe pose extraction.
* **Portable API**: FastAPI routes for `txt2img`, `controlnet`, `lora`, `batch`, `eval`.
* **Batch jobs**: Celery/Redis, retries, status tracking, deterministic outputs.
* **Shared warehouse**: one cache root (`AI_CACHE_ROOT`) for models, datasets, outputs; repo stays slim.
* **Evaluation**: CLIP/Face similarity, tag consistency, pose fidelity; automated reports.
* **UIs**: Gradio (quick), optional React, and PyQt (offline-friendly).
* **DevOps**: JSON logging, health checks, Docker Compose, symlink-friendly paths.

---

## Repository layout

```
charaforge-t2i-lab/
├─ backend/
│  ├─ api/            # FastAPI routes: txt2img, controlnet, lora, batch, eval
│  ├─ core/           # pipeline loader, lora manager, safety, utils
│  ├─ jobs/           # Celery workers and tasks
│  ├─ schemas/        # Pydantic request/response models
│  └─ main.py
├─ train/
│  ├─ configs/        # YAML configs for SD/SDXL LoRA
│  ├─ evaluators/     # metrics & reports
│  └─ lora_train.py
├─ tools/
│  ├─ dedupe/         # pHash
│  ├─ tagging/        # WD14 / DeepDanbooru
│  └─ pose/           # OpenPose / Mediapipe
├─ ui-web/            # Gradio or React prototype
├─ ui-desktop/        # PyQt shell
├─ datasets/          # metadata only (no raw images)
├─ docs/              # planning / architecture / rules / fine-tuning guides
├─ tests/
├─ .env.example
└─ .gitignore
```

---

## Prerequisites

* **OS**: Linux or Windows WSL2 (recommended for GPU)
* **Python**: 3.10+
* **GPU**: NVIDIA with recent CUDA (CPU fallback supported with reduced speed)
* **Redis**: for Celery jobs (Docker image is fine)
* **Optional**: Node 18+ for React app

---

## Installation

### Create conda env

```bash
conda create -n sd-lab python=3.10 -y
conda activate sd-lab
```

### Install Python deps

> Choose the proper PyTorch index URL for your CUDA. Example for CUDA 12.1:

```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers[torch] transformers accelerate peft xformers
pip install fastapi uvicorn[standard] pydantic-settings
pip install pillow opencv-python onnxruntime numpy pandas datasets
pip install celery redis
pip install gradio  # for quick WebUI
# optional metrics & logging
pip install scikit-image tqdm
```

### (Optional) Node / React

```bash
# Node 18+ recommended
node -v
npm -v
```

---

## Environment & shared warehouse

**Never commit large data or weights.** Point everything to a shared warehouse via `.env`:

`.env.example`

```bash
AI_CACHE_ROOT=/mnt/ai_warehouse/cache
API_CORS_ORIGINS=http://localhost:3000,http://localhost:7860
CUDA_VISIBLE_DEVICES=0
```

**Bootstrap (copy into every app entry / first notebook cell):**

```python
# English comments only
import os, pathlib, torch
AI_CACHE_ROOT = os.getenv("AI_CACHE_ROOT", "/mnt/ai_warehouse/cache")
for k, v in {
    "HF_HOME":               f"{AI_CACHE_ROOT}/hf",
    "TRANSFORMERS_CACHE":    f"{AI_CACHE_ROOT}/hf/transformers",
    "HF_DATASETS_CACHE":     f"{AI_CACHE_ROOT}/hf/datasets",
    "HUGGINGFACE_HUB_CACHE": f"{AI_CACHE_ROOT}/hf/hub",
    "TORCH_HOME":            f"{AI_CACHE_ROOT}/torch",
}.items():
    os.environ[k] = v
    pathlib.Path(v).mkdir(parents=True, exist_ok=True)

# Project-specific directories in the warehouse
for p in [
    f"{AI_CACHE_ROOT}/models/sd",
    f"{AI_CACHE_ROOT}/models/sdxl",
    f"{AI_CACHE_ROOT}/models/controlnet",
    f"{AI_CACHE_ROOT}/models/lora",
    f"{AI_CACHE_ROOT}/models/ipadapter",
    f"{AI_CACHE_ROOT}/datasets/raw",
    f"{AI_CACHE_ROOT}/datasets/processed",
    f"{AI_CACHE_ROOT}/datasets/metadata",
    f"{AI_CACHE_ROOT}/outputs/charaforge-t2i-lab",
]:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)
print("[cache]", AI_CACHE_ROOT, "| GPU:", torch.cuda.is_available())
```

---

## Quick start

### 1) Start Redis (for Celery)

```bash
docker run -p 6379:6379 --name redis -d redis:7
```

### 2) Run the API

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
# Health check: http://localhost:8000/api/v1/health
```

### 3) Start a worker

```bash
celery -A backend.jobs.worker worker --loglevel=INFO
```

### 4) Try a smoke request

```bash
curl -X POST http://localhost:8000/api/v1/txt2img \
  -H "Content-Type: application/json" \
  -d '{"prompt":"anime girl, blue hair, hoodie, looking at camera","steps":25,"width":768,"height":768,"seed":1234}'
```

Output will be saved under:

```
$AI_CACHE_ROOT/outputs/charaforge-t2i-lab/YYYY-MM-DD/<job_or_time>/XXXX.png
```

---

## HTTP API

Base prefix: `/api/v1`

### `GET /health`

Returns `{ "ok": true }`.

### `POST /txt2img`

Generate an image from a prompt (SD/SDXL selected by config).

```json
{
  "prompt": "anime, blue hair, hoodie, 3/4 view",
  "negative": "blurry, extra fingers",
  "width": 768, "height": 768, "steps": 25, "seed": 1234,
  "scheduler": "euler_a",
  "lora_ids": ["charA_sdxl_r16_v1"], "lora_scales": [0.8],
  "ip_adapter": {"face_ref": "/abs/path/face.png", "weight": 0.7}
}
```

**Response**

```json
{
  "image_path": "/.../outputs/charaforge-t2i-lab/2025-08-22/0001.png",
  "metadata_path": "/.../0001.json",
  "elapsed_ms": 1450
}
```

### `POST /controlnet/pose`

Conditioned generation with a pose image / JSON.

```json
{
  "prompt": "anime [character] running, masterpiece",
  "pose_image": "/abs/path/pose.png",
  "strength": 0.9, "guidance": 1.2,
  "lora_ids": ["charA_sd15_r16_v1"]
}
```

### `GET /lora/list`

List available LoRA adapters under the warehouse.

### `POST /lora/load` / `POST /lora/unload`

Load/unload LoRA adapters dynamically.

### `POST /batch/submit`

Submit a CSV/JSON job file; returns `job_id`.

### `GET /batch/status/{job_id}` / `POST /batch/cancel/{job_id}`

Track / cancel batch generation.

### `POST /eval/report`

Run a fixed prompt+pose validation to compute similarity & tag metrics.

---

## Dataset pipeline

**Goal**: produce clean metadata to train LoRA.

**Recommended schema (CSV/Parquet)**

```
image_path, character, outfit, pose_tag, style, tags, source, split, notes
```

**Steps**

1. **Dedupe** — `tools/dedupe/phash_dedupe.py`
2. **Tagging** — `tools/tagging/batch_tag.py` (WD14/DeepDanbooru)
3. **Pose** — `tools/pose/extract_pose.py` (OpenPose/Mediapipe)
4. **Merge** — write consolidated metadata to
   `$AI_CACHE_ROOT/datasets/metadata/charA.parquet`

---

## LoRA fine-tuning

**Configs** live in `train/configs/*.yaml`. Example (SDXL):

```yaml
model_id: stabilityai/stable-diffusion-xl-base-1.0
resolution: 1024
rank: 16
learning_rate: 1e-4
train_batch_size: 2
gradient_accumulation_steps: 8
mixed_precision: fp16
use_8bit_adam: true
max_train_steps: 8000
dataset_meta: ${AI_CACHE_ROOT}/datasets/metadata/charA.parquet
output_dir: ${AI_CACHE_ROOT}/models/lora/charA_sdxl_r16_v1
text_encoder_train: false
tag_field: tags
caption_template: "[style] [character] [outfit] [pose_tag], anime style"
save_every_steps: 1000
```

**Launch training**

```bash
accelerate config   # (first time)
accelerate launch train/lora_train.py --config train/configs/charA_sdxl_r16.yaml
```

**Low-VRAM tips**

* fp16, `use_8bit_adam`, gradient accumulation 4–8, attention/vae slicing
* SD1.5 @768px first for smoke; upgrade to SDXL @1024px after baseline is stable

---

## Evaluation & model selection

`train/evaluators/` provides:

* **Character similarity** (CLIP/Face embeddings)
* **Tag consistency** (compare predicted tags vs. expected)
* **Pose fidelity** (angle/keypoint deltas when using pose control)

**Run**

```bash
python train/evaluators/run_eval.py \
  --lora $AI_CACHE_ROOT/models/lora/charA_sdxl_r16_v1 \
  --prompts docs/prompts/charA_eval_prompts.json \
  --pose_dir $AI_CACHE_ROOT/datasets/metadata/charA_pose_samples \
  --out $AI_CACHE_ROOT/outputs/charaforge-t2i-lab/eval/charA_sdxl_r16_v1
```

---

## Batch generation

**CSV job example**

```
prompt,negative,pose_image,lora,seed,width,height,steps
"anime [character] running","blurry","/poses/run.png","charA_sdxl_r16_v1",42,1024,1024,25
```

**Submit**

```bash
curl -X POST http://localhost:8000/api/v1/batch/submit -F file=@jobs/charA_batch.csv
curl http://localhost:8000/api/v1/batch/status/<job_id>
```

---

## UI (Web & Desktop)

* **Gradio** (quickest)

  ```bash
  python ui-web/gradio_app.py  # minimal panel calling /txt2img & /controlnet/pose
  ```
* **React** (optional): Vite app calling the same REST endpoints (model switch, ControlNet upload, batch monitor).
* **PyQt** (optional): single-window client for offline use (can call local API or embed pipelines).

---

## Monitoring & Docker

* **Logging/metrics**: JSON logs, `/health`, simple `/metrics` endpoint; extend with Prometheus/Grafana as needed.
* **Docker Compose** (example):

```yaml
version: "3.9"
services:
  api:
    build: ./backend
    environment:
      - AI_CACHE_ROOT=/warehouse/cache
      - API_CORS_ORIGINS=http://localhost:3000
    volumes:
      - /warehouse/cache:/warehouse/cache
    ports: ["8000:8000"]
    command: uvicorn backend.main:app --host 0.0.0.0 --port 8000
  redis: { image: redis:7, ports: ["6379:6379"] }
  worker:
    build: ./backend
    environment: [ "AI_CACHE_ROOT=/warehouse/cache", "REDIS_URL=redis://redis:6379/0" ]
    volumes: [ "/warehouse/cache:/warehouse/cache" ]
    command: celery -A backend.jobs.worker worker --loglevel=INFO
```

---

## Git workflow

* **Branches**: `main` (stable), `develop` (integration), `feature/*`, `fix/*`, `hotfix/*`
* **Conventional Commits**:

  * `feat(train): add SDXL LoRA trainer (rank=16, fp16)`
  * `feat(api): /controlnet pose route`
  * `fix(worker): retry on CUDA OOM via attention slicing`
  * `docs(usage): dataset schema and prompt recipes`
  * `chore(devops): add docker-compose and health checks`
* **Merge**: feature → `develop` (**--no-ff**), milestone → `main`, then tag `vX.Y.Z`

---

## Roadmap (Stages 0→8)

| Stage | Goal                | Key deliverables                                                          |
| ----- | ------------------- | ------------------------------------------------------------------------- |
| **0** | Environment & smoke | `.env.example`, shared bootstrap, minimal SD1.5 inference                 |
| **1** | ControlNet pose     | `/controlnet/pose` route, pose extractor adapter                          |
| **2** | Data pipeline       | pHash dedupe, WD14/DeepDanbooru tagging, pose extraction, merged metadata |
| **3** | LoRA SD1.5          | baseline LoRA trainer + config; character consistency ≥0.8 (val set)      |
| **4** | Evaluation          | CLIP/Face + tag consistency metrics, auto report                          |
| **5** | Backend & batch     | `/txt2img` `/lora/*` `/batch/*`, Celery/Redis workers                     |
| **6** | UIs                 | Gradio WebUI MVP, PyQt shell                                              |
| **7** | Monitoring & deploy | docker-compose, health checks, simple metrics                             |
| **8** | Release             | model cards, E2E smoke, tag `v0.1.0`                                      |

---

## Security & licensing

* **Data & model usage**: You are responsible for ensuring you have the right to use all images and weights.
* **NSFW/sensitive content**: Optional filters are provided; enable them for demos or external use.
* **Model cards**: Each LoRA adapter should ship with `MODEL_CARD.md` describing sources, purpose, and limitations.
* **Repository policy**: Do **not** commit large binaries or private keys. Use `.env` and the shared warehouse.

---

## Acknowledgements

* Built on top of **Diffusers / Transformers / PEFT / Accelerate** and the broader Stable Diffusion ecosystem.
* Thanks to the authors of WD14/DeepDanbooru, OpenPose/Mediapipe, and the community for open tools and baselines.

---
