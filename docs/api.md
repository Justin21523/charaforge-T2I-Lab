# API Summary

Base prefix: `/api/v1`

The backend exposes a FastAPI contract for local AI generation and training workflows.
Responses include structured errors with `error`, `message`, `details`, and `request_id`.

## Core Endpoints

- `GET /health` / `/healthz`: runtime status, Redis availability, GPU availability, cache paths.
- `GET /readiness`: checks required cache path and Redis readiness.
- `GET /models`: list/search the local model registry.
- `POST /models/scan`: scan the AI model warehouse synchronously.
- `POST /models/scan/submit`: enqueue an async registry scan.
- `GET /datasets/list`: list project datasets.
- `POST /datasets/validate`: validate images before training.
- `POST /upload`: store uploaded files under the output warehouse.

## Generation Endpoints

- `POST /t2i/submit`: enqueue text-to-image generation.
- `GET /t2i/status/{job_id}`: poll job state and result image URLs.
- `POST /t2i/cancel/{job_id}`: request cancellation.
- `GET /t2i/jobs`: list owner-visible jobs.
- `POST /controlnet/{type}`: run guided generation with `pose`, `depth`, `canny`, or `lineart`.
- `POST /batch/submit`: submit many prompt tasks.

These endpoints require a compatible local AI environment and model access.

## Training Endpoints

- `POST /finetune/lora/train`: validate a dataset and submit a Celery LoRA job.
- `GET /finetune/lora/status/{job_id}`: inspect Celery state.
- `POST /finetune/lora/cancel/{job_id}`: revoke a running task.
- `WS /ws/train/{job_id}`: subscribe to Redis PubSub training progress.

## Auth Endpoints

- `GET /auth/me`: inspect current auth role/scopes.
- `POST /auth/token`: exchange API key for JWT access and refresh cookies.
- `POST /auth/refresh`: rotate refresh token.
- `POST /auth/logout`: revoke refresh token.
- `POST /auth/ws_ticket`: issue a short-lived WebSocket ticket.
- `GET|POST /auth/keys`: admin-managed API key operations.

When API keys are configured, protected endpoints enforce role/scope checks. Health endpoints
remain public.
