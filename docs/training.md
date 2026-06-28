# LoRA Training Workflow

The training path is designed for local or private GPU environments.

## Flow

1. Prepare a dataset under `${AI_DATASETS_ROOT}/${PROJECT_SLUG}/raw/<dataset_name>`.
2. Validate it with `POST /api/v1/datasets/validate`.
3. Submit a LoRA job with `POST /api/v1/finetune/lora/train`.
4. Celery runs `workers.tasks.training.train_lora`.
5. Progress is stored in Celery state and published to Redis PubSub.
6. The React dashboard subscribes to `WS /api/v1/ws/train/{job_id}`.
7. Final LoRA weights are exported into the model warehouse and registered.

## Requirements

- Redis running.
- Celery worker consuming the `training` queue.
- Compatible PyTorch/Diffusers/PEFT/Accelerate versions.
- Local base model access.
- Writable AI warehouse paths.

## Useful Commands

```bash
bash scripts/start_worker.sh
pytest -q tests/test_security_observability.py
```

The public GitHub Pages demo shows the training monitor as a mock/recorded flow only.
