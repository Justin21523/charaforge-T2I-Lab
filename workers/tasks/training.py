"""Celery training tasks.

Currently supported:
- LoRA fine-tuning via `core.train.lora_trainer.LoRATrainer`

The API can enqueue these tasks via:
- POST `/api/v1/finetune/lora/train`
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from core.config import get_app_paths, get_training_config
from core.train.dataset import T2IDataset, validate_image_dataset
from core.train.lora_trainer import LoRATrainer
from core.train.registry import ModelEntry, get_model_registry
from workers.celery_app import TaskProgress, celery_app

logger = logging.getLogger(__name__)


def _resolve_trainer_config(task_cfg: Dict[str, Any]) -> Dict[str, Any]:
    base = get_training_config("lora")

    num_epochs = int(task_cfg.get("num_train_epochs") or task_cfg.get("num_epochs") or base.get("num_train_epochs") or 10)
    checkpointing_steps = int(task_cfg.get("checkpointing_steps") or task_cfg.get("save_steps") or base.get("save_steps") or 500)

    return {
        **base,
        # LoRA
        "rank": int(task_cfg.get("lora_rank") or task_cfg.get("rank") or base.get("lora_rank") or 16),
        "lora_alpha": int(task_cfg.get("lora_alpha") or base.get("lora_alpha") or 32),
        "lora_dropout": float(task_cfg.get("lora_dropout") or base.get("lora_dropout") or 0.1),
        # Training
        "learning_rate": float(task_cfg.get("learning_rate") or base.get("learning_rate") or 1e-4),
        "train_batch_size": int(task_cfg.get("train_batch_size") or base.get("train_batch_size") or 1),
        "num_epochs": num_epochs,
        "max_train_steps": task_cfg.get("max_train_steps") or base.get("max_train_steps"),
        "gradient_accumulation_steps": int(
            task_cfg.get("gradient_accumulation_steps")
            or base.get("gradient_accumulation_steps")
            or 1
        ),
        "gradient_checkpointing": bool(
            task_cfg.get("gradient_checkpointing")
            if task_cfg.get("gradient_checkpointing") is not None
            else base.get("gradient_checkpointing", True)
        ),
        "mixed_precision": str(task_cfg.get("mixed_precision") or base.get("mixed_precision") or "fp16"),
        # Checkpointing/validation
        "checkpointing_steps": checkpointing_steps,
        "validation_steps": int(task_cfg.get("validation_steps") or base.get("validation_steps") or 100),
    }


def _export_final_lora(run_dir: Path, project_name: str) -> Path:
    """Copy the final LoRA weights into `/mnt/c/ai_models/lora/*` for easy loading."""
    app_paths = get_app_paths()
    lora_out_dir = app_paths.lora_weights
    lora_out_dir.mkdir(parents=True, exist_ok=True)

    final_dir = run_dir / "checkpoints" / "final"
    if not final_dir.exists():
        raise FileNotFoundError(f"Final checkpoint not found: {final_dir}")

    candidates = list(final_dir.glob("*.safetensors"))
    if not candidates:
        # PEFT fallback (older versions)
        candidates = list(final_dir.glob("*.bin"))
    if not candidates:
        raise FileNotFoundError(f"No LoRA weights found under: {final_dir}")

    src = candidates[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = lora_out_dir / f"{project_name}_{timestamp}{src.suffix}"
    shutil.copy2(src, dst)
    return dst


@celery_app.task(bind=True, name="workers.tasks.training.train_lora")
def train_lora(self, config: Dict[str, Any]) -> Dict[str, Any]:
    progress = TaskProgress(self, total_steps=100)

    project_name = str(config.get("project_name") or "").strip()
    dataset_path = str(config.get("dataset_path") or "").strip()
    instance_prompt = str(config.get("instance_prompt") or "").strip()
    base_model = str(config.get("base_model") or "").strip()

    if not project_name:
        return progress.fail("Missing config.project_name")
    if not dataset_path:
        return progress.fail("Missing config.dataset_path")
    if not instance_prompt:
        return progress.fail("Missing config.instance_prompt")
    if not base_model:
        return progress.fail("Missing config.base_model")

    run_dir = get_app_paths().training_runs / project_name
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        progress.update(5, "Validating dataset...")
        ok, errors = validate_image_dataset(Path(dataset_path))
        if not ok:
            return progress.fail("Dataset validation failed", errors=errors)

        progress.update(10, "Loading dataset...")
        dataset = T2IDataset.from_folder(
            folder_path=dataset_path,
            instance_prompt=instance_prompt,
            size=int(config.get("resolution") or 768),
            max_samples=config.get("max_samples"),
        )

        progress.update(20, "Preparing trainer config...")
        trainer_config = _resolve_trainer_config(config)
        (run_dir / "training_config.json").write_text(
            json.dumps(trainer_config, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        trainer = LoRATrainer(base_model=base_model, output_dir=str(run_dir), config=trainer_config)

        # Progress callback receives a dict from LoRATrainer.
        def _cb(logs: Dict[str, Any]):
            step = int(logs.get("step") or 0)
            total = int(trainer.config.get("max_train_steps") or 1)
            pct = 35 + int(min(1.0, step / max(total, 1)) * 55)
            msg = f"step {step}/{total}"
            if "loss" in logs:
                msg += f", loss={logs['loss']:.4f}"
            progress.update(pct, msg, step=step, total_steps=total, lr=logs.get("lr"))

        progress.update(30, "Training...")
        training_result = trainer.train(dataset, progress_callback=_cb)

        progress.update(90, "Exporting LoRA weights...")
        exported_path = _export_final_lora(run_dir, project_name)

        registry = get_model_registry()
        entry = ModelEntry(
            name=f"lora/{exported_path.stem}",
            path=str(exported_path),
            model_type="lora",
            size_mb=exported_path.stat().st_size / (1024 * 1024),
            description="Trained LoRA",
            tags=["lora", "trained"],
            created_at=datetime.now().isoformat(),
            metadata={
                "project_name": project_name,
                "base_model": base_model,
                "dataset_path": dataset_path,
                "instance_prompt": instance_prompt,
            },
        )
        registry.register_model(entry)

        progress.update(100, "Complete")
        return progress.complete(
            {
                "status": "completed",
                "project_name": project_name,
                "run_dir": str(run_dir),
                "model_id": entry.name,
                "model_path": str(exported_path),
                "training_result": training_result,
            }
        )

    except Exception as exc:
        logger.exception("LoRA training failed")
        return progress.fail(str(exc))
