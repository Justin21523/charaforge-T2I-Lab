# workers/tasks/training.py - Training tasks
from celery import current_task
from typing import Dict, Any
import traceback
import time

from workers.celery_app import celery_app
from core.train.lora_trainer import LoRATrainer
from core.train.dreambooth_trainer import DreamBoothTrainer
from core.train.dataset import T2IDataset
from core.train.registry import ModelRegistry
from core.config import get_run_output_dir


@celery_app.task(bind=True, name="train_lora")
def train_lora_task(self, run_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    LoRA training task. Expects config to include dataset_name, base_model, max_train_steps, etc.
    """

    def on_progress(info: Dict[str, Any]):
        total = config.get("max_train_steps", 1000)
        step = int(info.get("step", 0))
        progress = min(1.0, step / max(1, total))
        self.update_state(
            state="PROGRESS",
            meta={
                "run_id": run_id,
                "current_step": step,
                "total_steps": total,
                "progress": progress,
                "loss": info.get("loss"),
                "eta_minutes": info.get("elapsed", 0),  # can be refined
            },
        )

    try:
        registry = ModelRegistry()
        registry.register_run(run_id, config, status="running")

        output_dir = get_run_output_dir(run_id)
        trainer = LoRATrainer(
            base_model=config.get("base_model"),  # type: ignore
            output_dir=str(output_dir),
            config=config,
        )

        dataset = T2IDataset(
            dataset_name=config["dataset_name"],
            split=config.get("split", "train"),
            resolution=config.get("resolution", 768),
            max_samples=config.get("max_samples"),
        )

        result = trainer.train(dataset, progress_callback=on_progress)

        if result["status"] == "completed":
            model_path = str(output_dir / "checkpoints" / "final")
            registry.register_model(
                model_id=f"lora_{run_id}",
                run_id=run_id,
                model_type="lora",
                model_path=model_path,
                metadata={
                    "base_model": config.get("base_model"),
                    "rank": config.get("rank", 16),
                    "training_steps": result["total_steps"],
                    "final_loss": result["final_loss"],
                },
            )
            registry.update_run_status(
                run_id,
                "completed",
                metrics={"final_loss": result["final_loss"]},
                artifacts={"model_path": model_path},
            )
        else:
            registry.update_run_status(
                run_id, "failed", metrics={"error": result.get("error")}
            )

        return result
    except Exception as e:
        ModelRegistry().update_run_status(run_id, "failed", metrics={"error": str(e)})
        return {"status": "failed", "error": str(e)}


@celery_app.task(bind=True, name="train_dreambooth")
def train_dreambooth_task(self, run_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """DreamBooth training task"""

    try:
        registry = ModelRegistry()
        registry.update_run_status(run_id, "running")

        # TODO: Implement DreamBooth training
        trainer = DreamBoothTrainer(
            base_model=config.get("base_model"),  # type: ignore
            output_dir=str(get_run_output_dir(run_id)),
            config=config,
        )

        # Mock training for now
        time.sleep(2)

        registry.update_run_status(
            run_id, "failed", metrics={"error": "Not implemented"}
        )

        return {"status": "failed", "error": "DreamBooth training not implemented yet"}

    except Exception as e:
        error_msg = f"DreamBooth training failed: {str(e)}"
        registry = ModelRegistry()
        registry.update_run_status(run_id, "failed", metrics={"error": error_msg})

        return {"status": "failed", "error": error_msg}
