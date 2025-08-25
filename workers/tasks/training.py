# workers/tasks/training.py - Training tasks
from celery import current_task
from typing import Dict, Any
import traceback
import time

from workers.celery_app import celery_app
from core.train.lora_trainer import LoRATrainer, run_lora_training
from core.train.dreambooth_trainer import DreamBoothTrainer
from core.train.dataset import T2IDataset
from core.train.registry import ModelRegistry
from core.config import get_run_output_dir


@celery_app.task(bind=True, name="train_lora")
def train_lora_task(self, run_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """LoRA training task"""

    def progress_callback(progress_info):
        """Update task progress"""
        current_task.update_state(
            state="PROGRESS",
            meta={
                "current_step": progress_info["step"],
                "total_steps": config.get("max_train_steps", 5000),
                "loss": progress_info["loss"],
                "learning_rate": progress_info["lr"],
                "elapsed_time": progress_info["elapsed"],
            },
        )

    try:
        # Initialize registry
        registry = ModelRegistry()
        registry.update_run_status(run_id, "running")

        # Setup trainer
        base_model = config.get(
            "base_model", "stabilityai/stable-diffusion-xl-base-1.0"
        )
        output_dir = get_run_output_dir(run_id)

        trainer = LoRATrainer(
            base_model=base_model, output_dir=str(output_dir), config=config
        )

        # Load dataset
        dataset_name = config.get("dataset_name")
        if not dataset_name:
            raise ValueError("dataset_name required in config")

        dataset = T2IDataset(
            dataset_name=dataset_name,
            split=config.get("split", "train"),
            resolution=config.get("resolution", 768),
            max_samples=config.get("max_samples"),
        )

        # Start training
        result = trainer.train(dataset, progress_callback=progress_callback)

        # Register completed model
        if result["status"] == "completed":
            model_path = str(output_dir / "checkpoints" / "final")
            registry.register_model(
                model_id=f"lora_{run_id}",
                run_id=run_id,
                model_type="lora",
                model_path=model_path,
                metadata={
                    "base_model": base_model,
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

        return result

    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        print(f"[TrainingTask] {error_msg}")
        print(traceback.format_exc())

        # Update registry with error
        registry = ModelRegistry()
        registry.update_run_status(run_id, "failed", metrics={"error": error_msg})

        return {
            "status": "failed",
            "error": error_msg,
            "traceback": traceback.format_exc(),
        }


@celery_app.task(bind=True, name="train_dreambooth")
def train_dreambooth_task(self, run_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """DreamBooth training task"""

    try:
        registry = ModelRegistry()
        registry.update_run_status(run_id, "running")

        # TODO: Implement DreamBooth training
        trainer = DreamBoothTrainer(
            base_model=config.get("base_model"),
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
