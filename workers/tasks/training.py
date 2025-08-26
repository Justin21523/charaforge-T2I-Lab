# workers/tasks/training.py - Training tasks
from celery import current_task
from typing import Dict, Any
import traceback
import time
import json
import logging

from workers.celery_app import celery_app
from core.train.lora_trainer import LoRATrainer
from core.train.dataset import T2IDataset, DatasetRegistry
from core.train.registry import ModelRegistry
from core.config import get_run_output_dir


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="train_lora")
def train_lora_task(self, run_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Complete LoRA training task"""

    def progress_callback(progress_info):
        """Update task progress"""
        try:
            current_task.update_state(
                state="PROGRESS",
                meta={
                    "current_step": progress_info["step"],
                    "total_steps": config.get("max_train_steps", 5000),
                    "loss": progress_info["loss"],
                    "learning_rate": progress_info["lr"],
                    "elapsed_time": progress_info["elapsed"],
                    "epoch": progress_info.get("epoch", 0),
                },
            )
            logger.info(
                f"Training progress: step {progress_info['step']}, loss {progress_info['loss']:.4f}"
            )
        except Exception as e:
            logger.warning(f"Failed to update progress: {e}")

    try:
        logger.info(f"Starting LoRA training task: {run_id}")

        # Initialize registry
        registry = ModelRegistry()
        registry.update_run_status(run_id, "running")

        # Validate configuration
        required_fields = ["base_model", "dataset_name"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")

        # Setup trainer
        base_model = config["base_model"]
        output_dir = get_run_output_dir(run_id)

        logger.info(f"Setting up LoRA trainer:")
        logger.info(f"  - Base model: {base_model}")
        logger.info(f"  - Output dir: {output_dir}")
        logger.info(f"  - Rank: {config.get('rank', 16)}")

        trainer = LoRATrainer(
            base_model=base_model, output_dir=str(output_dir), config=config
        )

        # Load and validate dataset
        dataset_name = config["dataset_name"]
        dataset_registry = DatasetRegistry()
        available_datasets = [d["name"] for d in dataset_registry.list_datasets()]

        if dataset_name not in available_datasets:
            raise ValueError(
                f"Dataset not found: {dataset_name}. Available: {available_datasets}"
            )

        logger.info(f"Loading dataset: {dataset_name}")

        dataset = T2IDataset(
            dataset_name=dataset_name,
            split=config.get("split", "train"),
            resolution=config.get("resolution", 768),
            max_samples=config.get("max_samples"),
            caption_column=config.get("caption_column", "caption"),
            image_column=config.get("image_column", "image_path"),
        )

        logger.info(f"Dataset loaded: {len(dataset)} samples")

        # Start training
        logger.info("Starting training loop")
        result = trainer.train(dataset, progress_callback=progress_callback)

        # Process results
        if result["status"] == "completed":
            logger.info(
                f"Training completed successfully in {result['training_time']:.2f}s"
            )

            # Register completed model
            model_path = str(output_dir / "checkpoints" / "final")
            model_entry = registry.register_model(
                model_id=f"lora_{run_id}",
                run_id=run_id,
                model_type="lora",
                model_path=model_path,
                metadata={
                    "base_model": base_model,
                    "rank": config.get("rank", 16),
                    "training_steps": result["total_steps"],
                    "final_loss": result["final_loss"],
                    "dataset": dataset_name,
                    "resolution": config.get("resolution", 768),
                },
            )

            # Update registry
            registry.update_run_status(
                run_id,
                "completed",
                metrics={
                    "final_loss": result["final_loss"],
                    "total_steps": result["total_steps"],
                    "training_time": result["training_time"],
                },
                artifacts={
                    "model_path": model_path,
                    "samples_dir": str(output_dir / "samples"),
                    "logs_dir": str(output_dir / "logs"),
                },
            )

            logger.info(f"Model registered: {model_entry['model_id']}")

        return result

    except Exception as e:
        error_msg = f"LoRA training failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())

        # Update registry with error
        try:
            registry = ModelRegistry()
            registry.update_run_status(run_id, "failed", metrics={"error": error_msg})
        except Exception as registry_error:
            logger.error(f"Failed to update registry: {registry_error}")

        return {
            "status": "failed",
            "error": error_msg,
            "traceback": traceback.format_exc(),
        }


@celery_app.task(bind=True, name="train_dreambooth")
def train_dreambooth_task(self, run_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """DreamBooth training task (placeholder)"""

    try:
        logger.info(f"Starting DreamBooth training task: {run_id}")

        registry = ModelRegistry()
        registry.update_run_status(run_id, "running")

        # Simulate some work
        for i in range(5):
            time.sleep(1)
            current_task.update_state(
                state="PROGRESS",
                meta={
                    "current_step": i + 1,
                    "total_steps": 5,
                    "message": f"Step {i + 1}/5 - DreamBooth not implemented",
                },
            )

        registry.update_run_status(
            run_id, "failed", metrics={"error": "Not implemented"}
        )

        return {"status": "failed", "error": "DreamBooth training not implemented yet"}

    except Exception as e:
        error_msg = f"DreamBooth training failed: {str(e)}"
        logger.error(error_msg)

        try:
            registry = ModelRegistry()
            registry.update_run_status(run_id, "failed", metrics={"error": error_msg})
        except:
            pass

        return {"status": "failed", "error": error_msg}
