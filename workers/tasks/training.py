# workers/tasks/training.py - Training Tasks for Celery
"""
LoRA 與 DreamBooth 訓練的 Celery 任務
支援進度追蹤、中斷恢復、錯誤處理
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import torch
from celery import current_task

from workers.celery_app import celery_app, TaskProgress
from core.config import get_app_paths, get_training_config
from core.shared_cache import get_shared_cache
from core.train.lora_trainer import LoRATrainer
from core.train.dreambooth_trainer import DreamBoothTrainer
from core.train.dataset import T2IDataset
from core.train.evaluators import TrainingEvaluator

logger = logging.getLogger(__name__)

# ===== LoRA Training Task =====


@celery_app.task(bind=True, name="workers.tasks.training.train_lora")
def train_lora(self, config: Dict[str, Any]) -> Dict[str, Any]:
    """LoRA 訓練任務"""

    progress = TaskProgress(self, total_steps=100)  # Will be updated with actual steps

    try:
        logger.info(f"🎯 Starting LoRA training: {config['project_name']}")

        # Step 1: Setup and validation (10%)
        progress.update(5, "Initializing training environment...")

        app_paths = get_app_paths()
        project_dir = app_paths.training_runs / config["project_name"]
        project_dir.mkdir(parents=True, exist_ok=True)

        # Save training config
        config_file = project_dir / "training_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        progress.update(10, "Loading dataset...")

        # Step 2: Load and validate dataset (20%)
        dataset = load_training_dataset(config)
        progress.update(15, f"Dataset loaded: {len(dataset)} samples")

        # Validate dataset
        validation_result = validate_training_dataset(dataset, config)
        if not validation_result["valid"]:
            raise ValueError(
                f"Dataset validation failed: {validation_result['errors']}"
            )

        progress.update(20, "Dataset validation passed")

        # Step 3: Initialize trainer (30%)
        progress.update(25, "Initializing LoRA trainer...")

        trainer_config = prepare_lora_trainer_config(config, project_dir)
        trainer = LoRATrainer(
            base_model=config["base_model"],
            output_dir=str(project_dir),
            config=trainer_config,
        )

        progress.update(30, "Trainer initialized")

        # Update total steps based on actual training configuration
        total_steps = trainer_config.get("max_train_steps") or (
            len(dataset)
            * trainer_config["num_train_epochs"]
            // trainer_config["train_batch_size"]
        )
        progress.total_steps = total_steps

        # Step 4: Training loop (30% - 90%)
        progress.update(35, "Starting training...")

        def training_progress_callback(step: int, logs: Dict[str, Any]):
            """Training progress callback"""
            # Map training step to overall progress (35% to 90%)
            step_progress = 35 + int((step / total_steps) * 55)

            message = f"Training step {step}/{total_steps}"
            if "loss" in logs:
                message += f", loss: {logs['loss']:.4f}"
            if "lr" in logs:
                message += f", lr: {logs['lr']:.2e}"

            progress.update(
                step_progress,
                message,
                current_loss=logs.get("loss"),
                learning_rate=logs.get("lr"),
                gpu_memory_gb=get_gpu_memory_usage(),
            )

        # Start training
        training_result = trainer.train(
            dataset, progress_callback=training_progress_callback
        )

        progress.update(90, "Training completed, saving final model...")

        # Step 5: Final evaluation and cleanup (90% - 100%)
        progress.update(95, "Running final evaluation...")

        # Evaluate trained model
        evaluator = TrainingEvaluator(project_dir)
        eval_result = evaluator.evaluate_lora(
            trainer.model, validation_prompts=config.get("validation_prompts", [])
        )

        # Register trained model
        cache = get_shared_cache()
        model_id = f"lora_{config['project_name']}_{int(time.time())}"

        cache.register_model(
            model_id=model_id,
            model_type="lora",
            local_path=project_dir / "lora_weights",
            metadata={
                "project_name": config["project_name"],
                "base_model": config["base_model"],
                "training_config": trainer_config,
                "training_result": training_result,
                "evaluation": eval_result,
                "trained_at": datetime.now().isoformat(),
            },
        )

        # Prepare final result
        final_result = {
            "model_id": model_id,
            "project_name": config["project_name"],
            "output_path": str(project_dir),
            "training_time_seconds": training_result.get("training_time_seconds"),
            "final_loss": training_result.get("final_loss"),
            "total_steps_completed": training_result.get("total_steps"),
            "evaluation_metrics": eval_result,
            "model_size_mb": get_directory_size_mb(project_dir / "lora_weights"),
        }

        return progress.complete(final_result)

    except Exception as e:
        logger.error(f"❌ LoRA training failed: {e}", exc_info=True)
        return progress.fail(str(e), traceback=str(e))


# ===== DreamBooth Training Task =====


@celery_app.task(bind=True, name="workers.tasks.training.train_dreambooth")
def train_dreambooth(self, config: Dict[str, Any]) -> Dict[str, Any]:
    """DreamBooth 訓練任務"""

    progress = TaskProgress(self, total_steps=100)

    try:
        logger.info(f"🎯 Starting DreamBooth training: {config['project_name']}")

        # Similar structure to LoRA training but with DreamBooth trainer
        progress.update(10, "Initializing DreamBooth training...")

        app_paths = get_app_paths()
        project_dir = app_paths.training_runs / config["project_name"]
        project_dir.mkdir(parents=True, exist_ok=True)

        # Load instance and class datasets
        progress.update(20, "Loading instance and class datasets...")

        instance_dataset = load_dreambooth_dataset(
            config["instance_data_dir"], config["instance_prompt"], is_instance=True
        )

        class_dataset = None
        if config.get("class_data_dir") and config.get("with_prior_preservation", True):
            class_dataset = load_dreambooth_dataset(
                config["class_data_dir"], config["class_prompt"], is_instance=False
            )

        progress.update(30, "Initializing DreamBooth trainer...")

        trainer_config = prepare_dreambooth_trainer_config(config, project_dir)
        trainer = DreamBoothTrainer(
            base_model=config["base_model"],
            output_dir=str(project_dir),
            config=trainer_config,
        )

        # Training with progress callback
        def training_progress_callback(step: int, logs: Dict[str, Any]):
            step_progress = 35 + int((step / trainer_config["max_train_steps"]) * 55)

            message = f"DreamBooth step {step}/{trainer_config['max_train_steps']}"
            if "loss" in logs:
                message += f", loss: {logs['loss']:.4f}"

            progress.update(
                step_progress,
                message,
                current_loss=logs.get("loss"),
                learning_rate=logs.get("lr"),
                gpu_memory_gb=get_gpu_memory_usage(),
            )

        progress.update(35, "Starting DreamBooth training...")

        training_result = trainer.train(
            instance_dataset=instance_dataset,
            class_dataset=class_dataset,
            progress_callback=training_progress_callback,
        )

        progress.update(90, "Training completed, running evaluation...")

        # Evaluation and model registration
        evaluator = TrainingEvaluator(project_dir)
        eval_result = evaluator.evaluate_dreambooth(trainer.model)

        # Register model
        cache = get_shared_cache()
        model_id = f"dreambooth_{config['project_name']}_{int(time.time())}"

        cache.register_model(
            model_id=model_id,
            model_type="dreambooth",
            local_path=project_dir / "model",
            metadata={
                "project_name": config["project_name"],
                "instance_prompt": config["instance_prompt"],
                "class_prompt": config.get("class_prompt"),
                "training_config": trainer_config,
                "training_result": training_result,
                "evaluation": eval_result,
                "trained_at": datetime.now().isoformat(),
            },
        )

        final_result = {
            "model_id": model_id,
            "project_name": config["project_name"],
            "output_path": str(project_dir),
            "training_time_seconds": training_result.get("training_time_seconds"),
            "evaluation_metrics": eval_result,
        }

        return progress.complete(final_result)

    except Exception as e:
        logger.error(f"❌ DreamBooth training failed: {e}", exc_info=True)
        return progress.fail(str(e))


# ===== Dataset Validation Task =====


@celery_app.task(bind=True, name="workers.tasks.training.validate_dataset")
def validate_dataset_task(
    self, dataset_path: str, dataset_type: str = "folder"
) -> Dict[str, Any]:
    """資料集驗證任務"""

    progress = TaskProgress(self, total_steps=100)

    try:
        progress.update(10, "Starting dataset validation...")

        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")

        validation_result = {
            "valid": True,
            "total_images": 0,
            "valid_images": 0,
            "invalid_images": 0,
            "missing_captions": 0,
            "unsupported_formats": 0,
            "corrupted_files": 0,
            "warnings": [],
            "suggestions": [],
        }

        progress.update(20, "Scanning dataset files...")

        # Get all image files
        supported_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        image_files = []

        if dataset_type == "folder":
            for ext in supported_extensions:
                image_files.extend(dataset_path.glob(f"*{ext}"))
                image_files.extend(dataset_path.glob(f"*{ext.upper()}"))

        validation_result["total_images"] = len(image_files)

        if len(image_files) == 0:
            validation_result["valid"] = False
            validation_result["warnings"].append("No image files found in dataset")
            return progress.complete(validation_result)

        progress.update(40, f"Validating {len(image_files)} images...")

        # Validate each image
        for i, image_file in enumerate(image_files):
            try:
                # Check file size
                if image_file.stat().st_size == 0:
                    validation_result["corrupted_files"] += 1
                    continue

                # Try to load image
                from PIL import Image

                with Image.open(image_file) as img:
                    # Check image format
                    if img.format not in ["JPEG", "PNG", "WEBP", "BMP"]:
                        validation_result["unsupported_formats"] += 1
                        continue

                    # Check image size
                    if img.width < 256 or img.height < 256:
                        validation_result["warnings"].append(
                            f"Small image: {image_file.name} ({img.width}x{img.height})"
                        )

                    validation_result["valid_images"] += 1

                # Check for caption file
                caption_file = image_file.with_suffix(".txt")
                if not caption_file.exists():
                    validation_result["missing_captions"] += 1

                # Update progress
                if i % 10 == 0:
                    step = 40 + int((i / len(image_files)) * 40)
                    progress.update(step, f"Validated {i}/{len(image_files)} images")

            except Exception as e:
                validation_result["corrupted_files"] += 1
                validation_result["warnings"].append(
                    f"Corrupted file: {image_file.name} - {e}"
                )

        validation_result["invalid_images"] = (
            validation_result["corrupted_files"]
            + validation_result["unsupported_formats"]
        )

        progress.update(80, "Generating recommendations...")

        # Generate suggestions
        if validation_result["missing_captions"] > 0:
            validation_result["suggestions"].append(
                f"Add caption files (.txt) for {validation_result['missing_captions']} images"
            )

        if validation_result["invalid_images"] > len(image_files) * 0.1:
            validation_result["suggestions"].append(
                "Consider cleaning dataset - high number of invalid images"
            )

        if validation_result["valid_images"] < 10:
            validation_result["suggestions"].append(
                "Dataset is small - consider adding more images for better training"
            )

        # Final validation
        if validation_result["valid_images"] < 5:
            validation_result["valid"] = False
            validation_result["warnings"].append("Too few valid images for training")

        progress.update(90, "Validation completed")

        final_result = {
            "dataset_path": str(dataset_path),
            "validation_result": validation_result,
        }

        return progress.complete(final_result)

    except Exception as e:
        logger.error(f"❌ Dataset validation failed: {e}")
        return progress.fail(str(e))


# ===== Helper Functions =====


def load_training_dataset(config: Dict[str, Any]) -> T2IDataset:
    """載入訓練資料集"""
    dataset_type = config.get("dataset_type", "folder")

    if dataset_type == "folder":
        dataset = T2IDataset.from_folder(
            folder_path=config["dataset_path"],
            instance_prompt=config["instance_prompt"],
            size=config.get("resolution", 512),
        )
    elif dataset_type == "csv":
        dataset = T2IDataset.from_csv(
            csv_path=config["dataset_path"],
            image_column=config.get("image_column", "image"),
            caption_column=config.get("caption_column", "caption"),
            size=config.get("resolution", 512),
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    return dataset


def load_dreambooth_dataset(
    data_dir: str, prompt: str, is_instance: bool = True
) -> T2IDataset:
    """載入 DreamBooth 資料集"""
    return T2IDataset.from_folder(
        folder_path=data_dir, instance_prompt=prompt, size=512, is_instance=is_instance
    )


def validate_training_dataset(
    dataset: T2IDataset, config: Dict[str, Any]
) -> Dict[str, Any]:
    """驗證訓練資料集"""
    validation = {"valid": True, "errors": [], "warnings": []}

    # Check dataset size
    if len(dataset) < 5:
        validation["valid"] = False
        validation["errors"].append(
            f"Dataset too small: {len(dataset)} samples (minimum 5)"
        )

    # Check instance prompt
    if not config.get("instance_prompt", "").strip():
        validation["valid"] = False
        validation["errors"].append("Instance prompt is required")

    # TODO: Add more validation logic

    return validation


def prepare_lora_trainer_config(
    config: Dict[str, Any], project_dir: Path
) -> Dict[str, Any]:
    """準備 LoRA 訓練器配置"""

    # Get base training config and override with user config
    base_config = get_training_config("lora")

    trainer_config = {
        **base_config,
        "output_dir": str(project_dir),
        "project_name": config["project_name"],
        # LoRA parameters
        "lora_rank": config.get("lora_rank", base_config["lora_rank"]),
        "lora_alpha": config.get("lora_alpha", base_config["lora_alpha"]),
        "lora_dropout": config.get("lora_dropout", base_config["lora_dropout"]),
        "target_modules": config.get("target_modules"),
        # Training parameters
        "learning_rate": config.get("learning_rate", base_config["learning_rate"]),
        "train_batch_size": config.get(
            "train_batch_size", base_config["train_batch_size"]
        ),
        "num_train_epochs": config.get(
            "num_train_epochs", base_config["num_train_epochs"]
        ),
        "max_train_steps": config.get(
            "max_train_steps", base_config.get("max_train_steps")
        ),
        "gradient_accumulation_steps": config.get(
            "gradient_accumulation_steps", base_config["gradient_accumulation_steps"]
        ),
        "gradient_checkpointing": config.get(
            "gradient_checkpointing", base_config["gradient_checkpointing"]
        ),
        "mixed_precision": config.get(
            "mixed_precision", base_config["mixed_precision"]
        ),
        # Validation and checkpointing
        "validation_steps": config.get(
            "validation_steps", base_config["validation_steps"]
        ),
        "save_steps": config.get("save_steps", base_config["save_steps"]),
        "validation_prompt": config.get("validation_prompt"),
        # Optimization flags
        "use_8bit_adam": config.get("use_8bit_adam", True),
        "use_xformers": config.get("use_xformers", True),
        "enable_cpu_offload": config.get("enable_cpu_offload", False),
    }

    return trainer_config


def prepare_dreambooth_trainer_config(
    config: Dict[str, Any], project_dir: Path
) -> Dict[str, Any]:
    """準備 DreamBooth 訓練器配置"""

    base_config = get_training_config("dreambooth")

    trainer_config = {
        **base_config,
        "output_dir": str(project_dir),
        "project_name": config["project_name"],
        # Instance and class configuration
        "instance_data_dir": config["instance_data_dir"],
        "instance_prompt": config["instance_prompt"],
        "class_data_dir": config.get("class_data_dir"),
        "class_prompt": config.get("class_prompt"),
        # Training parameters
        "learning_rate": config.get("learning_rate", base_config["learning_rate"]),
        "train_batch_size": config.get(
            "train_batch_size", base_config["train_batch_size"]
        ),
        "num_train_epochs": config.get(
            "num_train_epochs", base_config["num_train_epochs"]
        ),
        "max_train_steps": config.get(
            "max_train_steps", base_config.get("max_train_steps")
        ),
        # DreamBooth specific
        "prior_loss_weight": config.get("prior_loss_weight", 1.0),
        "num_class_images": config.get("num_class_images", 50),
        "with_prior_preservation": config.get("with_prior_preservation", True),
        # Optimization
        "gradient_checkpointing": config.get("gradient_checkpointing", True),
        "mixed_precision": config.get("mixed_precision", "fp16"),
        "use_8bit_adam": config.get("use_8bit_adam", True),
    }

    return trainer_config


def get_gpu_memory_usage() -> float:
    """取得 GPU 記憶體使用量 (GB)"""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
    except ImportError:
        pass
    return 0.0


def get_directory_size_mb(directory: Path) -> float:
    """取得目錄大小 (MB)"""
    try:
        total_size = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
        return total_size / (1024 * 1024)
    except Exception:
        return 0.0


# ===== Model Export Task =====


@celery_app.task(bind=True, name="workers.tasks.training.export_model")
def export_trained_model(
    self, model_id: str, export_format: str = "safetensors"
) -> Dict[str, Any]:
    """匯出已訓練的模型"""

    progress = TaskProgress(self, total_steps=100)

    try:
        progress.update(10, f"Starting model export: {model_id}")

        cache = get_shared_cache()
        model_info = cache.get_model_info(model_id)

        if not model_info:
            raise ValueError(f"Model not found: {model_id}")

        progress.update(20, "Loading model...")

        app_paths = get_app_paths()
        export_dir = app_paths.exports / f"{model_id}_{int(time.time())}"
        export_dir.mkdir(parents=True, exist_ok=True)

        progress.update(40, "Converting model format...")

        # TODO: Implement actual model conversion
        # For now, just copy the model files
        import shutil

        source_path = model_info.path
        if source_path.is_dir():
            # Copy directory
            target_path = export_dir / source_path.name
            shutil.copytree(source_path, target_path)
        else:
            # Copy file
            target_path = export_dir / source_path.name
            shutil.copy2(source_path, target_path)

        progress.update(70, "Creating export metadata...")

        # Create export metadata
        export_metadata = {
            "model_id": model_id,
            "model_type": model_info.model_type,
            "export_format": export_format,
            "exported_at": datetime.now().isoformat(),
            "original_metadata": model_info.metadata,
            "export_size_mb": get_directory_size_mb(export_dir),
        }

        # Save metadata
        metadata_file = export_dir / "export_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(export_metadata, f, indent=2)

        progress.update(90, "Finalizing export...")

        # Create download archive if requested
        archive_path = None
        if export_format == "zip":
            import zipfile

            archive_path = export_dir.parent / f"{export_dir.name}.zip"

            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for file_path in export_dir.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(export_dir)
                        zipf.write(file_path, arcname)

        final_result = {
            "model_id": model_id,
            "export_path": str(export_dir),
            "archive_path": str(archive_path) if archive_path else None,
            "export_size_mb": export_metadata["export_size_mb"],
            "download_url": f"/finetune/download/{export_dir.name}",
        }

        return progress.complete(final_result)

    except Exception as e:
        logger.error(f"❌ Model export failed: {e}")
        return progress.fail(str(e))


# ===== Cleanup Task =====


@celery_app.task(bind=True, name="workers.tasks.training.cleanup_old_training_runs")
def cleanup_old_training_runs(self, max_age_days: int = 30) -> Dict[str, Any]:
    """清理舊的訓練執行記錄"""

    progress = TaskProgress(self, total_steps=100)

    try:
        progress.update(10, "Starting cleanup of old training runs...")

        app_paths = get_app_paths()
        training_runs_dir = app_paths.training_runs

        if not training_runs_dir.exists():
            return progress.complete({"message": "No training runs directory found"})

        cutoff_time = time.time() - (max_age_days * 24 * 3600)

        cleanup_stats = {
            "directories_removed": 0,
            "files_removed": 0,
            "space_freed_mb": 0.0,
        }

        progress.update(20, "Scanning training directories...")

        training_dirs = [d for d in training_runs_dir.iterdir() if d.is_dir()]

        for i, training_dir in enumerate(training_dirs):
            try:
                # Check directory age
                dir_mtime = training_dir.stat().st_mtime

                if dir_mtime < cutoff_time:
                    # Calculate size before deletion
                    dir_size_mb = get_directory_size_mb(training_dir)

                    # Remove directory
                    shutil.rmtree(training_dir)

                    cleanup_stats["directories_removed"] += 1
                    cleanup_stats["space_freed_mb"] += dir_size_mb

                    logger.info(f"Removed old training run: {training_dir.name}")

                # Update progress
                step = 20 + int((i / len(training_dirs)) * 70)
                progress.update(
                    step, f"Processed {i+1}/{len(training_dirs)} directories"
                )

            except Exception as e:
                logger.warning(f"Failed to process {training_dir}: {e}")

        progress.update(90, "Cleanup completed")

        final_result = {
            "cleanup_stats": cleanup_stats,
            "max_age_days": max_age_days,
            "total_directories_scanned": len(training_dirs),
        }

        return progress.complete(final_result)

    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        return progress.fail(str(e))
