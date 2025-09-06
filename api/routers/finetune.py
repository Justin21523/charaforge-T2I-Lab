# api/routers/finetune.py - Fixed Fine-tuning API Router
"""
微調訓練 API 路由 - 修正版
支援 LoRA、DreamBooth 訓練提交、進度查詢、模型管理
"""

import logging
import uuid
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from fastapi import (
    APIRouter,
    HTTPException,
    Depends,
    BackgroundTasks,
    UploadFile,
    File,
    Form,
)
from pydantic import BaseModel, Field

# Import core modules with error handling
try:
    from core.config import get_settings, get_app_paths
    from workers.celery_app import celery_app

    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

logger = logging.getLogger(__name__)
router = APIRouter()

# ===== Pydantic Models =====


class LoRATrainingRequest(BaseModel):
    """LoRA 訓練請求"""

    project_name: str = Field(..., description="專案名稱")
    base_model: str = Field(default="sd15", description="基礎模型 (sd15/sdxl)")
    dataset_path: str = Field(..., description="資料集路徑")
    instance_prompt: str = Field(..., description="實例提示詞")

    # LoRA 參數
    lora_rank: int = Field(default=16, ge=1, le=128, description="LoRA 秩")
    lora_alpha: int = Field(default=32, ge=1, le=256, description="LoRA alpha")
    lora_dropout: float = Field(default=0.1, ge=0.0, le=1.0, description="LoRA dropout")

    # 訓練參數
    num_train_epochs: int = Field(default=15, ge=1, le=100, description="訓練週期")
    train_batch_size: int = Field(default=1, ge=1, le=8, description="批次大小")
    learning_rate: float = Field(default=1e-4, gt=0, le=1e-2, description="學習率")
    gradient_checkpointing: bool = Field(default=True, description="梯度檢查點")

    # 可選參數
    class_prompt: Optional[str] = Field(default=None, description="類別提示詞")
    negative_prompt: Optional[str] = Field(default=None, description="負面提示詞")
    max_train_steps: Optional[int] = Field(default=None, description="最大訓練步數")


class DreamBoothTrainingRequest(BaseModel):
    """DreamBooth 訓練請求"""

    project_name: str = Field(..., description="專案名稱")
    base_model: str = Field(default="sd15", description="基礎模型")
    instance_data_dir: str = Field(..., description="實例資料目錄")
    instance_prompt: str = Field(..., description="實例提示詞")

    # Prior preservation
    with_prior_preservation: bool = Field(default=True, description="啟用先驗保存")
    class_data_dir: Optional[str] = Field(default=None, description="類別資料目錄")
    class_prompt: Optional[str] = Field(default=None, description="類別提示詞")
    num_class_images: int = Field(default=200, description="類別圖片數量")

    # 訓練參數
    num_train_epochs: int = Field(default=20, ge=1, le=100)
    train_batch_size: int = Field(default=1, ge=1, le=4)
    learning_rate: float = Field(default=5e-6, gt=0, le=1e-3)


class TrainingJobResponse(BaseModel):
    """訓練任務回應"""

    job_id: str = Field(..., description="任務 ID")
    project_name: str = Field(..., description="專案名稱")
    training_type: str = Field(..., description="訓練類型")
    status: str = Field(..., description="任務狀態")
    submitted_at: str = Field(..., description="提交時間")
    estimated_duration_minutes: Optional[int] = Field(
        default=None, description="預估時間"
    )

    # Progress tracking
    total_epochs: Optional[int] = Field(default=None)
    total_steps: Optional[int] = Field(default=None)


class JobStatusResponse(BaseModel):
    """任務狀態回應"""

    job_id: str
    status: str  # "PENDING", "PROGRESS", "SUCCESS", "FAILURE", "REVOKED"
    progress: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class TrainingConfig(BaseModel):
    """訓練配置模板"""

    name: str = Field(..., description="配置名稱")
    description: str = Field(..., description="配置描述")
    training_type: str = Field(..., description="訓練類型")
    config: Dict[str, Any] = Field(..., description="配置參數")
    recommended_for: List[str] = Field(default=[], description="推薦用途")


class DatasetValidationResponse(BaseModel):
    """資料集驗證回應"""

    dataset_path: str
    is_valid: bool
    total_images: int
    images_with_captions: int
    missing_captions: int
    warnings: List[str] = Field(default=[])
    suggestions: List[str] = Field(default=[])


# ===== Dependencies =====


async def validate_training_resources():
    """驗證訓練資源"""
    if not CORE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Core modules not available")

    try:
        import torch

        if not torch.cuda.is_available():
            raise HTTPException(
                status_code=503, detail="CUDA not available - GPU required for training"
            )

        # Check available GPU memory
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory_gb < 6.0:  # Minimum 6GB for training
            raise HTTPException(
                status_code=503,
                detail=f"Insufficient GPU memory: {gpu_memory_gb:.1f}GB (minimum 6GB required)",
            )

    except ImportError:
        raise HTTPException(status_code=503, detail="PyTorch not available")


async def validate_dataset_path(dataset_path: str) -> Path:
    """驗證資料集路徑"""
    if not dataset_path:
        raise HTTPException(status_code=400, detail="Dataset path is required")

    path = Path(dataset_path)
    if not path.exists():
        raise HTTPException(
            status_code=400, detail=f"Dataset path does not exist: {dataset_path}"
        )

    if not path.is_dir():
        raise HTTPException(
            status_code=400, detail=f"Dataset path must be a directory: {dataset_path}"
        )

    return path


# ===== API Endpoints =====


@router.post("/lora/train", response_model=TrainingJobResponse)
async def submit_lora_training(
    request: LoRATrainingRequest, _=Depends(validate_training_resources)
):
    """提交 LoRA 訓練任務"""

    try:
        job_id = str(uuid.uuid4())

        logger.info(f"Submitting LoRA training job {job_id}")
        logger.info(f"Project: {request.project_name}")
        logger.info(f"Base model: {request.base_model}")
        logger.info(f"LoRA rank: {request.lora_rank}")

        # Validate dataset exists
        await validate_dataset_path(request.dataset_path)

        # Prepare task config
        task_config = {
            "project_name": request.project_name,
            "base_model": request.base_model,
            "dataset_path": request.dataset_path,
            "instance_prompt": request.instance_prompt,
            "lora_rank": request.lora_rank,
            "lora_alpha": request.lora_alpha,
            "lora_dropout": request.lora_dropout,
            "num_train_epochs": request.num_train_epochs,
            "train_batch_size": request.train_batch_size,
            "learning_rate": request.learning_rate,
            "gradient_checkpointing": request.gradient_checkpointing,
            "class_prompt": request.class_prompt,
            "max_train_steps": request.max_train_steps,
        }

        # Submit to Celery
        if CORE_AVAILABLE:
            task = celery_app.send_task(
                "workers.tasks.training.train_lora",
                args=[task_config],
                task_id=job_id,
                queue="training",
            )

            logger.info(f"✅ LoRA training task submitted: {job_id}")
        else:
            logger.warning("⚠️ Core not available, returning mock response")

        # Estimate duration (rough calculation)
        estimated_duration = (
            request.num_train_epochs * 2
        )  # 2 minutes per epoch (rough estimate)

        response = TrainingJobResponse(
            job_id=job_id,
            project_name=request.project_name,
            training_type="lora",
            status="queued",
            submitted_at=datetime.now().isoformat(),
            estimated_duration_minutes=estimated_duration,
            total_epochs=request.num_train_epochs,
            total_steps=request.max_train_steps
            or (request.num_train_epochs * 100),  # Estimate
        )

        return response

    except Exception as e:
        logger.error(f"LoRA training submission failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Training submission failed: {str(e)}"
        )


@router.post("/dreambooth/train", response_model=TrainingJobResponse)
async def submit_dreambooth_training(
    request: DreamBoothTrainingRequest, _=Depends(validate_training_resources)
):
    """提交 DreamBooth 訓練任務"""

    try:
        job_id = str(uuid.uuid4())

        logger.info(f"Submitting DreamBooth training job {job_id}")
        logger.info(f"Project: {request.project_name}")

        # Validate paths
        await validate_dataset_path(request.instance_data_dir)
        if request.class_data_dir:
            await validate_dataset_path(request.class_data_dir)

        # Prepare task config
        task_config = {
            "project_name": request.project_name,
            "base_model": request.base_model,
            "instance_data_dir": request.instance_data_dir,
            "instance_prompt": request.instance_prompt,
            "with_prior_preservation": request.with_prior_preservation,
            "class_data_dir": request.class_data_dir,
            "class_prompt": request.class_prompt,
            "num_class_images": request.num_class_images,
            "num_train_epochs": request.num_train_epochs,
            "train_batch_size": request.train_batch_size,
            "learning_rate": request.learning_rate,
        }

        # Submit to Celery
        if CORE_AVAILABLE:
            task = celery_app.send_task(
                "workers.tasks.training.train_dreambooth",
                args=[task_config],
                task_id=job_id,
                queue="training",
            )

            logger.info(f"✅ DreamBooth training task submitted: {job_id}")

        estimated_duration = request.num_train_epochs * 3  # 3 minutes per epoch

        response = TrainingJobResponse(
            job_id=job_id,
            project_name=request.project_name,
            training_type="dreambooth",
            status="queued",
            submitted_at=datetime.now().isoformat(),
            estimated_duration_minutes=estimated_duration,
            total_epochs=request.num_train_epochs,
        )

        return response

    except Exception as e:
        logger.error(f"DreamBooth training submission failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Training submission failed: {str(e)}"
        )


@router.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """查詢訓練任務狀態"""

    try:
        if not CORE_AVAILABLE:
            # Return mock status if core not available
            return JobStatusResponse(
                job_id=job_id,
                status="PENDING",
                progress={"message": "Core modules not available - mock response"},
            )

        # Get task result from Celery
        task_result = celery_app.AsyncResult(job_id)

        response_data = {
            "job_id": job_id,
            "status": task_result.status,
        }

        if task_result.status == "PROGRESS":
            response_data["progress"] = task_result.info
        elif task_result.status == "SUCCESS":
            response_data["result"] = task_result.result
            response_data["completed_at"] = datetime.now().isoformat()
        elif task_result.status == "FAILURE":
            response_data["error"] = str(task_result.info)

        return JobStatusResponse(**response_data)

    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get job status: {str(e)}"
        )


@router.delete("/job/{job_id}")
async def cancel_job(job_id: str):
    """取消訓練任務"""

    try:
        if not CORE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Core modules not available")

        # Revoke task
        celery_app.control.revoke(job_id, terminate=True)

        logger.info(f"✅ Training job cancelled: {job_id}")

        return {
            "job_id": job_id,
            "status": "cancelled",
            "cancelled_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to cancel job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")


@router.get("/jobs")
async def list_active_jobs():
    """列出活躍的訓練任務"""

    try:
        if not CORE_AVAILABLE:
            return {"active_jobs": [], "message": "Core modules not available"}

        # Get active tasks from Celery
        inspect = celery_app.control.inspect()

        active_tasks = inspect.active()
        scheduled_tasks = inspect.scheduled()

        all_jobs = []

        # Process active tasks
        if active_tasks:
            for worker, tasks in active_tasks.items():
                for task in tasks:
                    if task["name"].startswith("workers.tasks.training"):
                        all_jobs.append(
                            {
                                "job_id": task["id"],
                                "name": task["name"],
                                "status": "running",
                                "worker": worker,
                                "started_at": task.get("time_start"),
                            }
                        )

        # Process scheduled tasks
        if scheduled_tasks:
            for worker, tasks in scheduled_tasks.items():
                for task in tasks:
                    if task["request"]["name"].startswith("workers.tasks.training"):
                        all_jobs.append(
                            {
                                "job_id": task["request"]["id"],
                                "name": task["request"]["name"],
                                "status": "scheduled",
                                "worker": worker,
                                "eta": task["eta"],
                            }
                        )

        return {"active_jobs": all_jobs, "total_count": len(all_jobs)}

    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")


@router.post("/dataset/validate", response_model=DatasetValidationResponse)
async def validate_dataset(dataset_path: str):
    """驗證訓練資料集"""

    try:
        logger.info(f"Validating dataset: {dataset_path}")

        # Validate path exists
        path = await validate_dataset_path(dataset_path)

        if CORE_AVAILABLE:
            # Submit validation task
            task = celery_app.send_task(
                "workers.tasks.training.validate_dataset",
                args=[dataset_path],
                queue="default",
            )

            # Wait for result (with timeout)
            result = task.get(timeout=60)  # 60 second timeout

            return DatasetValidationResponse(**result)
        else:
            # Simple fallback validation
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
            image_files = [
                f
                for f in path.rglob("*")
                if f.is_file() and f.suffix.lower() in image_extensions
            ]

            # Check for caption files
            caption_files = [f for f in path.rglob("*.txt") if f.is_file()]

            missing_captions = len(image_files) - len(caption_files)

            result = DatasetValidationResponse(
                dataset_path=dataset_path,
                is_valid=len(image_files) > 0,
                total_images=len(image_files),
                images_with_captions=len(caption_files),
                missing_captions=max(0, missing_captions),
                warnings=(
                    ["Some images missing captions"] if missing_captions > 0 else []
                ),
                suggestions=(
                    ["Add captions for better training results"]
                    if missing_captions > 0
                    else []
                ),
            )

            return result

    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Dataset validation failed: {str(e)}"
        )


@router.post("/dataset/upload")
async def upload_training_dataset(
    files: List[UploadFile] = File(...),
    project_name: str = Form(...),
    dataset_type: str = Form(default="images"),
):
    """上傳訓練資料集"""

    try:
        if not CORE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Core modules not available")

        app_paths = get_app_paths()
        upload_dir = app_paths.datasets_raw / project_name
        upload_dir.mkdir(parents=True, exist_ok=True)

        uploaded_files = []
        total_size = 0

        for file in files:
            # Validate file type
            if not file.content_type.startswith("image/"):  # type: ignore
                continue

            # Save file
            file_path = upload_dir / file.filename  # type: ignore
            content = await file.read()

            with open(file_path, "wb") as f:
                f.write(content)

            uploaded_files.append(
                {"filename": file.filename, "size_bytes": len(content)}
            )
            total_size += len(content)

        logger.info(f"Uploaded {len(uploaded_files)} files for project {project_name}")

        return {
            "project_name": project_name,
            "upload_path": str(upload_dir),
            "files_uploaded": len(uploaded_files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "files": uploaded_files,
        }

    except Exception as e:
        logger.error(f"Dataset upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Dataset upload failed: {str(e)}")


@router.get("/configs", response_model=List[TrainingConfig])
async def list_training_configs():
    """列出訓練配置模板"""

    try:
        # TODO: Load actual configs from files
        configs = [
            TrainingConfig(
                name="lora_anime",
                description="LoRA training optimized for anime/illustration style",
                training_type="lora",
                config={
                    "lora_rank": 16,
                    "lora_alpha": 32,
                    "learning_rate": 1e-4,
                    "num_train_epochs": 15,
                    "train_batch_size": 2,
                },
                recommended_for=["anime", "illustration", "character_art"],
            ),
            TrainingConfig(
                name="lora_realistic",
                description="LoRA training for realistic portrait photography",
                training_type="lora",
                config={
                    "lora_rank": 32,
                    "lora_alpha": 64,
                    "learning_rate": 5e-5,
                    "num_train_epochs": 20,
                    "train_batch_size": 1,
                },
                recommended_for=["portrait", "photography", "realistic"],
            ),
            TrainingConfig(
                name="dreambooth_person",
                description="DreamBooth training for person/character",
                training_type="dreambooth",
                config={
                    "num_train_epochs": 25,
                    "train_batch_size": 1,
                    "learning_rate": 5e-6,
                    "with_prior_preservation": True,
                    "num_class_images": 200,
                },
                recommended_for=["person", "character", "face"],
            ),
        ]

        return configs

    except Exception as e:
        logger.error(f"Failed to list configs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list configs: {str(e)}")


@router.get("/presets/{preset_name}")
async def get_training_preset(preset_name: str):
    """取得特定訓練預設配置"""

    try:
        # TODO: Load from actual config files
        presets = {
            "anime_style": {
                "description": "針對動漫風格優化的 LoRA 訓練",
                "config": {
                    "lora_rank": 16,
                    "lora_alpha": 32,
                    "learning_rate": 1e-4,
                    "num_train_epochs": 15,
                    "train_batch_size": 2,
                    "gradient_checkpointing": True,
                },
            },
            "realistic_portrait": {
                "description": "真實人像攝影 LoRA 訓練",
                "config": {
                    "lora_rank": 32,
                    "lora_alpha": 64,
                    "learning_rate": 5e-5,
                    "num_train_epochs": 20,
                    "train_batch_size": 1,
                    "gradient_checkpointing": True,
                },
            },
        }

        if preset_name not in presets:
            raise HTTPException(
                status_code=404, detail=f"Preset not found: {preset_name}"
            )

        return {"preset_name": preset_name, **presets[preset_name]}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get preset: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get preset: {str(e)}")
