# api/routers/t2i.py - Updated Text-to-Image API Router
"""
文字轉圖片生成 API 路由 - 完整實作版
支援 SD1.5/SDXL pipeline, LoRA, 實際圖片生成
"""

import asyncio
import logging
import uuid
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator

from core.config import get_settings, get_app_paths, get_model_config
from core.shared_cache import get_shared_cache
from core.exceptions import (
    CharaForgeError,
    T2IError,
    ModelLoadError,
    CUDAOutOfMemoryError,
)
from core.t2i.pipeline import get_pipeline_manager, T2IPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/t2i", tags=["text-to-image"])

# ===== Updated Pydantic Models =====


class GenerationRequest(BaseModel):
    """文字轉圖片生成請求"""

    # Core parameters
    prompt: str = Field(
        ..., min_length=1, max_length=2000, description="Text prompt for generation"
    )
    negative_prompt: Optional[str] = Field(
        default="", max_length=1000, description="Negative prompt"
    )

    # Model selection
    model_type: str = Field(
        default="sd15", regex="^(sd15|sdxl)$", description="Base model type"
    )
    model_name: Optional[str] = Field(
        default=None, description="Specific model name or preset"
    )
    lora_weights: Optional[List[Dict[str, Union[str, float]]]] = Field(
        default=[],
        description="LoRA weights to apply: [{'id': 'lora_name', 'weight': 1.0}]",
    )

    # Generation parameters
    width: int = Field(default=512, ge=256, le=2048, description="Image width")
    height: int = Field(default=512, ge=256, le=2048, description="Image height")
    num_inference_steps: int = Field(
        default=20, ge=5, le=150, description="Number of denoising steps"
    )
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0, description="CFG scale")
    seed: Optional[int] = Field(
        default=None, ge=0, le=2**32 - 1, description="Random seed"
    )

    # Batch generation
    num_images: int = Field(
        default=1, ge=1, le=4, description="Number of images to generate"
    )

    # Advanced options
    scheduler: str = Field(
        default="euler_a",
        regex="^(euler_a|ddim|dpm|lms)$",
        description="Sampling scheduler",
    )
    clip_skip: int = Field(default=1, ge=1, le=12, description="CLIP layers to skip")

    # Quality settings
    enable_vae_slicing: bool = Field(
        default=True, description="Enable VAE slicing for memory efficiency"
    )
    enable_xformers: bool = Field(
        default=True, description="Enable xFormers attention optimization"
    )

    @validator("width", "height")
    def validate_dimensions(cls, v, values):
        """Validate image dimensions are multiples of 8"""
        if v % 8 != 0:
            raise ValueError(f"Dimensions must be multiples of 8, got {v}")
        return v

    @validator("lora_weights")
    def validate_lora_weights(cls, v):
        """Validate LoRA weights list"""
        if len(v) > 5:
            raise ValueError("Maximum 5 LoRA weights allowed")

        for lora in v:
            if not isinstance(lora, dict):
                raise ValueError("LoRA weight must be a dictionary")
            if "id" not in lora:
                raise ValueError("LoRA weight must have 'id' field")
            if "weight" not in lora:
                lora["weight"] = 1.0
            if not isinstance(lora["weight"], (int, float)):
                raise ValueError("LoRA weight must be a number")
            if not (0.0 <= lora["weight"] <= 2.0):
                raise ValueError("LoRA weight must be between 0.0 and 2.0")

        return v


class GenerationResponse(BaseModel):
    """生成結果回應"""

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., regex="^(queued|processing|completed|failed)$")

    # Generation info
    prompt: str
    model_used: str
    generation_time_seconds: Optional[float] = None

    # Results (when completed)
    images: Optional[List[str]] = Field(
        default=[], description="List of generated image URLs"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default={}, description="Generation metadata"
    )

    # Progress info
    progress_percent: Optional[int] = Field(default=0, ge=0, le=100)
    current_step: Optional[int] = Field(default=0)
    total_steps: Optional[int] = Field(default=0)

    # Error info (when failed)
    error_message: Optional[str] = None

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


# ===== Dependencies =====


async def get_pipeline_dependency(model_type: str = "sd15") -> T2IPipeline:
    """Get T2I pipeline dependency"""
    try:
        manager = get_pipeline_manager()
        pipeline = manager.get_pipeline(model_type)

        # Load pipeline if not already loaded
        if pipeline.pipeline is None:
            success = pipeline.load_pipeline()
            if not success:
                raise HTTPException(
                    status_code=503, detail=f"Failed to load {model_type} pipeline"
                )

        return pipeline

    except Exception as e:
        logger.error(f"Pipeline dependency failed: {e}")
        raise HTTPException(status_code=503, detail=f"Pipeline not available: {str(e)}")


async def validate_generation_limits(request: GenerationRequest):
    """Validate generation request against limits"""
    settings = get_settings()

    # Check batch size limit
    if request.num_images > settings.api.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {request.num_images} exceeds limit {settings.api.max_batch_size}",
        )

    # Check steps limit
    if request.num_inference_steps > settings.api.max_steps:
        raise HTTPException(
            status_code=400,
            detail=f"Steps {request.num_inference_steps} exceeds limit {settings.api.max_steps}",
        )

    # Check image size
    total_pixels = request.width * request.height
    if total_pixels > 2048 * 2048:
        raise HTTPException(
            status_code=400, detail="Image size too large, maximum 2048x2048"
        )


# ===== API Endpoints =====


@router.post("/generate", response_model=GenerationResponse)
async def generate_image(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    _=Depends(validate_generation_limits),
):
    """生成文字轉圖片"""

    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())

        logger.info(f"Starting T2I generation job {job_id}")
        logger.info(f"Model: {request.model_type}, Prompt: {request.prompt[:100]}...")

        # Get pipeline manager
        manager = get_pipeline_manager()
        pipeline = manager.get_pipeline(request.model_type)

        # Load pipeline if needed
        if pipeline.pipeline is None:
            logger.info(f"Loading {request.model_type} pipeline...")
            success = pipeline.load_pipeline(request.model_name)
            if not success:
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to load {request.model_type} pipeline",
                )

        # Load LoRA weights if specified
        current_loras = set(pipeline.loaded_loras.keys())
        requested_loras = {lora["id"] for lora in request.lora_weights}

        # Unload LoRAs not in request
        for lora_id in current_loras - requested_loras:
            pipeline.unload_lora(lora_id)

        # Load new LoRAs
        for lora_config in request.lora_weights:
            lora_id = lora_config["id"]
            weight = lora_config.get("weight", 1.0)

            if lora_id not in pipeline.loaded_loras:
                success = pipeline.load_lora(lora_id, weight)
                if not success:
                    logger.warning(f"Failed to load LoRA: {lora_id}")

        # Perform generation
        try:
            result = pipeline.generate(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                num_images=request.num_images,
                seed=request.seed,
                scheduler=request.scheduler,
            )

            # Save generated images
            app_paths = get_app_paths()
            output_dir = app_paths.outputs / "images" / job_id
            saved_paths = result.save_images(output_dir, f"gen_{job_id}")

            # Convert to URLs
            image_urls = [f"/t2i/images/{job_id}/{Path(p).name}" for p in saved_paths]

            logger.info(
                f"✅ Generation completed: {len(image_urls)} images in {result.generation_time:.2f}s"
            )

            response = GenerationResponse(
                job_id=job_id,
                status="completed",
                prompt=request.prompt,
                model_used=f"{request.model_type}/{request.model_name or 'default'}",
                generation_time_seconds=result.generation_time,
                images=image_urls,
                metadata=result.metadata,
                progress_percent=100,
                current_step=request.num_inference_steps,
                total_steps=request.num_inference_steps,
            )

            return response

        except CUDAOutOfMemoryError as e:
            logger.error(f"CUDA OOM during generation: {e}")
            # Try cleanup and suggest lower settings
            pipeline.cleanup()

            raise HTTPException(
                status_code=507,
                detail={
                    "error": "GPU memory insufficient",
                    "suggestions": [
                        "Reduce image size",
                        "Reduce batch size",
                        "Enable CPU offload",
                        "Use lower precision (fp16)",
                    ],
                },
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"T2I generation setup failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Generation setup failed: {str(e)}"
        )


@router.get("/jobs/{job_id}", response_model=GenerationResponse)
async def get_generation_status(job_id: str):
    """查詢生成任務狀態"""

    try:
        # TODO: Query actual job status from database/cache
        # For now, check if images exist
        app_paths = get_app_paths()
        output_dir = app_paths.outputs / "images" / job_id

        if output_dir.exists():
            image_files = list(output_dir.glob("*.png"))
            if image_files:
                image_urls = [f"/t2i/images/{job_id}/{f.name}" for f in image_files]

                return GenerationResponse(
                    job_id=job_id,
                    status="completed",
                    prompt="Retrieved job",
                    model_used="unknown",
                    images=image_urls,
                    progress_percent=100,
                )

        # If no images found, assume job doesn't exist
        raise HTTPException(status_code=404, detail="Job not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status {job_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get job status: {str(e)}"
        )


@router.get("/images/{job_id}/{image_filename}")
async def get_generated_image(job_id: str, image_filename: str):
    """取得生成的圖片檔案"""

    try:
        app_paths = get_app_paths()
        image_path = app_paths.outputs / "images" / job_id / image_filename

        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")

        return FileResponse(
            path=str(image_path), media_type="image/png", filename=image_filename
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve image {job_id}/{image_filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to serve image: {str(e)}")


@router.get("/models", response_model=Dict[str, List[Dict[str, Any]]])
async def list_available_models():
    """列出可用的模型"""

    try:
        cache = get_shared_cache()
        settings = get_settings()

        # Get registered models by type
        models_by_type = {
            "sd15": [],
            "sdxl": [],
            "lora": [],
            "controlnet": [],
            "embedding": [],
        }

        # Add default models
        models_by_type["sd15"].append(
            {
                "id": "default",
                "name": settings.model.default_sd15_model,
                "type": "sd15",
                "size_mb": 0,
                "description": "Default SD1.5 model",
            }
        )

        models_by_type["sdxl"].append(
            {
                "id": "default",
                "name": settings.model.default_sdxl_model,
                "type": "sdxl",
                "size_mb": 0,
                "description": "Default SDXL model",
            }
        )

        # Add registered models
        for model_type in ["sd15", "sdxl", "lora", "controlnet", "embedding"]:
            models = cache.get_registered_models(model_type)
            for model in models:
                models_by_type[model_type].append(
                    {
                        "id": model.model_id,
                        "name": model.model_id,
                        "type": model.model_type,
                        "size_mb": model.size_mb,
                        "description": model.metadata.get("description", ""),
                        "cached_at": model.cached_at,
                    }
                )

        return models_by_type

    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.post("/lora/load")
async def load_lora_weight(
    lora_id: str = Field(..., description="LoRA model ID"),
    weight: float = Field(default=1.0, ge=0.0, le=2.0, description="LoRA weight"),
    model_type: str = Field(default="sd15", regex="^(sd15|sdxl)$"),
):
    """載入 LoRA 權重"""

    try:
        manager = get_pipeline_manager()
        pipeline = manager.get_pipeline(model_type)

        if pipeline.pipeline is None:
            raise HTTPException(
                status_code=400, detail=f"Pipeline {model_type} not loaded"
            )

        success = pipeline.load_lora(lora_id, weight)

        if success:
            return {
                "status": "success",
                "lora_id": lora_id,
                "weight": weight,
                "loaded_loras": pipeline.loaded_loras,
            }
        else:
            raise HTTPException(
                status_code=400, detail=f"Failed to load LoRA: {lora_id}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load LoRA {lora_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load LoRA: {str(e)}")


@router.post("/lora/unload")
async def unload_lora_weight(
    lora_id: str = Field(..., description="LoRA model ID to unload"),
    model_type: str = Field(default="sd15", regex="^(sd15|sdxl)$"),
):
    """卸載 LoRA 權重"""

    try:
        manager = get_pipeline_manager()
        pipeline = manager.get_pipeline(model_type)

        success = pipeline.unload_lora(lora_id)

        if success:
            return {
                "status": "success",
                "unloaded_lora": lora_id,
                "loaded_loras": pipeline.loaded_loras,
            }
        else:
            return {
                "status": "not_found",
                "message": f"LoRA {lora_id} was not loaded",
                "loaded_loras": pipeline.loaded_loras,
            }

    except Exception as e:
        logger.error(f"Failed to unload LoRA {lora_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to unload LoRA: {str(e)}")


@router.post("/preview")
async def generate_preview(
    prompt: str = Field(..., min_length=1, max_length=500),
    model_type: str = Field(default="sd15", regex="^(sd15|sdxl)$"),
    style_preset: Optional[str] = Field(default=None),
):
    """生成快速預覽（低品質、快速）"""

    try:
        # Create fast preview request
        preview_request = GenerationRequest(
            prompt=prompt,
            model_type=model_type,
            width=256,
            height=256,
            num_inference_steps=5,  # Very fast
            guidance_scale=5.0,
            num_images=1,
        )

        # Apply style preset if specified
        if style_preset:
            # TODO: Load and apply style preset
            pass

        # Generate using main endpoint (will be fast due to low settings)
        manager = get_pipeline_manager()
        pipeline = manager.get_pipeline(model_type)

        if pipeline.pipeline is None:
            success = pipeline.load_pipeline()
            if not success:
                raise HTTPException(
                    status_code=503, detail=f"Failed to load {model_type} pipeline"
                )

        result = pipeline.generate(
            prompt=preview_request.prompt,
            width=preview_request.width,
            height=preview_request.height,
            num_inference_steps=preview_request.num_inference_steps,
            guidance_scale=preview_request.guidance_scale,
            num_images=1,
        )

        # Save preview image
        job_id = f"preview_{int(time.time())}"
        app_paths = get_app_paths()
        output_dir = app_paths.outputs / "images" / job_id
        saved_paths = result.save_images(output_dir, "preview")

        image_url = f"/t2i/images/{job_id}/{Path(saved_paths[0]).name}"

        return {
            "job_id": job_id,
            "status": "completed",
            "preview_mode": True,
            "image_url": image_url,
            "generation_time": result.generation_time,
            "estimated_time_seconds": result.generation_time,
        }

    except Exception as e:
        logger.error(f"Preview generation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Preview generation failed: {str(e)}"
        )


@router.delete("/jobs/{job_id}")
async def cancel_generation_job(job_id: str):
    """取消生成任務"""

    try:
        # TODO: Implement actual job cancellation
        # For now, just try to delete output files
        app_paths = get_app_paths()
        output_dir = app_paths.outputs / "images" / job_id

        if output_dir.exists():
            import shutil

            shutil.rmtree(output_dir)

        logger.info(f"Cancelled/cleaned up generation job: {job_id}")

        return {
            "job_id": job_id,
            "status": "cancelled",
            "message": "Job cancelled and files cleaned up",
        }

    except Exception as e:
        logger.error(f"Failed to cancel job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")


@router.get("/system/status")
async def get_t2i_system_status():
    """取得 T2I 系統狀態"""

    try:
        manager = get_pipeline_manager()
        cache = get_shared_cache()
        device_config = cache.get_device_config()

        # Check pipeline status
        pipeline_status = {}
        for model_type in ["sd15", "sdxl"]:
            if model_type in manager.pipelines:
                pipeline = manager.pipelines[model_type]
                pipeline_status[model_type] = pipeline.get_status()
            else:
                pipeline_status[model_type] = {
                    "pipeline_loaded": False,
                    "device": "unknown",
                    "loaded_loras": {},
                }

        # Get memory usage
        memory_usage = {}
        if manager.active_pipeline and manager.active_pipeline in manager.pipelines:
            active_pipeline = manager.pipelines[manager.active_pipeline]
            memory_usage = active_pipeline.get_memory_usage()

        return {
            "status": "healthy",
            "device": device_config,
            "pipelines": pipeline_status,
            "active_pipeline": manager.active_pipeline,
            "memory_usage": memory_usage,
            "queue_length": 0,  # TODO: Get actual queue length from Celery
            "active_jobs": 0,  # TODO: Get actual active jobs
            "models_loaded": len(cache.get_registered_models()),
            "cache_size_gb": cache.get_cache_stats().get("total_size_gb", 0),
        }

    except Exception as e:
        logger.error(f"Failed to get T2I system status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get system status: {str(e)}"
        )


@router.post("/system/cleanup")
async def cleanup_t2i_system():
    """清理 T2I 系統記憶體"""

    try:
        manager = get_pipeline_manager()

        # Cleanup all pipelines
        manager.cleanup_all()

        # Force garbage collection
        import gc

        gc.collect()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        return {
            "status": "success",
            "message": "T2I system cleaned up successfully",
            "pipelines_cleared": len(manager.pipelines),
        }

    except Exception as e:
        logger.error(f"T2I system cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"System cleanup failed: {str(e)}")
