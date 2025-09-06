# api/routers/batch.py - Batch Processing Router
"""
批次處理 API 路由
支援批次圖片生成、批次訓練、批次處理任務
"""

import logging
import uuid
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator

from core.config import get_settings, get_app_paths, get_run_output_dir
from core.shared_cache import get_shared_cache
from core.exceptions import CharaForgeError, BatchProcessingError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/batch", tags=["batch-processing"])

# ===== Pydantic Models =====


class BatchGenerationRequest(BaseModel):
    """批次生成請求"""

    batch_name: str = Field(..., min_length=1, max_length=100)
    prompts: List[str] = Field(..., min_items=1, max_items=100)  # type: ignore

    # Common generation parameters
    model_type: str = Field(default="sd15", pattern="^(sd15|sdxl)$")
    width: int = Field(default=512, ge=256, le=2048)
    height: int = Field(default=512, ge=256, le=2048)
    num_inference_steps: int = Field(default=20, ge=5, le=150)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)

    # Batch-specific parameters
    seeds: Optional[List[int]] = Field(
        default=None, description="Seeds for each prompt (optional)"
    )
    lora_weights: Optional[List[Dict[str, Union[str, float]]]] = Field(default=[])

    # Output options
    save_individually: bool = Field(default=True)
    create_zip: bool = Field(default=False)

    @validator("seeds")
    def validate_seeds(cls, v, values):
        if v is not None:
            prompts = values.get("prompts", [])
            if len(v) != len(prompts):
                raise ValueError("Number of seeds must match number of prompts")
        return v


class BatchJobResponse(BaseModel):
    """批次任務回應"""

    batch_id: str = Field(..., description="Unique batch identifier")
    batch_name: str
    status: str = Field(..., pattern="^(queued|processing|completed|failed|cancelled)$")

    # Progress information
    total_items: int
    completed_items: int = 0
    failed_items: int = 0
    progress_percent: int = Field(default=0, ge=0, le=100)

    # Results
    output_directory: Optional[str] = None
    generated_files: List[str] = Field(default=[])
    zip_file: Optional[str] = None

    # Timing
    estimated_time_remaining: Optional[int] = None

    # Error handling
    errors: List[str] = Field(default=[])

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class CaptionRequest(BaseModel):
    """批次圖片說明生成請求"""

    batch_name: str = Field(..., min_length=1, max_length=100)
    image_paths: List[str] = Field(..., min_items=1, max_items=1000)  # type: ignore

    # Caption parameters
    model_name: str = Field(default="blip2", pattern="^(blip2|llava|qwen_vl)$")
    prompt_template: str = Field(default="Describe this image in detail:")
    max_length: int = Field(default=200, ge=10, le=1000)

    # Output options
    save_to_txt: bool = Field(default=True)
    save_to_json: bool = Field(default=False)


class VQARequest(BaseModel):
    """批次視覺問答請求"""

    batch_name: str = Field(..., min_length=1, max_length=100)
    image_paths: List[str] = Field(..., min_items=1, max_items=1000)  # type: ignore
    questions: List[str] = Field(..., min_items=1, max_items=10)  # type: ignore

    # VQA parameters
    model_name: str = Field(default="llava", pattern="^(llava|qwen_vl|minigpt4)$")

    # Output options
    save_results: bool = Field(default=True)


# ===== Background Task Functions =====


async def process_generation_batch(
    batch_request: BatchGenerationRequest, batch_id: str
) -> Dict[str, Any]:
    """處理批次生成任務"""
    try:
        logger.info(f"Starting batch generation: {batch_id}")

        app_paths = get_app_paths()
        output_dir = app_paths.outputs / "batch" / batch_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get T2I pipeline
        from core.t2i.pipeline import get_pipeline_manager

        manager = get_pipeline_manager()
        pipeline = manager.get_pipeline(batch_request.model_type)

        # Load pipeline if needed
        if pipeline.pipeline is None:
            success = pipeline.load_pipeline()
            if not success:
                raise BatchProcessingError(
                    f"Failed to load {batch_request.model_type} pipeline"
                )

        # Load LoRA weights if specified
        for lora_config in batch_request.lora_weights:  # type: ignore
            lora_id = lora_config["id"]
            weight = lora_config.get("weight", 1.0)
            pipeline.load_lora(lora_id, weight)  # type: ignore

        results = {
            "batch_id": batch_id,
            "total_items": len(batch_request.prompts),
            "completed_items": 0,
            "failed_items": 0,
            "generated_files": [],
            "errors": [],
        }

        # Process each prompt
        for i, prompt in enumerate(batch_request.prompts):
            try:
                # Determine seed
                seed = None
                if batch_request.seeds and i < len(batch_request.seeds):
                    seed = batch_request.seeds[i]

                # Generate image
                generation_result = pipeline.generate(
                    prompt=prompt,
                    width=batch_request.width,
                    height=batch_request.height,
                    num_inference_steps=batch_request.num_inference_steps,
                    guidance_scale=batch_request.guidance_scale,
                    seed=seed,
                    num_images=1,
                )

                # Save image
                if batch_request.save_individually:
                    filename = f"{batch_id}_{i:04d}_{generation_result.seed_used}.png"
                    image_path = output_dir / filename
                    generation_result.images[0].save(image_path)
                    results["generated_files"].append(str(image_path))

                results["completed_items"] += 1
                logger.info(
                    f"Batch {batch_id}: {i+1}/{len(batch_request.prompts)} completed"
                )

            except Exception as e:
                error_msg = f"Item {i} failed: {str(e)}"
                results["errors"].append(error_msg)
                results["failed_items"] += 1
                logger.error(f"Batch {batch_id}: {error_msg}")

        # Create ZIP if requested
        if batch_request.create_zip and results["generated_files"]:
            zip_path = output_dir / f"{batch_id}_results.zip"
            import zipfile

            with zipfile.ZipFile(zip_path, "w") as zipf:
                for file_path in results["generated_files"]:
                    zipf.write(file_path, Path(file_path).name)

            results["zip_file"] = str(zip_path)

        # Save batch metadata
        metadata = {
            "batch_request": batch_request.dict(),
            "results": results,
            "completed_at": datetime.now().isoformat(),
        }

        metadata_path = output_dir / "batch_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Batch generation completed: {batch_id}")
        return results

    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        raise BatchProcessingError(f"Batch generation failed: {str(e)}")


async def process_caption_batch(
    caption_request: CaptionRequest, batch_id: str
) -> Dict[str, Any]:
    """處理批次圖片說明生成 - 實作缺失函數"""
    try:
        logger.info(f"Starting batch captioning: {batch_id}")

        app_paths = get_app_paths()
        output_dir = app_paths.outputs / "batch" / batch_id
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "batch_id": batch_id,
            "total_items": len(caption_request.image_paths),
            "completed_items": 0,
            "failed_items": 0,
            "captions": {},
            "errors": [],
        }

        # TODO: Implement actual VLM captioning
        # For now, return placeholder results
        for i, image_path in enumerate(caption_request.image_paths):
            try:
                # Placeholder caption generation
                caption = f"Generated caption for {Path(image_path).name}: A detailed description of the image content."

                results["captions"][image_path] = caption
                results["completed_items"] += 1

                # Save to txt file if requested
                if caption_request.save_to_txt:
                    txt_path = output_dir / f"{Path(image_path).stem}_caption.txt"
                    txt_path.write_text(caption)

            except Exception as e:
                error_msg = f"Caption failed for {image_path}: {str(e)}"
                results["errors"].append(error_msg)
                results["failed_items"] += 1

        # Save JSON results if requested
        if caption_request.save_to_json:
            json_path = output_dir / "captions.json"
            with open(json_path, "w") as f:
                json.dump(results["captions"], f, indent=2)

        return results

    except Exception as e:
        logger.error(f"Batch captioning failed: {e}")
        raise BatchProcessingError(f"Batch captioning failed: {str(e)}")


async def process_vqa_batch(vqa_request: VQARequest, batch_id: str) -> Dict[str, Any]:
    """處理批次視覺問答 - 實作缺失函數"""
    try:
        logger.info(f"Starting batch VQA: {batch_id}")

        app_paths = get_app_paths()
        output_dir = app_paths.outputs / "batch" / batch_id
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "batch_id": batch_id,
            "total_items": len(vqa_request.image_paths),
            "completed_items": 0,
            "failed_items": 0,
            "vqa_results": {},
            "errors": [],
        }

        # TODO: Implement actual VQA processing
        # For now, return placeholder results
        for i, image_path in enumerate(vqa_request.image_paths):
            try:
                image_results = {}

                for question in vqa_request.questions:
                    # Placeholder VQA answer
                    answer = f"Answer to '{question}' for {Path(image_path).name}"
                    image_results[question] = answer

                results["vqa_results"][image_path] = image_results
                results["completed_items"] += 1

            except Exception as e:
                error_msg = f"VQA failed for {image_path}: {str(e)}"
                results["errors"].append(error_msg)
                results["failed_items"] += 1

        # Save results if requested
        if vqa_request.save_results:
            json_path = output_dir / "vqa_results.json"
            with open(json_path, "w") as f:
                json.dump(results["vqa_results"], f, indent=2)

        return results

    except Exception as e:
        logger.error(f"Batch VQA failed: {e}")
        raise BatchProcessingError(f"Batch VQA failed: {str(e)}")


# ===== API Endpoints =====


@router.post("/generation", response_model=BatchJobResponse)
async def submit_batch_generation(request: BatchGenerationRequest):
    """提交批次圖片生成任務"""

    try:
        batch_id = str(uuid.uuid4())

        logger.info(f"Submitting batch generation: {batch_id}")
        logger.info(f"Batch name: {request.batch_name}")
        logger.info(f"Prompts: {len(request.prompts)}")

        # TODO: Submit to Celery for actual async processing
        # For now, process synchronously (not recommended for production)

        response = BatchJobResponse(
            batch_id=batch_id,
            batch_name=request.batch_name,
            status="queued",
            total_items=len(request.prompts),
            completed_items=0,
            failed_items=0,
            progress_percent=0,
        )

        # In production, this would be:
        # from workers.tasks.batch import process_batch_generation
        # task = process_batch_generation.delay(request.dict(), batch_id)
        # response.task_id = task.id

        return response

    except Exception as e:
        logger.error(f"Batch generation submission failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to submit batch generation: {str(e)}"
        )


@router.post("/caption", response_model=BatchJobResponse)
async def submit_batch_captioning(request: CaptionRequest):
    """提交批次圖片說明生成任務"""

    try:
        batch_id = str(uuid.uuid4())

        logger.info(f"Submitting batch captioning: {batch_id}")
        logger.info(f"Images: {len(request.image_paths)}")

        response = BatchJobResponse(
            batch_id=batch_id,
            batch_name=request.batch_name,
            status="queued",
            total_items=len(request.image_paths),
            completed_items=0,
            failed_items=0,
            progress_percent=0,
        )

        return response

    except Exception as e:
        logger.error(f"Batch captioning submission failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to submit batch captioning: {str(e)}"
        )


@router.post("/vqa", response_model=BatchJobResponse)
async def submit_batch_vqa(request: VQARequest):
    """提交批次視覺問答任務"""

    try:
        batch_id = str(uuid.uuid4())

        logger.info(f"Submitting batch VQA: {batch_id}")
        logger.info(f"Images: {len(request.image_paths)}")
        logger.info(f"Questions: {len(request.questions)}")

        response = BatchJobResponse(
            batch_id=batch_id,
            batch_name=request.batch_name,
            status="queued",
            total_items=len(request.image_paths),
            completed_items=0,
            failed_items=0,
            progress_percent=0,
        )

        return response

    except Exception as e:
        logger.error(f"Batch VQA submission failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to submit batch VQA: {str(e)}"
        )


@router.get("/jobs/{batch_id}", response_model=BatchJobResponse)
async def get_batch_job_status(batch_id: str):
    """查詢批次任務狀態"""

    try:
        # TODO: Query actual job status from database/Celery
        # For now, return mock status

        app_paths = get_app_paths()
        output_dir = app_paths.outputs / "batch" / batch_id

        if output_dir.exists():
            # Check if metadata exists
            metadata_path = output_dir / "batch_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                results = metadata.get("results", {})

                return BatchJobResponse(
                    batch_id=batch_id,
                    batch_name=metadata.get("batch_request", {}).get(
                        "batch_name", "Unknown"
                    ),
                    status="completed",
                    total_items=results.get("total_items", 0),
                    completed_items=results.get("completed_items", 0),
                    failed_items=results.get("failed_items", 0),
                    progress_percent=100,
                    generated_files=results.get("generated_files", []),
                    zip_file=results.get("zip_file"),
                    errors=results.get("errors", []),
                )

        # If no output directory, assume job doesn't exist
        raise HTTPException(status_code=404, detail="Batch job not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get batch job status {batch_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get job status: {str(e)}"
        )


@router.get("/jobs")
async def list_batch_jobs(
    status: Optional[str] = None, limit: int = Field(default=50, ge=1, le=200)
):
    """列出批次任務"""

    try:
        app_paths = get_app_paths()
        batch_dir = app_paths.outputs / "batch"

        if not batch_dir.exists():
            return []

        jobs = []

        # Scan batch directories
        for job_dir in batch_dir.iterdir():
            if job_dir.is_dir():
                metadata_path = job_dir / "batch_metadata.json"

                if metadata_path.exists():
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)

                        results = metadata.get("results", {})

                        job_info = {
                            "batch_id": job_dir.name,
                            "batch_name": metadata.get("batch_request", {}).get(
                                "batch_name", "Unknown"
                            ),
                            "status": "completed",
                            "total_items": results.get("total_items", 0),
                            "completed_items": results.get("completed_items", 0),
                            "failed_items": results.get("failed_items", 0),
                            "created_at": metadata.get("completed_at", "Unknown"),
                        }

                        # Apply status filter
                        if status is None or job_info["status"] == status:
                            jobs.append(job_info)

                    except Exception as e:
                        logger.warning(
                            f"Failed to read metadata for {job_dir.name}: {e}"
                        )

        # Sort by creation time (newest first) and limit
        jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return jobs[:limit]

    except Exception as e:
        logger.error(f"Failed to list batch jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")


@router.delete("/jobs/{batch_id}")
async def cancel_batch_job(batch_id: str):
    """取消批次任務"""

    try:
        # TODO: Cancel actual Celery task
        logger.info(f"Cancelling batch job: {batch_id}")

        # Clean up output directory
        app_paths = get_app_paths()
        output_dir = app_paths.outputs / "batch" / batch_id

        if output_dir.exists():
            import shutil

            shutil.rmtree(output_dir)

        return {
            "batch_id": batch_id,
            "status": "cancelled",
            "message": "Batch job cancelled and files cleaned up",
        }

    except Exception as e:
        logger.error(f"Failed to cancel batch job {batch_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")


@router.get("/download/{batch_id}")
async def download_batch_results(batch_id: str):
    """下載批次處理結果"""

    try:
        app_paths = get_app_paths()
        output_dir = app_paths.outputs / "batch" / batch_id

        # Check for ZIP file first
        zip_file = output_dir / f"{batch_id}_results.zip"
        if zip_file.exists():
            return FileResponse(
                path=str(zip_file),
                media_type="application/zip",
                filename=f"batch_{batch_id}_results.zip",
            )

        # If no ZIP, create one on the fly
        if output_dir.exists():
            import zipfile
            from io import BytesIO

            zip_buffer = BytesIO()

            with zipfile.ZipFile(zip_buffer, "w") as zipf:
                for file_path in output_dir.rglob("*"):
                    if file_path.is_file() and file_path.suffix in [
                        ".png",
                        ".jpg",
                        ".txt",
                        ".json",
                    ]:
                        arcname = file_path.relative_to(output_dir)
                        zipf.write(file_path, arcname)

            zip_buffer.seek(0)

            # Save the ZIP for future downloads
            with open(zip_file, "wb") as f:
                f.write(zip_buffer.getvalue())

            return FileResponse(
                path=str(zip_file),
                media_type="application/zip",
                filename=f"batch_{batch_id}_results.zip",
            )

        raise HTTPException(status_code=404, detail="Batch results not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download batch results {batch_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to download results: {str(e)}"
        )


@router.get("/templates")
async def get_batch_templates():
    """取得批次處理模板"""

    try:
        templates = {
            "generation": {
                "name": "Batch Image Generation",
                "description": "Generate multiple images from prompts",
                "example": {
                    "batch_name": "anime_portraits",
                    "prompts": [
                        "anime girl with blue hair",
                        "anime boy with red eyes",
                        "anime character in school uniform",
                    ],
                    "model_type": "sd15",
                    "width": 512,
                    "height": 512,
                    "save_individually": True,
                    "create_zip": True,
                },
            },
            "caption": {
                "name": "Batch Image Captioning",
                "description": "Generate captions for multiple images",
                "example": {
                    "batch_name": "image_captions",
                    "image_paths": ["/path/to/image1.jpg", "/path/to/image2.jpg"],
                    "model_name": "blip2",
                    "save_to_txt": True,
                },
            },
            "vqa": {
                "name": "Batch Visual Question Answering",
                "description": "Answer questions about multiple images",
                "example": {
                    "batch_name": "image_analysis",
                    "image_paths": ["/path/to/image1.jpg", "/path/to/image2.jpg"],
                    "questions": [
                        "What is in this image?",
                        "What colors are dominant?",
                    ],
                    "model_name": "llava",
                },
            },
        }

        return templates

    except Exception as e:
        logger.error(f"Failed to get batch templates: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get templates: {str(e)}"
        )


@router.get("/system/status")
async def get_batch_system_status():
    """取得批次處理系統狀態"""

    try:
        app_paths = get_app_paths()
        batch_dir = app_paths.outputs / "batch"

        # Count jobs by status
        total_jobs = 0
        if batch_dir.exists():
            total_jobs = len([d for d in batch_dir.iterdir() if d.is_dir()])

        # TODO: Get actual queue status from Celery

        return {
            "status": "healthy",
            "total_jobs": total_jobs,
            "active_jobs": 0,  # TODO: Get from Celery
            "queued_jobs": 0,  # TODO: Get from Celery
            "completed_jobs": total_jobs,  # Assume all are completed for now
            "storage_used_gb": 0,  # TODO: Calculate actual storage
            "available_models": ["sd15", "sdxl", "blip2", "llava"],
            "max_batch_size": 100,
        }

    except Exception as e:
        logger.error(f"Failed to get batch system status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get system status: {str(e)}"
        )
