# backend/jobs/batch_tasks.py - Batch Task Implementations
from celery import current_task
from backend.jobs.worker import celery_app
from backend.core.pipeline_loader import (
    get_caption_pipeline,
    get_vqa_pipeline,
    get_t2i_pipeline,
)
from backend.utils.logging import get_logger
from PIL import Image
import json
import traceback
from datetime import datetime
import uuid
import os

logger = get_logger(__name__)


@celery_app.task(bind=True, name="process_caption_batch")
def process_caption_batch(self, batch_data):
    """Process batch caption generation tasks"""
    try:
        logger.info(f"Starting caption batch: {self.request.id}")

        # Update task state
        self.update_state(
            state="PROGRESS",
            meta={
                "current": 0,
                "total": len(batch_data["items"]),
                "status": "Loading model...",
            },
        )

        # Load caption pipeline
        pipeline = get_caption_pipeline()
        results = []

        for i, item in enumerate(batch_data["items"]):
            try:
                # Load image
                image_path = item["image_path"]
                image = Image.open(image_path).convert("RGB")

                # Generate caption
                caption = pipeline(
                    image,
                    max_length=item.get("max_length", 50),
                    num_beams=item.get("num_beams", 3),
                )[0]["generated_text"]

                # Save result
                result = {
                    "id": item.get("id", str(uuid.uuid4())),
                    "image_path": image_path,
                    "caption": caption,
                    "timestamp": datetime.now().isoformat(),
                    "status": "success",
                }
                results.append(result)

                # Update progress
                self.update_state(
                    state="PROGRESS",
                    meta={
                        "current": i + 1,
                        "total": len(batch_data["items"]),
                        "status": f'Processed {i + 1}/{len(batch_data["items"])}',
                    },
                )

            except Exception as e:
                logger.error(f"Error processing item {i}: {str(e)}")
                results.append(
                    {
                        "id": item.get("id", str(uuid.uuid4())),
                        "image_path": item.get("image_path", ""),
                        "error": str(e),
                        "status": "failed",
                    }
                )

        # Save batch results
        output_dir = f"{AI_CACHE_ROOT}/outputs/multi-modal-lab/batch"
        batch_id = batch_data.get("batch_id", self.request.id)
        result_file = f"{output_dir}/caption_batch_{batch_id}.json"

        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "batch_id": batch_id,
                    "task_id": self.request.id,
                    "type": "caption",
                    "completed_at": datetime.now().isoformat(),
                    "total_items": len(batch_data["items"]),
                    "successful": len(
                        [r for r in results if r.get("status") == "success"]
                    ),
                    "failed": len([r for r in results if r.get("status") == "failed"]),
                    "results": results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info(f"Caption batch completed: {batch_id}")
        return {
            "batch_id": batch_id,
            "result_file": result_file,
            "total_items": len(batch_data["items"]),
            "successful": len([r for r in results if r.get("status") == "success"]),
            "failed": len([r for r in results if r.get("status") == "failed"]),
        }

    except Exception as e:
        logger.error(f"Batch caption task failed: {str(e)}\n{traceback.format_exc()}")
        self.update_state(
            state="FAILURE", meta={"error": str(e), "traceback": traceback.format_exc()}
        )
        raise


@celery_app.task(bind=True, name="process_vqa_batch")
def process_vqa_batch(self, batch_data):
    """Process batch VQA tasks"""
    try:
        logger.info(f"Starting VQA batch: {self.request.id}")

        self.update_state(
            state="PROGRESS",
            meta={
                "current": 0,
                "total": len(batch_data["items"]),
                "status": "Loading VQA model...",
            },
        )

        pipeline = get_vqa_pipeline()
        results = []

        for i, item in enumerate(batch_data["items"]):
            try:
                # Load image and question
                image_path = item["image_path"]
                question = item["question"]
                image = Image.open(image_path).convert("RGB")

                # Generate answer
                answer = pipeline(
                    image, question, max_length=item.get("max_length", 100)
                )

                result = {
                    "id": item.get("id", str(uuid.uuid4())),
                    "image_path": image_path,
                    "question": question,
                    "answer": answer,
                    "timestamp": datetime.now().isoformat(),
                    "status": "success",
                }
                results.append(result)

                self.update_state(
                    state="PROGRESS",
                    meta={
                        "current": i + 1,
                        "total": len(batch_data["items"]),
                        "status": f'Processed {i + 1}/{len(batch_data["items"])}',
                    },
                )

            except Exception as e:
                logger.error(f"Error processing VQA item {i}: {str(e)}")
                results.append(
                    {
                        "id": item.get("id", str(uuid.uuid4())),
                        "image_path": item.get("image_path", ""),
                        "question": item.get("question", ""),
                        "error": str(e),
                        "status": "failed",
                    }
                )

        # Save results
        output_dir = f"{AI_CACHE_ROOT}/outputs/multi-modal-lab/batch"
        batch_id = batch_data.get("batch_id", self.request.id)
        result_file = f"{output_dir}/vqa_batch_{batch_id}.json"

        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "batch_id": batch_id,
                    "task_id": self.request.id,
                    "type": "vqa",
                    "completed_at": datetime.now().isoformat(),
                    "total_items": len(batch_data["items"]),
                    "successful": len(
                        [r for r in results if r.get("status") == "success"]
                    ),
                    "failed": len([r for r in results if r.get("status") == "failed"]),
                    "results": results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info(f"VQA batch completed: {batch_id}")
        return {
            "batch_id": batch_id,
            "result_file": result_file,
            "total_items": len(batch_data["items"]),
            "successful": len([r for r in results if r.get("status") == "success"]),
            "failed": len([r for r in results if r.get("status") == "failed"]),
        }

    except Exception as e:
        logger.error(f"Batch VQA task failed: {str(e)}\n{traceback.format_exc()}")
        self.update_state(
            state="FAILURE", meta={"error": str(e), "traceback": traceback.format_exc()}
        )
        raise
