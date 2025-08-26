# backend/api/batch.py - Batch API Endpoints
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import logging
import json
import uuid
import pandas as pd
import io
from typing import Optional
from workers.celery_app import celery_app

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/batch/submit", response_model=dict)
async def submit_batch_job(
    task_type: str = Form(...),
    file: UploadFile = File(...),
    batch_name: Optional[str] = Form(None),
):
    """Submit a batch processing job"""
    try:
        # Validate task type
        valid_types = ["caption", "vqa", "t2i"]
        if task_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid task_type. Must be one of: {valid_types}",
            )

        # Generate batch ID
        batch_id = str(uuid.uuid4())

        # Read uploaded file
        content = await file.read()

        # Parse file based on extension
        if file.filename.endswith(".csv"):  # type: ignore
            df = pd.read_csv(io.StringIO(content.decode("utf-8")))
            items = df.to_dict("records")
        elif file.filename.endswith(".json"):  # type: ignore
            data = json.loads(content.decode("utf-8"))
            items = data if isinstance(data, list) else data.get("items", [])
        else:
            raise HTTPException(
                status_code=400, detail="File must be CSV or JSON format"
            )

        # Validate required fields
        if task_type == "caption":
            required_fields = ["image_path"]
        elif task_type == "vqa":
            required_fields = ["image_path", "question"]
        else:
            raise HTTPException(
                status_code=400, detail=f"Task type {task_type} not implemented yet"
            )

        for item in items:
            for field in required_fields:
                if field not in item:
                    raise HTTPException(
                        status_code=400, detail=f"Missing required field: {field}"
                    )

        # Submit task to Celery
        batch_data = {
            "batch_id": batch_id,
            "batch_name": batch_name or f"{task_type}_batch_{batch_id[:8]}",
            "task_type": task_type,
            "items": items,
        }

        if task_type == "caption":
            task = process_caption_batch.delay(batch_data)
        elif task_type == "vqa":
            task = process_vqa_batch.delay(batch_data)

        logger.info(f"Submitted batch job: {batch_id}, task_id: {task.id}")

        return {
            "batch_id": batch_id,
            "task_id": task.id,
            "status": "submitted",
            "total_items": len(items),
            "message": f"Batch job submitted successfully with {len(items)} items",
        }

    except Exception as e:
        logger.error(f"Error submitting batch job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/batch/status/{task_id}")
async def get_batch_status(task_id: str):
    """Get batch job status"""
    try:
        # Get task status from Celery
        task = celery_app.AsyncResult(task_id)

        response = {
            "task_id": task_id,
            "status": task.status,
            "result": None,
            "error": None,
            "progress": None,
        }

        if task.status == "PENDING":
            response["message"] = "Task is waiting to be processed"
        elif task.status == "PROGRESS":
            response["progress"] = task.result
            response["message"] = task.result.get("status", "Processing...")
        elif task.status == "SUCCESS":
            response["result"] = task.result
            response["message"] = "Task completed successfully"
        elif task.status == "FAILURE":
            response["error"] = str(task.result)
            response["message"] = "Task failed"

        return response

    except Exception as e:
        logger.error(f"Error getting batch status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/batch/list")
async def list_batch_jobs():
    """List all batch jobs"""
    try:
        # Get active tasks from Celery
        inspect = celery_app.control.inspect()
        active_tasks = inspect.active()
        scheduled_tasks = inspect.scheduled()

        tasks = []

        # Process active tasks
        if active_tasks:
            for worker, task_list in active_tasks.items():
                for task in task_list:
                    tasks.append(
                        {
                            "task_id": task["id"],
                            "name": task["name"],
                            "worker": worker,
                            "status": "RUNNING",
                            "args": task.get("args", []),
                            "kwargs": task.get("kwargs", {}),
                        }
                    )

        # Process scheduled tasks
        if scheduled_tasks:
            for worker, task_list in scheduled_tasks.items():
                for task in task_list:
                    tasks.append(
                        {
                            "task_id": task["request"]["id"],
                            "name": task["request"]["name"],
                            "worker": worker,
                            "status": "SCHEDULED",
                            "eta": task.get("eta"),
                            "args": task["request"].get("args", []),
                            "kwargs": task["request"].get("kwargs", {}),
                        }
                    )

        return {"tasks": tasks, "total": len(tasks)}

    except Exception as e:
        logger.error(f"Error listing batch jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/batch/cancel/{task_id}")
async def cancel_batch_job(task_id: str):
    """Cancel a batch job"""
    try:
        celery_app.control.revoke(task_id, terminate=True)
        logger.info(f"Cancelled batch job: {task_id}")

        return {
            "task_id": task_id,
            "status": "cancelled",
            "message": "Task cancelled successfully",
        }

    except Exception as e:
        logger.error(f"Error cancelling batch job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
