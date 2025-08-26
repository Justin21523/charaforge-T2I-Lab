# api/routers/export.py - Export endpoints implementation
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import zipfile
import tempfile
import json
from pathlib import Path

from core.config import get_cache_paths, get_run_output_dir
from core.train.registry import ModelRegistry

router = APIRouter()


class ExportLoRARequest(BaseModel):
    run_id: str = Field(..., description="Training run ID to export")
    format: str = Field(default="safetensors", pattern="^(safetensors|pytorch)$")
    include_samples: bool = Field(default=True, description="Include sample images")
    include_logs: bool = Field(default=False, description="Include training logs")
    include_config: bool = Field(default=True, description="Include training config")


class ExportResponse(BaseModel):
    export_id: str
    download_url: str
    size_bytes: int
    expires_at: datetime
    included_files: List[str]


@router.post("/lora", response_model=ExportResponse)
async def export_lora(request: ExportLoRARequest, background_tasks: BackgroundTasks):
    """Export LoRA model and artifacts as downloadable package"""

    try:
        # Verify run exists
        registry = ModelRegistry()
        run_info = registry.get_run(request.run_id)

        if not run_info:
            raise HTTPException(
                status_code=404, detail=f"Run not found: {request.run_id}"
            )

        if run_info["status"] != "completed":
            raise HTTPException(
                status_code=400, detail=f"Run not completed: {run_info['status']}"
            )

        # Get run directory
        run_dir = get_run_output_dir(request.run_id)
        if not run_dir.exists():
            raise HTTPException(
                status_code=404, detail="Run output directory not found"
            )

        # Create export package
        export_id = f"export_{request.run_id}_{int(datetime.now().timestamp())}"
        cache_paths = get_cache_paths()
        export_dir = cache_paths.outputs / "exports"
        export_dir.mkdir(exist_ok=True)

        zip_path = export_dir / f"{export_id}.zip"
        included_files = []

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Add LoRA weights
            checkpoints_dir = run_dir / "checkpoints" / "final"
            if checkpoints_dir.exists():
                for file in checkpoints_dir.rglob("*"):
                    if file.is_file():
                        arcname = f"lora_weights/{file.relative_to(checkpoints_dir)}"
                        zipf.write(file, arcname)
                        included_files.append(arcname)

            # Add samples if requested
            if request.include_samples:
                samples_dir = run_dir / "samples"
                if samples_dir.exists():
                    for file in samples_dir.rglob("*.png"):
                        arcname = f"samples/{file.relative_to(samples_dir)}"
                        zipf.write(file, arcname)
                        included_files.append(arcname)

            # Add config if requested
            if request.include_config:
                config_files = [
                    run_dir / "training_logs.json",
                    checkpoints_dir / "training_state.json",
                ]

                for file in config_files:
                    if file.exists():
                        arcname = f"config/{file.name}"
                        zipf.write(file, arcname)
                        included_files.append(arcname)

            # Add logs if requested
            if request.include_logs:
                logs_dir = run_dir / "logs"
                if logs_dir.exists():
                    for file in logs_dir.rglob("*.log"):
                        arcname = f"logs/{file.relative_to(logs_dir)}"
                        zipf.write(file, arcname)
                        included_files.append(arcname)

            # Add metadata
            metadata = {
                "export_id": export_id,
                "run_id": request.run_id,
                "exported_at": datetime.now().isoformat(),
                "format": request.format,
                "run_info": run_info,
                "included_files": included_files,
            }

            zipf.writestr("metadata.json", json.dumps(metadata, indent=2))
            included_files.append("metadata.json")

        # Schedule cleanup after 24 hours
        expire_time = datetime.now() + timedelta(hours=24)
        background_tasks.add_task(cleanup_export, zip_path, delay=24 * 3600)

        size_bytes = zip_path.stat().st_size

        return ExportResponse(
            export_id=export_id,
            download_url=f"/api/v1/export/download/{export_id}",
            size_bytes=size_bytes,
            expires_at=expire_time,
            included_files=included_files,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.get("/download/{export_id}")
async def download_export(export_id: str):
    """Download exported package"""

    cache_paths = get_cache_paths()
    export_file = cache_paths.outputs / "exports" / f"{export_id}.zip"

    if not export_file.exists():
        raise HTTPException(status_code=404, detail="Export not found or expired")

    return FileResponse(
        path=export_file, filename=f"{export_id}.zip", media_type="application/zip"
    )


@router.get("/list")
async def list_exports():
    """List available exports"""

    cache_paths = get_cache_paths()
    export_dir = cache_paths.outputs / "exports"

    if not export_dir.exists():
        return {"exports": []}

    exports = []
    for zip_file in export_dir.glob("*.zip"):
        try:
            stat = zip_file.stat()
            created = datetime.fromtimestamp(stat.st_ctime)

            exports.append(
                {
                    "export_id": zip_file.stem,
                    "size_bytes": stat.st_size,
                    "created_at": created.isoformat(),
                    "download_url": f"/api/v1/export/download/{zip_file.stem}",
                }
            )
        except Exception:
            continue

    return {"exports": exports}


async def cleanup_export(file_path: Path, delay: int = 0):
    """Background task to cleanup old exports"""
    import asyncio

    await asyncio.sleep(delay)

    try:
        if file_path.exists():
            file_path.unlink()
    except Exception:
        pass
