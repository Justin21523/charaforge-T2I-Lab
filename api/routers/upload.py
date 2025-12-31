"""File upload endpoints (datasets, control images, etc.)."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from core.config import get_app_paths

router = APIRouter(tags=["upload"])


def _uploads_root() -> Path:
    app_paths = get_app_paths()
    return app_paths.outputs / "uploads"


def _safe_segment(value: str) -> str:
    value = (value or "").strip().lower()
    if not value:
        return "misc"
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789-_"
    cleaned = "".join(ch for ch in value if ch in allowed)
    return cleaned or "misc"


@router.post("/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    file_type: str = Form("image"),
) -> Dict[str, Any]:
    root = _uploads_root()
    bucket = _safe_segment(file_type)
    dest_dir = root / bucket
    dest_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "").suffix.lower()
    if not suffix:
        suffix = ".bin"

    upload_id = uuid.uuid4().hex
    dest_path = dest_dir / f"{upload_id}{suffix}"

    data = await file.read()
    dest_path.write_bytes(data)

    base = str(request.base_url).rstrip("/")
    return {
        "upload_id": upload_id,
        "file_type": bucket,
        "filename": file.filename,
        "size_bytes": len(data),
        "path": str(dest_path),
        "url": f"{base}/api/v1/upload/{bucket}/{upload_id}{suffix}",
    }


@router.get("/upload/{bucket}/{filename}")
async def get_upload(bucket: str, filename: str):
    bucket_safe = _safe_segment(bucket)
    if "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    root = _uploads_root()
    path = (root / bucket_safe / filename).resolve()
    if root.resolve() not in path.parents:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path=str(path), filename=path.name)

