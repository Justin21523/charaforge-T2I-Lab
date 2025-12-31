"""Dataset management helpers.

Conventions (AI_WAREHOUSE 3.0):
- Datasets live under: `${AI_DATASETS_ROOT}/${PROJECT_SLUG}/`
- Recommended: `raw/<dataset_name>/` with `*.png|jpg|webp` + optional `*.txt` captions.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any, Dict, List, Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from core.config import get_cache_paths, get_settings
from core.train.dataset import validate_image_dataset

router = APIRouter(prefix="/datasets", tags=["datasets"])

DatasetKind = Literal["raw", "processed"]


def _datasets_root() -> Path:
    return get_cache_paths().datasets.resolve()


def _resolve_dataset_path(value: str) -> Path:
    raw = (value or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="dataset_path is required")

    base_root = _datasets_root()
    p = Path(raw)

    if p.is_absolute():
        candidate = p
    else:
        if "/" not in raw and "\\" not in raw:
            candidate = base_root / "raw" / raw
        else:
            candidate = base_root / raw

    resolved = candidate.resolve()
    if base_root not in resolved.parents and resolved != base_root:
        raise HTTPException(status_code=400, detail=f"dataset_path must be under {base_root}")
    return resolved


def _safe_extract_zip(zip_path: Path, dest: Path) -> int:
    extracted = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue

            # Prevent Zip Slip.
            target = (dest / member.filename).resolve()
            if dest.resolve() not in target.parents and target != dest.resolve():
                raise HTTPException(status_code=400, detail="Invalid zip contents")

            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as out:
                out.write(src.read())
            extracted += 1
    return extracted


class DatasetValidateRequest(BaseModel):
    dataset_path: str = Field(..., min_length=1, max_length=2000)


@router.get("/root")
async def get_dataset_root() -> Dict[str, Any]:
    root = _datasets_root()
    return {
        "datasets_root": str(root),
        "raw_root": str(root / "raw"),
        "processed_root": str(root / "processed"),
        "example": str(root / "raw" / "my_dataset"),
    }


@router.get("/list")
async def list_datasets(kind: DatasetKind = "raw") -> List[Dict[str, Any]]:
    root = _datasets_root() / kind
    if not root.exists():
        return []

    items = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        items.append({"name": p.name, "path": str(p)})
    return items


@router.post("/validate")
async def validate_dataset(req: DatasetValidateRequest) -> Dict[str, Any]:
    path = _resolve_dataset_path(req.dataset_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"dataset_path not found: {path}")
    if not path.is_dir():
        raise HTTPException(status_code=400, detail="dataset_path must be a directory")

    ok, errors = validate_image_dataset(path)
    return {"ok": ok, "errors": errors, "dataset_path": str(path)}


@router.post("/upload")
async def upload_dataset_zip(
    dataset_name: str = Form(..., min_length=1, max_length=120),
    file: UploadFile = File(...),
    kind: DatasetKind = Form("raw"),
    overwrite: bool = Form(False),
) -> Dict[str, Any]:
    settings = get_settings()
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip uploads are supported")

    base_root = _datasets_root() / kind
    dest = (base_root / dataset_name).resolve()
    if base_root.resolve() not in dest.parents:
        raise HTTPException(status_code=400, detail="Invalid dataset_name")

    if dest.exists() and not overwrite:
        raise HTTPException(
            status_code=409, detail="Dataset already exists (use overwrite=true)"
        )

    dest.mkdir(parents=True, exist_ok=True)
    temp_zip = dest / "__upload.zip"

    # Enforce a basic size guard (server-side).
    max_bytes = int(settings.api.max_file_size_mb) * 1024 * 1024
    written = 0
    with open(temp_zip, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                raise HTTPException(status_code=413, detail="Upload too large")
            f.write(chunk)

    extracted = _safe_extract_zip(temp_zip, dest)
    temp_zip.unlink(missing_ok=True)

    ok, errors = validate_image_dataset(dest)
    return {
        "status": "ok" if ok else "warning",
        "dataset_name": dataset_name,
        "dataset_path": str(dest),
        "files_extracted": extracted,
        "validation": {"ok": ok, "errors": errors},
    }
