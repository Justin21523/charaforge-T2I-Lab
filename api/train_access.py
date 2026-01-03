"""Training job access metadata (owner tracking).

Stores an owner mapping for training jobs so status/cancel/WebSocket streams can
enforce owner-only access (admin can override).
"""

from __future__ import annotations

import json
import time
from hashlib import sha256
from pathlib import Path

from core.config import get_app_paths, get_cache_paths

ACCESS_META_FILENAME = "_access.json"


def _job_key(job_id: str) -> str:
    raw = str(job_id or "")
    digest = sha256(raw.encode("utf-8")).hexdigest()[:32]
    return f"job_{digest}"


def access_meta_path(job_id: str) -> Path:
    cache_paths = get_cache_paths()
    return cache_paths.cache / "jobs" / "train" / _job_key(job_id) / ACCESS_META_FILENAME


def _legacy_access_meta_path(job_id: str) -> Path:
    app_paths = get_app_paths()
    return app_paths.outputs / "train" / _job_key(job_id) / ACCESS_META_FILENAME


def write_train_access_meta(job_id: str, *, owner: str) -> None:
    job_id = str(job_id or "").strip()
    owner = str(owner or "").strip()
    if not job_id or not owner:
        return

    path = access_meta_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"job_id": job_id, "owner": owner, "created_at": time.time()}
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        return


def read_train_access_owner(job_id: str) -> str | None:
    job_id = str(job_id or "").strip()
    if not job_id:
        return None

    path = access_meta_path(job_id)
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception:
        legacy = _legacy_access_meta_path(job_id)
        try:
            raw = legacy.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception:
            return None
    if not isinstance(data, dict):
        return None
    if str(data.get("job_id") or "") != job_id:
        return None
    owner = data.get("owner")
    return str(owner) if owner else None
