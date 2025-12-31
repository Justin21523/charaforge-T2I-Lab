"""Persistent API key store (hashed keys on disk).

This backs admin-only key management endpoints and allows running the API
without baking shared secrets into frontend environment variables.
"""

from __future__ import annotations

import hmac
import json
import logging
import secrets
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import pbkdf2_hmac
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from core.config import get_cache_paths

logger = logging.getLogger(__name__)

KEY_PREFIX = "cfk"
PBKDF2_ITERATIONS = 200_000
SALT_BYTES = 16
SECRET_BYTES = 32


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pbkdf2_hex(secret: str, salt: bytes, iterations: int) -> str:
    derived = pbkdf2_hmac("sha256", secret.encode("utf-8"), salt, int(iterations))
    return derived.hex()


def _parse_key(raw_key: str) -> Optional[Tuple[str, str]]:
    prefix = f"{KEY_PREFIX}_"
    if not raw_key or not raw_key.startswith(prefix):
        return None
    rest = raw_key[len(prefix) :]
    if "." not in rest:
        return None
    key_id, secret = rest.split(".", 1)
    key_id = key_id.strip()
    secret = secret.strip()
    if not key_id or not secret:
        return None
    return key_id, secret


@dataclass(frozen=True)
class VerifiedKey:
    key_id: str
    role: str
    scopes: Set[str]
    label: Optional[str] = None


class APIKeyStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._cache_mtime: Optional[float] = None
        self._cache: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def default(cls) -> "APIKeyStore":
        cache_paths = get_cache_paths()
        return cls(cache_paths.cache / "auth" / "api_keys.json")

    def has_active_keys(self) -> bool:
        keys = self._load_records()
        return any(not record.get("revoked_at") for record in keys.values())

    def verify(self, raw_key: str) -> Optional[VerifiedKey]:
        parsed = _parse_key(raw_key)
        if not parsed:
            return None

        key_id, secret = parsed
        record = self._load_records().get(key_id)
        if not record or record.get("revoked_at"):
            return None

        try:
            salt = bytes.fromhex(str(record.get("salt_hex") or ""))
            iterations = int(record.get("iterations") or PBKDF2_ITERATIONS)
            expected = str(record.get("hash_hex") or "")
        except Exception:
            return None

        computed = _pbkdf2_hex(secret, salt, iterations)
        if not expected or not hmac.compare_digest(computed, expected):
            return None

        role = str(record.get("role") or "user")
        scopes = set(record.get("scopes") or [])
        label = record.get("label")
        return VerifiedKey(key_id=key_id, role=role, scopes=scopes, label=label)

    def list_keys(self, *, include_revoked: bool = False) -> List[Dict[str, Any]]:
        records = self._load_records()
        out = []
        for record in records.values():
            if record.get("revoked_at") and not include_revoked:
                continue
            out.append(
                {
                    "key_id": record.get("key_id"),
                    "label": record.get("label"),
                    "role": record.get("role"),
                    "scopes": record.get("scopes") or [],
                    "created_at": record.get("created_at"),
                    "revoked_at": record.get("revoked_at"),
                }
            )
        out.sort(key=lambda r: str(r.get("created_at") or ""))
        return out

    def create_key(
        self,
        *,
        role: str = "user",
        scopes: Iterable[str] = (),
        label: Optional[str] = None,
    ) -> Tuple[str, str]:
        role = str(role or "user").lower()
        if role not in {"admin", "user"}:
            raise ValueError("role must be admin|user")

        scopes_list = sorted({s.strip() for s in scopes if str(s).strip()})

        key_id = uuid.uuid4().hex[:12]
        secret = secrets.token_urlsafe(SECRET_BYTES)
        raw_key = f"{KEY_PREFIX}_{key_id}.{secret}"

        salt = secrets.token_bytes(SALT_BYTES)
        hash_hex = _pbkdf2_hex(secret, salt, PBKDF2_ITERATIONS)

        record = {
            "key_id": key_id,
            "label": label,
            "role": role,
            "scopes": scopes_list,
            "created_at": _utc_now_iso(),
            "revoked_at": None,
            "salt_hex": salt.hex(),
            "hash_hex": hash_hex,
            "iterations": PBKDF2_ITERATIONS,
        }

        with self._lock:
            records = self._load_records_unlocked()
            records[key_id] = record
            self._save_records_unlocked(records)

        return key_id, raw_key

    def revoke_key(self, key_id: str) -> bool:
        key_id = str(key_id or "").strip()
        if not key_id:
            return False

        with self._lock:
            records = self._load_records_unlocked()
            record = records.get(key_id)
            if not record or record.get("revoked_at"):
                return False
            record["revoked_at"] = _utc_now_iso()
            records[key_id] = record
            self._save_records_unlocked(records)
            return True

    def rotate_key(self, key_id: str) -> Tuple[str, str] | None:
        key_id = str(key_id or "").strip()
        if not key_id:
            return None

        with self._lock:
            records = self._load_records_unlocked()
            record = records.get(key_id)
            if not record:
                return None

            role = str(record.get("role") or "user")
            scopes = record.get("scopes") or []
            label = record.get("label")

            record["revoked_at"] = record.get("revoked_at") or _utc_now_iso()
            records[key_id] = record

            new_id = uuid.uuid4().hex[:12]
            secret = secrets.token_urlsafe(SECRET_BYTES)
            raw_key = f"{KEY_PREFIX}_{new_id}.{secret}"
            salt = secrets.token_bytes(SALT_BYTES)
            hash_hex = _pbkdf2_hex(secret, salt, PBKDF2_ITERATIONS)

            records[new_id] = {
                "key_id": new_id,
                "label": label,
                "role": role,
                "scopes": list(scopes),
                "created_at": _utc_now_iso(),
                "revoked_at": None,
                "salt_hex": salt.hex(),
                "hash_hex": hash_hex,
                "iterations": PBKDF2_ITERATIONS,
            }

            self._save_records_unlocked(records)
            return new_id, raw_key

    def _load_records(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return dict(self._load_records_unlocked())

    def _load_records_unlocked(self) -> Dict[str, Dict[str, Any]]:
        path = self.path
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            self._cache_mtime = None
            self._cache = {}
            return {}
        except Exception:
            return {}

        if self._cache_mtime == mtime and self._cache:
            return self._cache

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to load API key store: %s", path)
            self._cache_mtime = mtime
            self._cache = {}
            return {}

        keys = data.get("keys") if isinstance(data, dict) else None
        if not isinstance(keys, list):
            self._cache_mtime = mtime
            self._cache = {}
            return {}

        records: Dict[str, Dict[str, Any]] = {}
        for item in keys:
            if not isinstance(item, dict):
                continue
            key_id = str(item.get("key_id") or "").strip()
            if not key_id:
                continue
            records[key_id] = dict(item)

        self._cache_mtime = mtime
        self._cache = records
        return records

    def _save_records_unlocked(self, records: Dict[str, Dict[str, Any]]) -> None:
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": 1,
            "keys": list(records.values()),
        }

        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

        try:
            self._cache_mtime = path.stat().st_mtime
            self._cache = dict(records)
        except Exception:
            return

