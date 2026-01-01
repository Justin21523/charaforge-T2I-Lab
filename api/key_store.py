"""Persistent API key store (hashed keys on disk).

This backs admin-only key management endpoints and allows running the API
without baking shared secrets into frontend environment variables.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import secrets
import threading
import time
import uuid
from contextlib import contextmanager
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
LOCK_STALE_SECONDS = 120.0
LOCK_TIMEOUT_SECONDS = 5.0
LOCK_POLL_SECONDS = 0.05
LAST_USED_MIN_INTERVAL_SECONDS = 60.0


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
        self._last_used_touch: Dict[str, float] = {}

    def _lock_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".lock")

    def _audit_path(self) -> Path:
        return self.path.with_name("api_keys_audit.jsonl")

    @contextmanager
    def _file_lock(self) -> Any:
        lock_path = self._lock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS

        acquired = False
        while time.monotonic() < deadline:
            try:
                fd = os.open(
                    str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
                )
            except FileExistsError:
                try:
                    age = time.time() - lock_path.stat().st_mtime
                    if age > LOCK_STALE_SECONDS:
                        lock_path.unlink()
                except FileNotFoundError:
                    pass
                except Exception:
                    pass
                time.sleep(LOCK_POLL_SECONDS)
                continue
            except Exception as exc:
                raise RuntimeError(f"Failed to acquire key store lock: {exc}") from exc
            else:
                try:
                    try:
                        os.write(fd, f"pid={os.getpid()}\n".encode("utf-8"))
                    except Exception:
                        pass
                finally:
                    try:
                        os.close(fd)
                    except Exception:
                        pass
                acquired = True
                break

        if not acquired:
            raise TimeoutError("Timed out acquiring key store lock")

        try:
            yield
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass

    def _append_audit(self, event: Dict[str, Any]) -> None:
        payload = dict(event or {})
        payload.setdefault("ts", _utc_now_iso())
        path = self._audit_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            return

    @classmethod
    def default(cls) -> "APIKeyStore":
        cache_paths = get_cache_paths()
        return cls(cache_paths.cache / "auth" / "api_keys.json")

    def has_active_keys(self) -> bool:
        keys = self._load_records()
        return any(not record.get("revoked_at") for record in keys.values())

    def is_active_key_id(self, key_id: str) -> bool:
        key_id = str(key_id or "").strip()
        if not key_id:
            return False
        record = self._load_records().get(key_id)
        return bool(record and not record.get("revoked_at"))

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
                    "last_used_at": record.get("last_used_at"),
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
        actor_key_id: Optional[str] = None,
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
            "last_used_at": None,
            "revoked_at": None,
            "salt_hex": salt.hex(),
            "hash_hex": hash_hex,
            "iterations": PBKDF2_ITERATIONS,
        }

        with self._lock:
            with self._file_lock():
                records = self._load_records_unlocked()
                records[key_id] = record
                self._save_records_unlocked(records)

        self._append_audit(
            {
                "action": "create_key",
                "actor_key_id": actor_key_id,
                "key_id": key_id,
                "role": role,
                "scopes": scopes_list,
                "label": label,
            }
        )
        return key_id, raw_key

    def revoke_key(self, key_id: str, *, actor_key_id: Optional[str] = None) -> bool:
        key_id = str(key_id or "").strip()
        if not key_id:
            return False

        with self._lock:
            with self._file_lock():
                records = self._load_records_unlocked()
                record = records.get(key_id)
                if not record or record.get("revoked_at"):
                    return False
                record["revoked_at"] = _utc_now_iso()
                records[key_id] = record
                self._save_records_unlocked(records)

        self._append_audit(
            {
                "action": "revoke_key",
                "actor_key_id": actor_key_id,
                "key_id": key_id,
            }
        )
        return True

    def rotate_key(
        self, key_id: str, *, actor_key_id: Optional[str] = None
    ) -> Tuple[str, str] | None:
        key_id = str(key_id or "").strip()
        if not key_id:
            return None

        with self._lock:
            with self._file_lock():
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
                    "last_used_at": None,
                    "revoked_at": None,
                    "salt_hex": salt.hex(),
                    "hash_hex": hash_hex,
                    "iterations": PBKDF2_ITERATIONS,
                }

                self._save_records_unlocked(records)

        self._append_audit(
            {
                "action": "rotate_key",
                "actor_key_id": actor_key_id,
                "old_key_id": key_id,
                "key_id": new_id,
            }
        )
        return new_id, raw_key

    def mark_used(self, key_id: str) -> None:
        key_id = str(key_id or "").strip()
        if not key_id:
            return
        now = time.monotonic()

        with self._lock:
            previous = self._last_used_touch.get(key_id)
            if previous is not None and now - previous < LAST_USED_MIN_INTERVAL_SECONDS:
                return
            self._last_used_touch[key_id] = now

        try:
            with self._lock:
                with self._file_lock():
                    records = self._load_records_unlocked()
                    record = records.get(key_id)
                    if not record or record.get("revoked_at"):
                        return
                    record["last_used_at"] = _utc_now_iso()
                    records[key_id] = record
                    self._save_records_unlocked(records)
        except Exception:
            return

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
