"""Refresh token storage (Redis-backed with file fallback).

Refresh tokens are opaque secrets. We store only a SHA-256 hash on disk/Redis.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

from core.config import get_cache_paths

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore

logger = logging.getLogger(__name__)

REFRESH_PREFIX = "cfr"
LOCK_STALE_SECONDS = 120.0
LOCK_TIMEOUT_SECONDS = 5.0
LOCK_POLL_SECONDS = 0.05


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _token_hash(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def _refresh_path() -> Path:
    cache_paths = get_cache_paths()
    return cache_paths.cache / "auth" / "refresh_tokens.json"


class RefreshTokenStore:
    def __init__(self, *, path: Optional[Path] = None, redis_url: Optional[str] = None):
        self._path = path or _refresh_path()
        self._redis_url = str(redis_url or "").strip() or None
        self._redis_client = None

    @classmethod
    def default(cls, *, redis_url: Optional[str] = None) -> "RefreshTokenStore":
        return cls(redis_url=redis_url)

    def _redis(self):
        if redis is None:
            return None
        if not self._redis_url:
            return None
        if self._redis_client is None:
            try:
                self._redis_client = redis.Redis.from_url(  # type: ignore[attr-defined]
                    self._redis_url,
                    decode_responses=True,
                    socket_connect_timeout=0.2,
                    socket_timeout=0.2,
                    retry_on_timeout=False,
                )
            except Exception:
                self._redis_client = None
                return None
        return self._redis_client

    def _redis_key(self, token_hash: str) -> str:
        return f"charaforge:auth:refresh:{token_hash}"

    def _redis_subject_key(self, subject: str) -> str:
        return f"charaforge:auth:refresh:sub:{subject}"

    def _redis_key_id_key(self, key_id: str) -> str:
        return f"charaforge:auth:refresh:kid:{key_id}"

    def issue(
        self,
        *,
        subject: str,
        role: str,
        scopes: Set[str],
        key_id: str | None,
        ttl_seconds: int,
        now: Optional[float] = None,
    ) -> Tuple[str, int]:
        ttl_seconds = int(ttl_seconds or 0)
        if ttl_seconds <= 0:
            ttl_seconds = 30 * 24 * 3600

        issued_at = int(float(now if now is not None else time.time()))
        expires_at = int(issued_at + ttl_seconds)

        token_id = uuid.uuid4().hex[:12]
        secret = secrets.token_urlsafe(32)
        raw_token = f"{REFRESH_PREFIX}_{token_id}.{secret}"
        token_hash = _token_hash(raw_token)

        record: Dict[str, Any] = {
            "token_id": token_id,
            "token_hash": token_hash,
            "subject": str(subject),
            "role": str(role or "user"),
            "scopes": sorted({s for s in (scopes or set()) if str(s).strip()}),
            "key_id": str(key_id) if key_id else None,
            "created_at": _utc_now_iso(),
            "last_used_at": None,
            "revoked_at": None,
            "expires_at": expires_at,
        }

        client = self._redis()
        if client is not None:
            try:
                key = self._redis_key(token_hash)
                subject_key = self._redis_subject_key(str(subject))
                key_id_value = str(key_id) if key_id else ""
                key_id_key = self._redis_key_id_key(key_id_value) if key_id_value else ""
                pipe = client.pipeline()
                pipe.set(key, json.dumps(record, ensure_ascii=False), ex=ttl_seconds)
                pipe.sadd(subject_key, token_hash)
                pipe.expire(subject_key, ttl_seconds + 60)
                if key_id_key:
                    pipe.sadd(key_id_key, token_hash)
                    pipe.expire(key_id_key, ttl_seconds + 60)
                pipe.execute()
                return raw_token, expires_at
            except Exception:
                logger.debug("Refresh token Redis store unavailable; falling back to file")

        self._issue_file(token_hash=token_hash, record=record)
        return raw_token, expires_at

    def verify(self, raw_token: str, *, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        raw_token = str(raw_token or "").strip()
        if not raw_token or "." not in raw_token:
            return None
        token_hash = _token_hash(raw_token)
        current = float(now if now is not None else time.time())

        client = self._redis()
        if client is not None:
            try:
                raw = client.get(self._redis_key(token_hash))
                if not raw:
                    return None
                data = json.loads(raw)
                if not isinstance(data, dict):
                    return None
                if data.get("revoked_at"):
                    return None
                exp = data.get("expires_at")
                if exp is None:
                    return None
                if current > float(exp):
                    return None
                try:
                    client.set(
                        self._redis_key(token_hash),
                        json.dumps(
                            {
                                **data,
                                "last_used_at": _utc_now_iso(),
                            },
                            ensure_ascii=False,
                        ),
                        ex=max(60, int(float(exp) - current)),
                    )
                except Exception:
                    pass
                return data
            except Exception:
                logger.debug("Refresh token Redis verify failed; falling back to file")

        return self._verify_file(token_hash=token_hash, now=current)

    def revoke(self, raw_token: str) -> bool:
        raw_token = str(raw_token or "").strip()
        if not raw_token:
            return False
        token_hash = _token_hash(raw_token)

        client = self._redis()
        if client is not None:
            try:
                raw = client.get(self._redis_key(token_hash))
                if raw:
                    try:
                        data = json.loads(raw)
                        subject = str(data.get("subject") or "")
                        key_id = str(data.get("key_id") or "")
                    except Exception:
                        subject = ""
                        key_id = ""
                    pipe = client.pipeline()
                    pipe.delete(self._redis_key(token_hash))
                    if subject:
                        pipe.srem(self._redis_subject_key(subject), token_hash)
                    if key_id:
                        pipe.srem(self._redis_key_id_key(key_id), token_hash)
                    pipe.execute()
                else:
                    client.delete(self._redis_key(token_hash))
                return True
            except Exception:
                logger.debug("Refresh token Redis revoke failed; falling back to file")

        return self._revoke_file(token_hash=token_hash)

    def revoke_subject(self, subject: str) -> int:
        subject = str(subject or "").strip()
        if not subject:
            return 0

        client = self._redis()
        if client is not None:
            try:
                subject_key = self._redis_subject_key(subject)
                hashes = list(client.smembers(subject_key) or [])
                pipe = client.pipeline()
                for token_hash in hashes:
                    pipe.delete(self._redis_key(str(token_hash)))
                pipe.delete(subject_key)
                pipe.execute()
                return len(hashes)
            except Exception:
                logger.debug("Refresh token Redis revoke_subject failed; falling back to file")

        return self._revoke_subject_file(subject)

    def revoke_key_id(self, key_id: str) -> int:
        key_id = str(key_id or "").strip()
        if not key_id:
            return 0

        client = self._redis()
        if client is not None:
            try:
                key_id_key = self._redis_key_id_key(key_id)
                hashes = list(client.smembers(key_id_key) or [])
                pipe = client.pipeline()
                for token_hash in hashes:
                    pipe.delete(self._redis_key(str(token_hash)))
                pipe.delete(key_id_key)
                pipe.execute()
                return len(hashes)
            except Exception:
                logger.debug("Refresh token Redis revoke_key_id failed; falling back to file")

        return self._revoke_key_id_file(key_id)

    def _lock_path(self) -> Path:
        return self._path.with_suffix(self._path.suffix + ".lock")

    @contextmanager
    def _file_lock(self):
        lock_path = self._lock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS

        acquired = False
        while time.monotonic() < deadline:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
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
                raise RuntimeError(f"Failed to acquire refresh token lock: {exc}") from exc
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
            raise TimeoutError("Timed out acquiring refresh token lock")

        try:
            yield
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass

    def _load_file_unlocked(self) -> Dict[str, Dict[str, Any]]:
        path = self._path
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except Exception:
            return {}

        try:
            data = json.loads(raw)
        except Exception:
            return {}
        tokens = data.get("tokens") if isinstance(data, dict) else None
        if not isinstance(tokens, list):
            return {}

        out: Dict[str, Dict[str, Any]] = {}
        for item in tokens:
            if not isinstance(item, dict):
                continue
            token_hash = str(item.get("token_hash") or "").strip()
            if not token_hash:
                continue
            out[token_hash] = dict(item)
        return out

    def _save_file_unlocked(self, records: Dict[str, Dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "tokens": list(records.values())}

        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path)

    def _prune_expired(self, records: Dict[str, Dict[str, Any]], *, now: float) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for token_hash, record in records.items():
            try:
                exp = float(record.get("expires_at") or 0.0)
            except Exception:
                continue
            if exp and now > exp:
                continue
            if record.get("revoked_at"):
                continue
            out[token_hash] = record
        return out

    def _issue_file(self, *, token_hash: str, record: Dict[str, Any]) -> None:
        with self._file_lock():
            now = time.time()
            records = self._load_file_unlocked()
            records = self._prune_expired(records, now=now)
            records[token_hash] = record
            self._save_file_unlocked(records)

    def _verify_file(self, *, token_hash: str, now: float) -> Optional[Dict[str, Any]]:
        with self._file_lock():
            records = self._load_file_unlocked()
            records = self._prune_expired(records, now=now)
            record = records.get(token_hash)
            if not record:
                self._save_file_unlocked(records)
                return None
            record["last_used_at"] = _utc_now_iso()
            records[token_hash] = record
            self._save_file_unlocked(records)
            return dict(record)

    def _revoke_file(self, *, token_hash: str) -> bool:
        with self._file_lock():
            now = time.time()
            records = self._load_file_unlocked()
            records = self._prune_expired(records, now=now)
            if token_hash in records:
                records.pop(token_hash, None)
                self._save_file_unlocked(records)
                return True
            self._save_file_unlocked(records)
            return False

    def _revoke_subject_file(self, subject: str) -> int:
        with self._file_lock():
            now = time.time()
            records = self._load_file_unlocked()
            records = self._prune_expired(records, now=now)
            deleted = 0
            for token_hash in list(records.keys()):
                record = records.get(token_hash) or {}
                if str(record.get("subject") or "") == subject:
                    records.pop(token_hash, None)
                    deleted += 1
            self._save_file_unlocked(records)
            return deleted

    def _revoke_key_id_file(self, key_id: str) -> int:
        with self._file_lock():
            now = time.time()
            records = self._load_file_unlocked()
            records = self._prune_expired(records, now=now)
            deleted = 0
            for token_hash in list(records.keys()):
                record = records.get(token_hash) or {}
                if str(record.get("key_id") or "") == key_id:
                    records.pop(token_hash, None)
                    deleted += 1
            self._save_file_unlocked(records)
            return deleted
