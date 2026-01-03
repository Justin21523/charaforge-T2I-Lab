"""Async model scan job manager (Redis-backed with in-process worker fallback)."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

from core.train.registry import ModelScanCancelledError, get_model_registry

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore

logger = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "succeeded", "failed", "canceled"]


def _snapshot_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "job_id": record.get("job_id"),
        "owner": record.get("owner"),
        "status": record.get("status"),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "started_at": record.get("started_at"),
        "finished_at": record.get("finished_at"),
        "replace": bool(record.get("replace")),
        "cancel_requested": bool(record.get("cancel_requested")),
        "result": record.get("result"),
        "error": record.get("error"),
    }


@dataclass
class CancelResult:
    snapshot: Dict[str, Any]
    canceled: bool


class _MemoryBackend:
    def __init__(self, *, worker_enabled: bool, job_ttl_seconds: int):
        self._worker_enabled = bool(worker_enabled)
        self._job_ttl_seconds = int(job_ttl_seconds or 0)

        self._lock = threading.RLock()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._expires_at: Dict[str, float] = {}
        self._active_job_id: str | None = None
        self._cancel_events: Dict[str, threading.Event] = {}

        self._queue: "queue.Queue[str]" = queue.Queue()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None

    def start(self) -> None:
        if not self._worker_enabled:
            return
        if self._worker and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()

    def shutdown(self, timeout_s: float = 1.0) -> None:
        self._stop_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=timeout_s)

    def submit(self, *, owner: str, replace: bool) -> Tuple[str, bool]:
        self.start()

        with self._lock:
            active = self._active_job_id
            if active:
                existing = self._jobs.get(active)
                if existing and str(existing.get("status") or "") in {"queued", "running"}:
                    return active, False
                self._active_job_id = None

            job_id = str(uuid.uuid4())
            now = time.time()
            record: Dict[str, Any] = {
                "job_id": job_id,
                "owner": owner,
                "status": "queued",
                "created_at": now,
                "updated_at": now,
                "started_at": None,
                "finished_at": None,
                "replace": bool(replace),
                "cancel_requested": False,
                "result": None,
                "error": None,
            }
            self._jobs[job_id] = record
            self._active_job_id = job_id
            self._queue.put(job_id)
            return job_id, True

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            expiry = self._expires_at.get(job_id)
            if expiry is not None and time.time() >= expiry:
                self._expires_at.pop(job_id, None)
                self._jobs.pop(job_id, None)
                if self._active_job_id == job_id:
                    self._active_job_id = None
                return None

            record = self._jobs.get(job_id)
            return _snapshot_from_record(record) if record else None

    def get_owner(self, job_id: str) -> str | None:
        with self._lock:
            record = self._jobs.get(job_id)
            owner = record.get("owner") if record else None
        return str(owner) if owner else None

    def list_jobs(
        self, *, owner: str | None = None, limit: int = 50, status: str | None = None
    ) -> list[Dict[str, Any]]:
        limit = max(1, min(int(limit or 50), 200))
        now = time.time()

        with self._lock:
            expired = [job_id for job_id, expiry in self._expires_at.items() if now >= expiry]
            for job_id in expired:
                self._expires_at.pop(job_id, None)
                self._jobs.pop(job_id, None)
                self._cancel_events.pop(job_id, None)
                if self._active_job_id == job_id:
                    self._active_job_id = None

            records = list(self._jobs.values())

        if owner:
            records = [r for r in records if str(r.get("owner") or "") == owner]
        if status:
            records = [r for r in records if str(r.get("status") or "") == status]

        records.sort(key=lambda r: float(r.get("created_at") or 0.0), reverse=True)
        return [_snapshot_from_record(r) for r in records[:limit]]

    def delete_job(self, job_id: str) -> bool:
        job_id = str(job_id or "").strip()
        if not job_id:
            return False
        with self._lock:
            existed = job_id in self._jobs
            self._jobs.pop(job_id, None)
            self._expires_at.pop(job_id, None)
            self._cancel_events.pop(job_id, None)
            if self._active_job_id == job_id:
                self._active_job_id = None
            return existed

    def cleanup_jobs(
        self,
        *,
        owner: str | None = None,
        limit: int = 2000,
        dry_run: bool = True,
    ) -> Dict[str, int]:
        limit = max(1, min(int(limit or 2000), 10_000))
        now = time.time()

        with self._lock:
            expired = [
                job_id
                for job_id, expiry in self._expires_at.items()
                if expiry is not None and now >= float(expiry)
            ]

            if owner:
                expired = [
                    job_id
                    for job_id in expired
                    if str(self._jobs.get(job_id, {}).get("owner") or "") == owner
                ]

            expired.sort(key=lambda job_id: float(self._expires_at.get(job_id, 0.0)))
            expired = expired[:limit]

            if dry_run:
                return {"scanned": len(expired), "stale": len(expired), "removed": 0}

            for job_id in expired:
                self._expires_at.pop(job_id, None)
                self._jobs.pop(job_id, None)
                self._cancel_events.pop(job_id, None)
                if self._active_job_id == job_id:
                    self._active_job_id = None

        return {"scanned": len(expired), "stale": len(expired), "removed": len(expired)}

    def global_counts(self) -> Dict[str, int]:
        now = time.time()
        with self._lock:
            expired = [job_id for job_id, expiry in self._expires_at.items() if now >= expiry]
            for job_id in expired:
                self._expires_at.pop(job_id, None)
                self._jobs.pop(job_id, None)
                self._cancel_events.pop(job_id, None)
                if self._active_job_id == job_id:
                    self._active_job_id = None

            queued = 0
            running = 0
            for record in self._jobs.values():
                status = str(record.get("status") or "")
                if status == "queued":
                    queued += 1
                elif status == "running":
                    running += 1
            lease_active = 1 if bool(self._active_job_id) else 0

        return {"queued": queued, "running": running, "lease_active": lease_active}

    def cancel(self, job_id: str) -> Optional[CancelResult]:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None

            status = str(record.get("status") or "")
            record["cancel_requested"] = True
            record["updated_at"] = time.time()

            if status == "running":
                event = self._cancel_events.get(job_id)
                if event is not None:
                    event.set()
                return CancelResult(snapshot=_snapshot_from_record(record), canceled=False)

            if status != "queued":
                return CancelResult(snapshot=_snapshot_from_record(record), canceled=False)

            record["status"] = "canceled"
            record["finished_at"] = time.time()
            record["error"] = {"error": "CANCELED", "message": "Canceled"}
            if self._active_job_id == job_id:
                self._active_job_id = None
            self._apply_ttl_locked(job_id, record)
            return CancelResult(snapshot=_snapshot_from_record(record), canceled=True)

    def _apply_ttl_locked(self, job_id: str, record: Dict[str, Any]) -> None:
        if self._job_ttl_seconds <= 0:
            return
        status = str(record.get("status") or "")
        if status in {"succeeded", "failed", "canceled"}:
            self._expires_at[job_id] = time.time() + float(self._job_ttl_seconds)

    def _run_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                job_id = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue

            with self._lock:
                record = self._jobs.get(job_id)
                if record is None:
                    continue
                if str(record.get("status") or "") != "queued":
                    continue

                if bool(record.get("cancel_requested")):
                    record["status"] = "canceled"
                    record["finished_at"] = time.time()
                    record["updated_at"] = time.time()
                    record["error"] = {"error": "CANCELED", "message": "Canceled"}
                    self._apply_ttl_locked(job_id, record)
                    if self._active_job_id == job_id:
                        self._active_job_id = None
                    continue

                record["status"] = "running"
                record["started_at"] = time.time()
                record["updated_at"] = time.time()
                cancel_event = threading.Event()
                self._cancel_events[job_id] = cancel_event

            replace = bool(record.get("replace"))
            try:
                registry = get_model_registry()
                last_result: Dict[str, Any] | None = None
                for _ in range(20):
                    if self._stop_event.is_set():
                        raise RuntimeError("Worker stopped")
                    if cancel_event.is_set():
                        raise ModelScanCancelledError("Cancelled")
                    last_result = registry.scan_filesystem(
                        replace, cancel_check=cancel_event.is_set
                    )
                    if str(last_result.get("status") or "") != "busy":
                        break
                    time.sleep(0.25)
                result = last_result or {"status": "failed", "error": "NO_RESULT"}
                if str(result.get("status") or "") == "busy":
                    raise RuntimeError("Model scan still busy after retries")

                with self._lock:
                    record = self._jobs.get(job_id)
                    if record is None:
                        continue
                    record["status"] = "succeeded"
                    record["finished_at"] = time.time()
                    record["updated_at"] = time.time()
                    record["result"] = result
                    self._apply_ttl_locked(job_id, record)
                    if self._active_job_id == job_id:
                        self._active_job_id = None
            except ModelScanCancelledError:
                with self._lock:
                    record = self._jobs.get(job_id)
                    if record is None:
                        continue
                    record["status"] = "canceled"
                    record["finished_at"] = time.time()
                    record["updated_at"] = time.time()
                    record["cancel_requested"] = True
                    record["error"] = {"error": "CANCELED", "message": "Canceled"}
                    self._apply_ttl_locked(job_id, record)
                    if self._active_job_id == job_id:
                        self._active_job_id = None
            except Exception as exc:
                with self._lock:
                    record = self._jobs.get(job_id)
                    if record is None:
                        continue
                    record["status"] = "failed"
                    record["finished_at"] = time.time()
                    record["updated_at"] = time.time()
                    record["error"] = {"error": "FAILED", "message": str(exc)}
                    self._apply_ttl_locked(job_id, record)
                    if self._active_job_id == job_id:
                        self._active_job_id = None
            finally:
                with self._lock:
                    self._cancel_events.pop(job_id, None)


class _RedisBackend:
    def __init__(self, redis_url: str, *, worker_enabled: bool, job_ttl_seconds: int):
        self._redis_url = redis_url
        self._worker_enabled = bool(worker_enabled)
        self._job_ttl_seconds = int(job_ttl_seconds or 0)

        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None

        self._client = redis.from_url(  # type: ignore[union-attr]
            redis_url,
            decode_responses=True,
            socket_connect_timeout=0.2,
            socket_timeout=0.2,
        )

        self._namespace = "charaforge:model_scan"

    def _active_ttl_seconds(self) -> int:
        if self._job_ttl_seconds > 0:
            return max(self._job_ttl_seconds, 3600)
        return 3600

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception:
            return False

    def start(self) -> None:
        if not self._worker_enabled:
            return
        if self._worker and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()

    def shutdown(self, timeout_s: float = 1.0) -> None:
        self._stop_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=timeout_s)

    def submit(self, *, owner: str, replace: bool) -> Tuple[str, bool]:
        self.start()

        job_id = str(uuid.uuid4())
        active_ttl = self._active_ttl_seconds()
        try:
            claimed = self._client.set(self._active_key(), job_id, ex=active_ttl, nx=True)
        except Exception as exc:
            raise RuntimeError(f"Redis error: {exc}") from exc

        if not claimed:
            existing = self._client.get(self._active_key())
            return str(existing or ""), False

        now = time.time()
        record: Dict[str, Any] = {
            "job_id": job_id,
            "owner": owner,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
            "replace": bool(replace),
            "cancel_requested": False,
            "result": None,
            "error": None,
        }
        payload = json.dumps(record, ensure_ascii=False)

        pipe = self._client.pipeline()
        if self._job_ttl_seconds > 0:
            pipe.set(self._job_key(job_id), payload, ex=active_ttl)
        else:
            pipe.set(self._job_key(job_id), payload)
        pipe.rpush(self._queue_key(), job_id)
        pipe.zadd(self._jobs_index_key(), {job_id: now})
        pipe.zadd(self._owner_jobs_index_key(owner), {job_id: now})
        pipe.execute()

        return job_id, True

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        record = self._load_job(job_id)
        return _snapshot_from_record(record) if record else None

    def get_owner(self, job_id: str) -> str | None:
        record = self._load_job(job_id)
        owner = record.get("owner") if record else None
        return str(owner) if owner else None

    def list_jobs(
        self, *, owner: str | None = None, limit: int = 50, status: str | None = None
    ) -> list[Dict[str, Any]]:
        limit = max(1, min(int(limit or 50), 200))
        key = self._owner_jobs_index_key(owner) if owner else self._jobs_index_key()

        fetch = min(1000, max(limit * 5, limit))
        try:
            job_ids = [str(v) for v in (self._client.zrevrange(key, 0, fetch - 1) or [])]
        except Exception:
            job_ids = []

        out: list[Dict[str, Any]] = []
        stale: list[str] = []
        for job_id in job_ids:
            record = self._load_job(job_id)
            if record is None:
                stale.append(job_id)
                continue
            snap = _snapshot_from_record(record)
            if status and str(snap.get("status") or "") != status:
                continue
            out.append(snap)
            if len(out) >= limit:
                break

        if stale:
            try:
                self._client.zrem(key, *stale)
            except Exception:
                pass

        return out

    def delete_job(self, job_id: str) -> bool:
        job_id = str(job_id or "").strip()
        if not job_id:
            return False

        record = self._load_job(job_id) or {}
        owner = str(record.get("owner") or "")

        try:
            pipe = self._client.pipeline()
            pipe.delete(self._job_key(job_id))
            pipe.delete(self._cancel_key(job_id))
            pipe.lrem(self._queue_key(), 0, job_id)
            pipe.zrem(self._jobs_index_key(), job_id)
            if owner:
                pipe.zrem(self._owner_jobs_index_key(owner), job_id)
            pipe.execute()
        except Exception:
            return False

        self._release_active_lease(job_id)
        return True

    def cleanup_jobs(
        self,
        *,
        owner: str | None = None,
        limit: int = 2000,
        dry_run: bool = True,
    ) -> Dict[str, int]:
        limit = max(1, min(int(limit or 2000), 10_000))
        key = self._owner_jobs_index_key(owner) if owner else self._jobs_index_key()

        try:
            job_ids = [str(v) for v in (self._client.zrange(key, 0, limit - 1) or [])]
        except Exception:
            job_ids = []

        stale: list[str] = []
        try:
            pipe = self._client.pipeline()
            for job_id in job_ids:
                pipe.exists(self._job_key(job_id))
            exists_values = pipe.execute() if job_ids else []
            for job_id, exists in zip(job_ids, exists_values):
                if not exists:
                    stale.append(job_id)
        except Exception:
            stale = []

        removed = 0
        if stale and not dry_run:
            try:
                removed = int(self._client.zrem(key, *stale) or 0)
            except Exception:
                removed = 0

        return {"scanned": len(job_ids), "stale": len(stale), "removed": removed}

    def global_counts(self) -> Dict[str, int]:
        try:
            active_job_id = str(self._client.get(self._active_key()) or "")
        except Exception:
            active_job_id = ""

        lease_active = 1 if active_job_id else 0
        queued = 0
        running = 0

        if active_job_id:
            record = self._load_job(active_job_id)
            if record is not None:
                status = str(record.get("status") or "")
                if status == "queued":
                    queued = 1
                elif status == "running":
                    running = 1
            else:
                try:
                    queued = int(self._client.llen(self._queue_key()) or 0)
                except Exception:
                    queued = 0
                running = 0 if queued else 1

        return {"queued": queued, "running": running, "lease_active": lease_active}

    def cancel(self, job_id: str) -> Optional[CancelResult]:
        record = self._load_job(job_id)
        if record is None:
            return None

        status = str(record.get("status") or "")
        record["cancel_requested"] = True
        record["updated_at"] = time.time()
        try:
            self._client.set(self._cancel_key(job_id), "1", ex=self._active_ttl_seconds())
        except Exception:
            pass

        if status != "queued":
            self._save_job(job_id, record)
            return CancelResult(snapshot=_snapshot_from_record(record), canceled=False)

        record["status"] = "canceled"
        record["finished_at"] = time.time()
        record["error"] = {"error": "CANCELED", "message": "Canceled"}
        pipe = self._client.pipeline()
        pipe.lrem(self._queue_key(), 0, job_id)
        pipe.execute()
        self._release_active_lease(job_id)
        self._save_job(job_id, record)
        return CancelResult(snapshot=_snapshot_from_record(record), canceled=True)

    def _active_key(self) -> str:
        return f"{self._namespace}:active"

    def _cancel_key(self, job_id: str) -> str:
        return f"{self._namespace}:cancel:{job_id}"

    def _queue_key(self) -> str:
        return f"{self._namespace}:queue"

    def _job_key(self, job_id: str) -> str:
        return f"{self._namespace}:job:{job_id}"

    def _jobs_index_key(self) -> str:
        return f"{self._namespace}:jobs"

    def _owner_jobs_index_key(self, owner: str | None) -> str:
        return f"{self._namespace}:owner:{owner}:jobs"

    def _release_active_lease(self, job_id: str) -> None:
        try:
            self._client.eval(
                "if redis.call('get', KEYS[1]) == ARGV[1] then "
                "return redis.call('del', KEYS[1]) else return 0 end",
                1,
                self._active_key(),
                job_id,
            )
        except Exception:
            return

    def _renew_active_lease(self, job_id: str, ttl_seconds: int) -> None:
        ttl_seconds = int(ttl_seconds or 0)
        if ttl_seconds <= 0:
            return
        try:
            self._client.eval(
                "if redis.call('get', KEYS[1]) == ARGV[1] then "
                "return redis.call('expire', KEYS[1], ARGV[2]) else return 0 end",
                1,
                self._active_key(),
                job_id,
                ttl_seconds,
            )
        except Exception:
            return

    def _renew_job_lease(self, job_id: str, ttl_seconds: int) -> None:
        ttl_seconds = int(ttl_seconds or 0)
        if ttl_seconds <= 0:
            return
        try:
            self._client.expire(self._job_key(job_id), ttl_seconds)
        except Exception:
            return

    def _renew_cancel_lease(self, job_id: str, ttl_seconds: int) -> None:
        ttl_seconds = int(ttl_seconds or 0)
        if ttl_seconds <= 0:
            return
        try:
            self._client.expire(self._cancel_key(job_id), ttl_seconds)
        except Exception:
            return

    def _is_cancel_requested(self, job_id: str) -> bool:
        try:
            return bool(self._client.exists(self._cancel_key(job_id)))
        except Exception:
            return False

    def _load_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        try:
            raw = self._client.get(self._job_key(job_id))
            if not raw:
                return None
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
            return None
        except Exception:
            return None

    def _save_job(self, job_id: str, record: Dict[str, Any]) -> None:
        payload = json.dumps(record, ensure_ascii=False)
        status = str(record.get("status") or "")
        terminal = status in {"succeeded", "failed", "canceled"}

        ttl: int | None = None
        if self._job_ttl_seconds > 0:
            ttl = int(self._job_ttl_seconds) if terminal else int(self._active_ttl_seconds())

        try:
            if ttl is not None:
                self._client.set(self._job_key(job_id), payload, ex=ttl)
            else:
                self._client.set(self._job_key(job_id), payload)
        except Exception:
            return

    def _run_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._client.blpop(self._queue_key(), timeout=1)
            except Exception:
                time.sleep(0.25)
                continue

            if not item:
                continue

            _, job_id = item
            record = self._load_job(job_id)
            if record is None:
                continue

            if str(record.get("status") or "") != "queued":
                continue

            if bool(record.get("cancel_requested")) or self._is_cancel_requested(job_id):
                record["status"] = "canceled"
                record["finished_at"] = time.time()
                record["updated_at"] = time.time()
                record["error"] = {"error": "CANCELED", "message": "Canceled"}
                self._save_job(job_id, record)
                self._release_active_lease(job_id)
                continue

            record["status"] = "running"
            record["started_at"] = time.time()
            record["updated_at"] = time.time()
            self._save_job(job_id, record)

            replace = bool(record.get("replace"))
            try:
                registry = get_model_registry()
                last_result: Dict[str, Any] | None = None
                active_ttl = self._active_ttl_seconds()
                last_cancel_check = 0.0
                cancel_cached = False
                last_lease_renew = 0.0

                def _cancel_check() -> bool:
                    nonlocal cancel_cached
                    nonlocal last_cancel_check
                    if cancel_cached:
                        return True
                    now = time.monotonic()
                    if now - last_cancel_check < 0.25:
                        return False
                    last_cancel_check = now
                    cancel_cached = self._is_cancel_requested(job_id)
                    return cancel_cached

                def _heartbeat() -> None:
                    nonlocal last_lease_renew
                    now = time.monotonic()
                    if now - last_lease_renew < 10.0:
                        return
                    last_lease_renew = now
                    self._renew_active_lease(job_id, active_ttl)
                    self._renew_job_lease(job_id, active_ttl)
                    self._renew_cancel_lease(job_id, active_ttl)

                for _ in range(20):
                    if self._stop_event.is_set():
                        raise RuntimeError("Worker stopped")
                    if _cancel_check():
                        raise ModelScanCancelledError("Cancelled")
                    _heartbeat()
                    last_result = registry.scan_filesystem(
                        replace, cancel_check=_cancel_check, heartbeat=_heartbeat
                    )
                    if str(last_result.get("status") or "") != "busy":
                        break
                    time.sleep(0.25)
                result = last_result or {"status": "failed", "error": "NO_RESULT"}
                if str(result.get("status") or "") == "busy":
                    raise RuntimeError("Model scan still busy after retries")

                record = self._load_job(job_id) or record
                record["status"] = "succeeded"
                record["finished_at"] = time.time()
                record["updated_at"] = time.time()
                record["result"] = result
                self._save_job(job_id, record)
                try:
                    self._client.delete(self._cancel_key(job_id))
                except Exception:
                    pass
                self._release_active_lease(job_id)
            except ModelScanCancelledError:
                record = self._load_job(job_id) or record
                record["status"] = "canceled"
                record["finished_at"] = time.time()
                record["updated_at"] = time.time()
                record["cancel_requested"] = True
                record["error"] = {"error": "CANCELED", "message": "Canceled"}
                self._save_job(job_id, record)
                try:
                    self._client.delete(self._cancel_key(job_id))
                except Exception:
                    pass
                self._release_active_lease(job_id)
            except Exception as exc:
                record = self._load_job(job_id) or record
                record["status"] = "failed"
                record["finished_at"] = time.time()
                record["updated_at"] = time.time()
                record["error"] = {"error": "FAILED", "message": str(exc)}
                self._save_job(job_id, record)
                try:
                    self._client.delete(self._cancel_key(job_id))
                except Exception:
                    pass
                self._release_active_lease(job_id)


class ModelScanJobManager:
    def __init__(
        self,
        *,
        redis_url: str | None = None,
        worker_enabled: bool = True,
        job_ttl_seconds: int = 3600,
    ):
        self._backend: _MemoryBackend | _RedisBackend

        if redis_url and str(redis_url).startswith("redis") and redis is not None:
            try:
                candidate = _RedisBackend(
                    redis_url,
                    worker_enabled=worker_enabled,
                    job_ttl_seconds=job_ttl_seconds,
                )
                if candidate.ping():
                    self._backend = candidate
                    self._backend.start()
                    return
                logger.warning("Redis unavailable; falling back to in-memory model scan jobs")
            except Exception as exc:
                logger.warning(
                    "Redis unavailable; falling back to in-memory model scan jobs: %s",
                    exc,
                )

        self._backend = _MemoryBackend(
            worker_enabled=worker_enabled,
            job_ttl_seconds=job_ttl_seconds,
        )
        if worker_enabled:
            self._backend.start()

    def shutdown(self, timeout_s: float = 1.0) -> None:
        self._backend.shutdown(timeout_s=timeout_s)

    def submit(self, *, owner: str, replace: bool) -> Tuple[str, bool]:
        return self._backend.submit(owner=owner, replace=replace)

    def get(self, job_id: str) -> Dict[str, Any] | None:
        return self._backend.get(job_id)

    def get_owner(self, job_id: str) -> str | None:
        return self._backend.get_owner(job_id)

    def list_jobs(
        self, *, owner: str | None = None, limit: int = 50, status: str | None = None
    ) -> list[Dict[str, Any]]:
        return self._backend.list_jobs(owner=owner, limit=limit, status=status)

    def delete_job(self, job_id: str) -> bool:
        return self._backend.delete_job(job_id)

    def cleanup_jobs(
        self,
        *,
        owner: str | None = None,
        limit: int = 2000,
        dry_run: bool = True,
    ) -> Dict[str, int]:
        return self._backend.cleanup_jobs(owner=owner, limit=limit, dry_run=dry_run)

    def global_counts(self) -> Dict[str, int]:
        return self._backend.global_counts()

    def cancel(self, job_id: str) -> CancelResult | None:
        return self._backend.cancel(job_id)
