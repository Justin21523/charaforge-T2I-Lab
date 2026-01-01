"""Async T2I job manager (Redis-backed with in-process worker fallback)."""

from __future__ import annotations

import json
import logging
import queue
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from api.schemas.t2i import GenerateRequest
from core.config import get_app_paths, get_settings
from core.t2i.pipeline import PIPELINE_LOCK, GenerationParams, get_pipeline_manager

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore

logger = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "succeeded", "failed", "canceled"]
ACCESS_META_FILENAME = "_access.json"


class GenerationCancelledError(RuntimeError):
    pass


def job_dir(job_id: str) -> Path:
    app_paths = get_app_paths()
    return app_paths.outputs / "t2i" / job_id


def access_meta_path(job_id: str) -> Path:
    return job_dir(job_id) / ACCESS_META_FILENAME


def write_access_meta(job_id: str, *, owner: str) -> None:
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


def read_access_owner(job_id: str) -> str | None:
    path = access_meta_path(job_id)
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    owner = data.get("owner")
    return str(owner) if owner else None


def _pick_model_id(req: GenerateRequest) -> str:
    settings = get_settings()
    model_type = (req.model_type or "").lower()
    if model_type not in {"sd15", "sdxl"}:
        model_type = "sdxl"

    if req.model:
        return req.model

    return (
        settings.model.default_sdxl_model
        if model_type == "sdxl"
        else settings.model.default_sd15_model
    )


def _sampler_to_scheduler(sampler: str) -> Tuple[str, Dict[str, Any]]:
    name = (sampler or "").strip().lower()

    if name in {"ddim"}:
        return "DDIM", {}
    if name in {"lms"}:
        return "LMS", {}
    if name in {"euler a", "euler_a", "euler-ancestral"}:
        return "EulerAncestral", {}
    if name in {"euler", "heun"}:
        return "EulerAncestral", {}
    if name.startswith("dpm++"):
        kwargs: Dict[str, Any] = {"use_karras_sigmas": "karras" in name}
        if "sde" in name:
            kwargs["algorithm_type"] = "sde-dpmsolver++"
        return "DPMSolverMultistep", kwargs

    return "DPMSolverMultistep", {}


def validate_generate_request(req: GenerateRequest) -> None:
    settings = get_settings()

    if req.width % 8 != 0 or req.height % 8 != 0:
        raise ValueError("Width/height must be multiples of 8")

    if req.batch_size > settings.api.max_batch_size:
        raise ValueError(f"batch_size exceeds limit ({settings.api.max_batch_size})")
    if req.steps > settings.api.max_steps:
        raise ValueError(f"steps exceeds limit ({settings.api.max_steps})")


def run_generate_sync(
    req: GenerateRequest,
    job_id: str,
    *,
    cancel_event: threading.Event | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> Dict[str, Any]:
    validate_generate_request(req)

    seed = req.seed if req.seed >= 0 else secrets.randbelow(2**32)
    model_id = _pick_model_id(req)
    scheduler_name, scheduler_kwargs = _sampler_to_scheduler(req.sampler)

    output_dir = job_dir(job_id)

    def _callback_on_step_end(step: int, timestep: int, callback_kwargs: dict) -> None:
        if on_progress:
            on_progress(step + 1, req.steps)
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelledError("Cancelled")

    manager = get_pipeline_manager()

    with PIPELINE_LOCK:
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelledError("Cancelled")

        if not manager.pipeline_loaded or manager.current_model != model_id:
            if not manager.load_model(model_id):
                raise RuntimeError(f"Failed to load model: {model_id}")

        manager.set_scheduler(scheduler_name, **scheduler_kwargs)

        if req.loras:
            lora_configs = []
            for lora in req.loras:
                if not isinstance(lora, dict) or "name" not in lora:
                    continue
                lora_configs.append(
                    {
                        "name": str(lora["name"]),
                        "scale": float(lora.get("scale", 1.0)),
                        "enabled": bool(lora.get("enabled", True)),
                    }
                )
            manager.load_lora_stack(lora_configs)
        else:
            manager.unload_loras()

        params = GenerationParams(
            prompt=req.prompt,
            negative_prompt=req.negative or None,
            width=req.width,
            height=req.height,
            num_inference_steps=req.steps,
            guidance_scale=req.cfg_scale,
            num_images_per_prompt=req.batch_size,
            seed=seed,
            clip_skip=req.clip_skip,
        )

        result = manager.generate(params, callback_on_step_end=_callback_on_step_end)
        elapsed_ms = int(result.generation_time * 1000)

        saved = manager.save_generation_result(result, output_dir)

    return {
        "seed": seed,
        "elapsed_ms": elapsed_ms,
        "metadata": result.metadata,
        "saved_paths": [p.name for p in saved],
    }


def _snapshot_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    progress = None
    step = record.get("progress_step")
    total = record.get("progress_total")
    if step is not None and total is not None:
        progress = {"step": int(step), "total": int(total)}

    return {
        "job_id": record.get("job_id"),
        "owner": record.get("owner"),
        "status": record.get("status"),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "started_at": record.get("started_at"),
        "finished_at": record.get("finished_at"),
        "attempts": record.get("attempts"),
        "cancel_requested": bool(record.get("cancel_requested")),
        "progress": progress,
        "seed": record.get("seed"),
        "elapsed_ms": record.get("elapsed_ms"),
        "metadata": record.get("metadata") or {},
        "saved_paths": record.get("saved_paths") or [],
        "error": record.get("error"),
    }


@dataclass
class _MemoryJob:
    job_id: str
    owner: str
    request: GenerateRequest
    status: JobStatus = "queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    attempts: int = 0
    cancel_requested: bool = False
    progress_step: Optional[int] = None
    progress_total: Optional[int] = None
    seed: Optional[int] = None
    elapsed_ms: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    saved_paths: List[str] = field(default_factory=list)
    error: Optional[Dict[str, Any]] = None
    _cancel_event: Optional[threading.Event] = field(default=None, repr=False)

    def to_record(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "owner": self.owner,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "attempts": self.attempts,
            "cancel_requested": self.cancel_requested,
            "progress_step": self.progress_step,
            "progress_total": self.progress_total,
            "seed": self.seed,
            "elapsed_ms": self.elapsed_ms,
            "metadata": dict(self.metadata or {}),
            "saved_paths": list(self.saved_paths),
            "error": dict(self.error) if self.error else None,
        }


class _MemoryBackend:
    def __init__(
        self,
        *,
        max_jobs: int = 200,
        worker_enabled: bool = True,
        max_global_concurrent: int = 0,
    ):
        self._max_jobs = max_jobs
        self._worker_enabled = bool(worker_enabled)
        self._max_global_concurrent = int(max_global_concurrent)
        self._lock = threading.Lock()
        self._jobs: Dict[str, _MemoryJob] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self._worker_enabled:
            return
        if self._worker and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="t2i-job-worker-memory",
            daemon=True,
        )
        self._worker.start()

    def shutdown(self, timeout_s: float = 1.0) -> None:
        self._stop_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=timeout_s)

    def submit(self, req: GenerateRequest, *, owner: str) -> str:
        self.start()

        job_id = str(uuid.uuid4())
        job = _MemoryJob(job_id=job_id, owner=owner, request=req, status="queued")
        with self._lock:
            self._jobs[job_id] = job
            self._prune_locked()
        self._queue.put(job_id)
        return job_id

    def cancel(self, job_id: str) -> Dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None

            job.cancel_requested = True
            job.updated_at = time.time()
            if job.status == "queued":
                job.status = "canceled"
                job.finished_at = time.time()
                job.error = {"error": "CANCELED", "message": "Canceled"}
            elif job.status == "running":
                if job._cancel_event is not None:
                    job._cancel_event.set()
            return _snapshot_from_record(job.to_record())

    def get(self, job_id: str) -> Dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return _snapshot_from_record(job.to_record()) if job else None

    def get_owner(self, job_id: str) -> str | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return str(job.owner) if job else None

    def global_counts(self) -> Dict[str, int]:
        with self._lock:
            queued = 0
            running = 0
            for job in self._jobs.values():
                if job.status == "queued":
                    queued += 1
                elif job.status == "running":
                    running += 1
            return {"queued": queued, "running": running}

    def list_jobs(
        self, *, owner: str | None = None, limit: int = 50, status: str | None = None
    ) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit or 50), 200))
        with self._lock:
            jobs = list(self._jobs.values())
        if owner:
            jobs = [j for j in jobs if j.owner == owner]
        if status:
            jobs = [j for j in jobs if j.status == status]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return [_snapshot_from_record(j.to_record()) for j in jobs[:limit]]

    def delete_job(self, job_id: str) -> bool:
        job_id = str(job_id or "").strip()
        if not job_id:
            return False
        with self._lock:
            existed = job_id in self._jobs
            self._jobs.pop(job_id, None)
            return existed

    def owner_counts(self, owner: str) -> Dict[str, int]:
        with self._lock:
            queued = 0
            running = 0
            for job in self._jobs.values():
                if job.owner != owner:
                    continue
                if job.status == "queued":
                    queued += 1
                elif job.status == "running":
                    running += 1
            return {"queued": queued, "running": running}

    def _prune_locked(self) -> None:
        if len(self._jobs) <= self._max_jobs:
            return
        ordered = sorted(self._jobs.values(), key=lambda j: j.created_at)
        to_remove = ordered[: max(0, len(ordered) - self._max_jobs)]
        for job in to_remove:
            if job.status in {"queued", "running"}:
                continue
            self._jobs.pop(job.job_id, None)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job_id = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            with self._lock:
                job = self._jobs.get(job_id)
                if not job or job.status != "queued":
                    continue

                job.status = "running"
                job.started_at = time.time()
                job.updated_at = time.time()
                job.progress_step = 0
                job.progress_total = job.request.steps
                job._cancel_event = threading.Event()
                cancel_event = job._cancel_event

            def _on_progress(step: int, total: int) -> None:
                with self._lock:
                    current = self._jobs.get(job_id)
                    if not current or current.status != "running":
                        return
                    current.progress_step = step
                    current.progress_total = total
                    current.updated_at = time.time()

            try:
                result = run_generate_sync(
                    job.request,
                    job_id,
                    cancel_event=cancel_event,
                    on_progress=_on_progress,
                )
            except GenerationCancelledError:
                with self._lock:
                    current = self._jobs.get(job_id)
                    if current:
                        current.status = "canceled"
                        current.finished_at = time.time()
                        current.updated_at = time.time()
                        current.error = {"error": "CANCELED", "message": "Canceled"}
                        current._cancel_event = None
            except Exception as exc:
                with self._lock:
                    current = self._jobs.get(job_id)
                    if current:
                        current.status = "failed"
                        current.finished_at = time.time()
                        current.updated_at = time.time()
                        current.error = {"error": "GENERATION_FAILED", "message": str(exc)}
                        current._cancel_event = None
            else:
                with self._lock:
                    current = self._jobs.get(job_id)
                    if current:
                        current.status = "succeeded"
                        current.finished_at = time.time()
                        current.updated_at = time.time()
                        current.seed = int(result.get("seed", 0))
                        current.elapsed_ms = int(result.get("elapsed_ms", 0))
                        current.metadata = result.get("metadata") or {}
                        current.saved_paths = list(result.get("saved_paths") or [])
                        current.error = None
                        current._cancel_event = None


class _RedisBackend:
    def __init__(
        self,
        redis_url: str,
        *,
        namespace: str = "charaforge:t2i",
        job_ttl_seconds: int = 86400,
        stale_seconds: int = 600,
        max_attempts: int = 2,
        max_concurrent_per_owner: int = 1,
        max_global_concurrent: int = 0,
        worker_enabled: bool = True,
    ):
        if redis is None:
            raise RuntimeError("redis-py is not installed")

        self._namespace = namespace
        self._redis_url = redis_url
        self._job_ttl_seconds = int(job_ttl_seconds)
        self._stale_seconds = int(stale_seconds)
        self._max_attempts = int(max_attempts)
        self._max_concurrent_per_owner = int(max_concurrent_per_owner)
        self._max_global_concurrent = int(max_global_concurrent)
        self._worker_enabled = bool(worker_enabled)

        self._worker_id = uuid.uuid4().hex
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None

        self._client = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=0.2,
            socket_timeout=0.2,
            retry_on_timeout=False,
        )

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
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="t2i-job-worker-redis",
            daemon=True,
        )
        self._worker.start()

    def shutdown(self, timeout_s: float = 1.0) -> None:
        self._stop_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=timeout_s)

    def submit(self, req: GenerateRequest, *, owner: str) -> str:
        self.start()

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
            "attempts": 0,
            "cancel_requested": False,
            "progress_step": None,
            "progress_total": int(req.steps),
            "seed": None,
            "elapsed_ms": None,
            "metadata": {},
            "saved_paths": [],
            "error": None,
            "request": req.model_dump(),
        }

        payload = json.dumps(record, ensure_ascii=False)
        pipe = self._client.pipeline()
        pipe.set(self._job_key(job_id), payload)
        pipe.rpush(self._queue_key(), job_id)
        pipe.sadd(self._owner_queued_key(owner), job_id)
        pipe.zadd(self._jobs_index_key(), {job_id: now})
        pipe.zadd(self._owner_jobs_index_key(owner), {job_id: now})
        pipe.execute()

        return job_id

    def cancel(self, job_id: str) -> Dict[str, Any] | None:
        record = self._load_job(job_id)
        if record is None:
            return None

        status = str(record.get("status") or "")
        owner = str(record.get("owner") or "")

        record["cancel_requested"] = True
        record["updated_at"] = time.time()
        self._client.set(self._cancel_key(job_id), "1", ex=max(self._job_ttl_seconds, 300))

        if status == "queued":
            record["status"] = "canceled"
            record["finished_at"] = time.time()
            record["error"] = {"error": "CANCELED", "message": "Canceled"}
            pipe = self._client.pipeline()
            pipe.lrem(self._queue_key(), 0, job_id)
            if owner:
                pipe.srem(self._owner_queued_key(owner), job_id)
                pipe.srem(self._owner_running_key(owner), job_id)
            pipe.execute()

        self._save_job(job_id, record)
        return _snapshot_from_record(record)

    def get(self, job_id: str) -> Dict[str, Any] | None:
        record = self._load_job(job_id)
        return _snapshot_from_record(record) if record else None

    def get_owner(self, job_id: str) -> str | None:
        record = self._load_job(job_id)
        if record is None:
            return None
        owner = record.get("owner")
        return str(owner) if owner else None

    def owner_counts(self, owner: str) -> Dict[str, int]:
        if not owner:
            return {"queued": 0, "running": 0}
        try:
            pipe = self._client.pipeline()
            pipe.scard(self._owner_queued_key(owner))
            pipe.scard(self._owner_running_key(owner))
            queued, running = pipe.execute()
            return {"queued": int(queued or 0), "running": int(running or 0)}
        except Exception:
            return {"queued": 0, "running": 0}

    def global_counts(self) -> Dict[str, int]:
        try:
            pipe = self._client.pipeline()
            pipe.llen(self._queue_key())
            pipe.llen(self._processing_key())
            queued, processing = pipe.execute()
            return {"queued": int(queued or 0), "running": int(processing or 0)}
        except Exception:
            return {"queued": 0, "running": 0}

    def list_jobs(
        self, *, owner: str | None = None, limit: int = 50, status: str | None = None
    ) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit or 50), 200))
        key = self._owner_jobs_index_key(owner) if owner else self._jobs_index_key()

        fetch = min(1000, max(limit * 5, limit))
        try:
            job_ids = [str(v) for v in (self._client.zrevrange(key, 0, fetch - 1) or [])]
        except Exception:
            job_ids = []

        out: List[Dict[str, Any]] = []
        stale: List[str] = []
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
            pipe.lrem(self._processing_key(), 0, job_id)
            pipe.zrem(self._jobs_index_key(), job_id)
            if owner:
                pipe.srem(self._owner_queued_key(owner), job_id)
                pipe.srem(self._owner_running_key(owner), job_id)
                pipe.zrem(self._owner_jobs_index_key(owner), job_id)
            pipe.execute()
        except Exception:
            return False

        for index in range(1, int(self._max_global_concurrent or 0) + 1):
            self._release_global_slot(self._slot_key(index), job_id)

        return True

    def _requeue_processing(self, job_id: str) -> None:
        try:
            pipe = self._client.pipeline()
            pipe.lrem(self._processing_key(), 0, job_id)
            pipe.rpush(self._queue_key(), job_id)
            pipe.execute()
        except Exception:
            try:
                self._client.rpush(self._queue_key(), job_id)
            except Exception:
                pass
            try:
                self._client.lrem(self._processing_key(), 0, job_id)
            except Exception:
                pass
        time.sleep(0.25)

    def _slot_key(self, index: int) -> str:
        return f"{self._namespace}:gpu_slot:{index}"

    def _slot_ttl_seconds(self) -> int:
        return max(600, int(self._stale_seconds or 0) * 3)

    def _acquire_global_slot(self, job_id: str) -> Tuple[bool, str | None]:
        if self._max_global_concurrent <= 0:
            return True, None

        ttl = self._slot_ttl_seconds()
        try:
            for index in range(1, self._max_global_concurrent + 1):
                key = self._slot_key(index)
                acquired = self._client.set(key, job_id, nx=True, ex=ttl)
                if acquired:
                    return True, key
        except Exception:
            return True, None
        return False, None

    def _renew_global_slot(self, slot_key: str, job_id: str) -> None:
        if not slot_key:
            return
        ttl = self._slot_ttl_seconds()
        try:
            self._client.eval(
                "if redis.call('get', KEYS[1]) == ARGV[1] then "
                "return redis.call('expire', KEYS[1], ARGV[2]) else return 0 end",
                1,
                slot_key,
                job_id,
                str(int(ttl)),
            )
        except Exception:
            return

    def _release_global_slot(self, slot_key: str, job_id: str) -> None:
        if not slot_key:
            return
        try:
            self._client.eval(
                "if redis.call('get', KEYS[1]) == ARGV[1] then "
                "return redis.call('del', KEYS[1]) else return 0 end",
                1,
                slot_key,
                job_id,
            )
        except Exception:
            return

    def _worker_loop(self) -> None:
        next_requeue = 0.0
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now >= next_requeue:
                self._requeue_stale()
                next_requeue = now + 10.0

            try:
                job_id = self._client.brpoplpush(
                    self._queue_key(), self._processing_key(), timeout=1
                )
            except Exception:
                time.sleep(0.25)
                continue

            if not job_id:
                continue
            self._process_job(str(job_id))

    def _process_job(self, job_id: str) -> None:
        record = self._load_job(job_id)
        if record is None:
            self._cleanup_processing(job_id)
            return

        status = str(record.get("status") or "")
        owner = str(record.get("owner") or "")

        if status != "queued":
            self._cleanup_processing(job_id)
            return

        if self._is_cancel_requested(job_id):
            record["status"] = "canceled"
            record["finished_at"] = time.time()
            record["updated_at"] = time.time()
            record["cancel_requested"] = True
            record["error"] = {"error": "CANCELED", "message": "Canceled"}
            self._save_job(job_id, record)
            if owner:
                self._client.srem(self._owner_queued_key(owner), job_id)
                self._client.srem(self._owner_running_key(owner), job_id)
            self._cleanup_processing(job_id)
            return

        if owner and self._max_concurrent_per_owner > 0:
            try:
                running_now = int(self._client.scard(self._owner_running_key(owner)) or 0)
            except Exception:
                running_now = 0

            if running_now >= self._max_concurrent_per_owner:
                self._requeue_processing(job_id)
                return

        slot_allowed, slot_key = self._acquire_global_slot(job_id)
        if not slot_allowed:
            self._requeue_processing(job_id)
            return

        try:
            record["status"] = "running"
            record["started_at"] = time.time()
            record["updated_at"] = time.time()
            record["progress_step"] = 0
            record["progress_total"] = int(record.get("progress_total") or 0)

            self._save_job(job_id, record)

            if owner:
                pipe = self._client.pipeline()
                pipe.srem(self._owner_queued_key(owner), job_id)
                pipe.sadd(self._owner_running_key(owner), job_id)
                pipe.execute()

            cancel_event = threading.Event()
            last_slot_renew = 0.0

            def _on_progress(step: int, total: int) -> None:
                if self._is_cancel_requested(job_id):
                    cancel_event.set()

                record_update = self._load_job(job_id) or {}
                if record_update.get("status") != "running":
                    return
                record_update["progress_step"] = int(step)
                record_update["progress_total"] = int(total)
                record_update["updated_at"] = time.time()
                self._save_job(job_id, record_update)

                nonlocal last_slot_renew
                if slot_key:
                    now = time.time()
                    if now - last_slot_renew >= 10.0:
                        self._renew_global_slot(slot_key, job_id)
                        last_slot_renew = now

            try:
                req_payload = record.get("request") or {}
                req = GenerateRequest.model_validate(req_payload)
                result = run_generate_sync(
                    req,
                    job_id,
                    cancel_event=cancel_event,
                    on_progress=_on_progress,
                )
            except GenerationCancelledError:
                final = self._load_job(job_id) or record
                final["status"] = "canceled"
                final["finished_at"] = time.time()
                final["updated_at"] = time.time()
                final["cancel_requested"] = True
                final["error"] = {"error": "CANCELED", "message": "Canceled"}
                self._save_job(job_id, final)
            except Exception as exc:
                final = self._load_job(job_id) or record
                final["status"] = "failed"
                final["finished_at"] = time.time()
                final["updated_at"] = time.time()
                final["error"] = {"error": "GENERATION_FAILED", "message": str(exc)}
                self._save_job(job_id, final)
            else:
                final = self._load_job(job_id) or record
                final["status"] = "succeeded"
                final["finished_at"] = time.time()
                final["updated_at"] = time.time()
                final["seed"] = int(result.get("seed", 0))
                final["elapsed_ms"] = int(result.get("elapsed_ms", 0))
                final["metadata"] = result.get("metadata") or {}
                final["saved_paths"] = list(result.get("saved_paths") or [])
                final["error"] = None
                self._save_job(job_id, final)

            if owner:
                pipe = self._client.pipeline()
                pipe.srem(self._owner_queued_key(owner), job_id)
                pipe.srem(self._owner_running_key(owner), job_id)
                pipe.execute()

            self._client.delete(self._cancel_key(job_id))
            self._cleanup_processing(job_id)
        finally:
            if slot_key:
                self._release_global_slot(slot_key, job_id)

    def _requeue_stale(self) -> None:
        lock_key = f"{self._namespace}:reaper_lock"
        lock_value = self._worker_id
        try:
            acquired = self._client.set(lock_key, lock_value, nx=True, ex=15)
        except Exception:
            return
        if not acquired:
            return

        try:
            processing = self._client.lrange(self._processing_key(), 0, -1) or []
            now = time.time()
            for job_id in processing:
                record = self._load_job(job_id)
                if record is None:
                    self._cleanup_processing(job_id)
                    continue

                status = str(record.get("status") or "")
                if status in {"succeeded", "failed", "canceled"}:
                    self._cleanup_processing(job_id)
                    continue

                updated_at = float(record.get("updated_at") or 0.0)
                if self._stale_seconds > 0 and now - updated_at < self._stale_seconds:
                    continue

                owner = str(record.get("owner") or "")
                attempts = int(record.get("attempts") or 0) + 1
                record["attempts"] = attempts
                record["updated_at"] = time.time()

                if attempts >= self._max_attempts:
                    record["status"] = "failed"
                    record["finished_at"] = time.time()
                    record["error"] = {
                        "error": "STALE_WORKER",
                        "message": "Job was abandoned by a worker",
                    }
                    self._save_job(job_id, record)
                    if owner:
                        self._client.srem(self._owner_running_key(owner), job_id)
                        self._client.srem(self._owner_queued_key(owner), job_id)
                    self._cleanup_processing(job_id)
                    continue

                record["status"] = "queued"
                record["started_at"] = None
                record["progress_step"] = None
                record["error"] = None
                self._save_job(job_id, record)

                pipe = self._client.pipeline()
                pipe.lrem(self._processing_key(), 0, job_id)
                pipe.rpush(self._queue_key(), job_id)
                if owner:
                    pipe.srem(self._owner_running_key(owner), job_id)
                    pipe.sadd(self._owner_queued_key(owner), job_id)
                pipe.execute()
        except Exception:
            return
        finally:
            try:
                current = self._client.get(lock_key)
                if current == lock_value:
                    self._client.delete(lock_key)
            except Exception:
                return

    def _cleanup_processing(self, job_id: str) -> None:
        try:
            self._client.lrem(self._processing_key(), 0, job_id)
        except Exception:
            return

    def _is_cancel_requested(self, job_id: str) -> bool:
        try:
            return bool(self._client.exists(self._cancel_key(job_id)))
        except Exception:
            return False

    def _job_key(self, job_id: str) -> str:
        return f"{self._namespace}:job:{job_id}"

    def _cancel_key(self, job_id: str) -> str:
        return f"{self._namespace}:cancel:{job_id}"

    def _queue_key(self) -> str:
        return f"{self._namespace}:queue"

    def _processing_key(self) -> str:
        return f"{self._namespace}:processing"

    def _owner_queued_key(self, owner: str) -> str:
        return f"{self._namespace}:owner:{owner}:queued"

    def _owner_running_key(self, owner: str) -> str:
        return f"{self._namespace}:owner:{owner}:running"

    def _jobs_index_key(self) -> str:
        return f"{self._namespace}:jobs"

    def _owner_jobs_index_key(self, owner: str | None) -> str:
        return f"{self._namespace}:owner:{owner}:jobs"

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
        try:
            if terminal and self._job_ttl_seconds > 0:
                self._client.set(self._job_key(job_id), payload, ex=self._job_ttl_seconds)
            else:
                self._client.set(self._job_key(job_id), payload)
        except Exception:
            return


class T2IJobManager:
    def __init__(
        self,
        *,
        redis_url: str | None = None,
        worker_enabled: bool = True,
        job_ttl_seconds: int = 86400,
        stale_seconds: int = 600,
        max_attempts: int = 2,
        max_concurrent_per_owner: int = 1,
        max_global_concurrent: int = 0,
    ):
        self._backend: _MemoryBackend | _RedisBackend

        if redis_url and str(redis_url).startswith("redis") and redis is not None:
            try:
                candidate = _RedisBackend(
                    redis_url,
                    job_ttl_seconds=job_ttl_seconds,
                    stale_seconds=stale_seconds,
                    max_attempts=max_attempts,
                    max_concurrent_per_owner=max_concurrent_per_owner,
                    max_global_concurrent=max_global_concurrent,
                    worker_enabled=worker_enabled,
                )
                if candidate.ping():
                    self._backend = candidate
                    self._backend.start()
                    return
                logger.warning("Redis unavailable; falling back to in-memory T2I jobs")
            except Exception as exc:
                logger.warning("Redis unavailable; falling back to in-memory T2I jobs: %s", exc)

        self._backend = _MemoryBackend(
            worker_enabled=worker_enabled, max_global_concurrent=max_global_concurrent
        )
        if worker_enabled:
            self._backend.start()

    def shutdown(self, timeout_s: float = 1.0) -> None:
        self._backend.shutdown(timeout_s=timeout_s)

    def submit(self, req: GenerateRequest, *, owner: str) -> str:
        return self._backend.submit(req, owner=owner)

    def cancel(self, job_id: str) -> Dict[str, Any] | None:
        return self._backend.cancel(job_id)

    def get(self, job_id: str) -> Dict[str, Any] | None:
        return self._backend.get(job_id)

    def get_owner(self, job_id: str) -> str | None:
        return self._backend.get_owner(job_id)

    def owner_counts(self, owner: str) -> Dict[str, int]:
        return self._backend.owner_counts(owner)

    def global_counts(self) -> Dict[str, int]:
        return self._backend.global_counts()

    def list_jobs(
        self, *, owner: str | None = None, limit: int = 50, status: str | None = None
    ) -> List[Dict[str, Any]]:
        return self._backend.list_jobs(owner=owner, limit=limit, status=status)

    def delete_job(self, job_id: str) -> bool:
        return self._backend.delete_job(job_id)
