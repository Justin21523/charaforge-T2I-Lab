"""Async T2I job manager (in-process worker thread)."""

from __future__ import annotations

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

JobStatus = Literal["queued", "running", "succeeded", "failed", "canceled"]


class GenerationCancelledError(RuntimeError):
    pass


def job_dir(job_id: str) -> Path:
    app_paths = get_app_paths()
    return app_paths.outputs / "t2i" / job_id


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


@dataclass
class T2IJob:
    job_id: str
    request: GenerateRequest
    status: JobStatus = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    cancel_requested: bool = False
    progress_step: Optional[int] = None
    progress_total: Optional[int] = None
    seed: Optional[int] = None
    elapsed_ms: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    saved_paths: List[str] = field(default_factory=list)
    error: Optional[Dict[str, Any]] = None
    _cancel_event: Optional[threading.Event] = field(default=None, repr=False)

    def snapshot(self) -> Dict[str, Any]:
        progress = None
        if self.progress_step is not None and self.progress_total is not None:
            progress = {"step": self.progress_step, "total": self.progress_total}
        return {
            "job_id": self.job_id,
            "status": self.status,
            "cancel_requested": self.cancel_requested,
            "progress": progress,
            "seed": self.seed,
            "elapsed_ms": self.elapsed_ms,
            "metadata": dict(self.metadata or {}),
            "saved_paths": list(self.saved_paths),
            "error": dict(self.error) if self.error else None,
        }


class T2IJobManager:
    def __init__(self, *, max_jobs: int = 200):
        self._max_jobs = max_jobs
        self._lock = threading.Lock()
        self._jobs: Dict[str, T2IJob] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._worker_loop, name="t2i-job-worker", daemon=True)
        self._worker.start()

    def shutdown(self, timeout_s: float = 1.0) -> None:
        self._stop_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=timeout_s)

    def submit(self, req: GenerateRequest) -> str:
        self.start()

        job_id = str(uuid.uuid4())
        job = T2IJob(job_id=job_id, request=req, status="queued")
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
            if job.status == "queued":
                job.status = "canceled"
                job.finished_at = time.time()
                job.error = {"error": "CANCELED", "message": "Canceled"}
            elif job.status == "running":
                if job._cancel_event is not None:
                    job._cancel_event.set()
            return job.snapshot()

    def get(self, job_id: str) -> Dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.snapshot() if job else None

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
                        current.error = {"error": "CANCELED", "message": "Canceled"}
                        current._cancel_event = None
            except Exception as exc:
                with self._lock:
                    current = self._jobs.get(job_id)
                    if current:
                        current.status = "failed"
                        current.finished_at = time.time()
                        current.error = {"error": "GENERATION_FAILED", "message": str(exc)}
                        current._cancel_event = None
            else:
                with self._lock:
                    current = self._jobs.get(job_id)
                    if current:
                        current.status = "succeeded"
                        current.finished_at = time.time()
                        current.seed = int(result.get("seed", 0))
                        current.elapsed_ms = int(result.get("elapsed_ms", 0))
                        current.metadata = result.get("metadata") or {}
                        current.saved_paths = list(result.get("saved_paths") or [])
                        current.error = None
                        current._cancel_event = None
