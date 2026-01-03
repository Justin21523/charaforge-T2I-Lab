import asyncio
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport

from api.main import app
from core.config import get_cache_paths


def _touch(path: Path, content: bytes = b"{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


@pytest.mark.anyio
async def test_models_scan_creates_registry_entries():
    models_root = Path(os.environ["AI_MODELS_ROOT"])

    # Fake local base models (diffusers-style folder markers).
    _touch(models_root / "stable-diffusion" / "sd15" / "demo_sd15" / "model_index.json")
    _touch(models_root / "stable-diffusion" / "sdxl" / "demo_sdxl" / "model_index.json")

    # Fake LoRA.
    _touch(models_root / "lora" / "demo_lora.safetensors", content=b"not-a-real-lora")

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/api/v1/models/scan", json={"replace": True})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["models_after"] >= 3

        res = await client.get("/api/v1/models", params={"model_type": "lora"})
        assert res.status_code == 200
        payload = res.json()
        assert payload["count"] == 1
        assert payload["models"][0]["name"] == "lora/demo_lora"


@pytest.mark.anyio
async def test_models_scan_returns_409_when_in_progress(tmp_path: Path):
    lock_path = get_cache_paths().cache / "locks" / "models_scan.lock"
    ready_path = tmp_path / "scan_lock_ready.txt"

    repo_root = Path(__file__).resolve().parents[1]
    code = textwrap.dedent(
        """
        from pathlib import Path
        import sys
        import time

        from core.file_lock import file_lock

        lock_path = Path(sys.argv[1])
        ready_path = Path(sys.argv[2])

        with file_lock(lock_path, timeout_s=1.0):
            ready_path.write_text("ready", encoding="utf-8")
            time.sleep(10)
        """
    ).strip()

    proc = subprocess.Popen(
        [sys.executable, "-c", code, str(lock_path), str(ready_path)],
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    transport = ASGITransport(app=app)
    try:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if ready_path.exists():
                break
            if proc.poll() is not None:
                stdout, stderr = proc.communicate(timeout=1.0)
                pytest.fail(
                    f"Lock holder exited early (code={proc.returncode}).\n"
                    f"stdout:\n{stdout}\n\nstderr:\n{stderr}"
                )
            await asyncio.sleep(0.05)

        assert ready_path.exists()

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.post("/api/v1/models/scan", json={"replace": True})
            assert res.status_code == 409
            assert "X-Request-ID" in res.headers
            data = res.json()
            assert data.get("error") == "SCAN_IN_PROGRESS"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)


@pytest.mark.anyio
async def test_models_scan_job_submit_and_status_succeeds():
    models_root = Path(os.environ["AI_MODELS_ROOT"])

    _touch(models_root / "stable-diffusion" / "sd15" / "demo_sd15" / "model_index.json")
    _touch(models_root / "stable-diffusion" / "sdxl" / "demo_sdxl" / "model_index.json")
    _touch(models_root / "lora" / "demo_lora.safetensors", content=b"not-a-real-lora")

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        submitted = await client.post("/api/v1/models/scan/submit", json={"replace": True})
        assert submitted.status_code == 200
        job_id = submitted.json()["job_id"]

        snapshot: dict | None = None
        for _ in range(60):
            res = await client.get(f"/api/v1/models/scan/status/{job_id}")
            assert res.status_code == 200
            snapshot = res.json()
            if snapshot.get("status") in {"succeeded", "failed", "canceled"}:
                break
            await asyncio.sleep(0.05)

        assert snapshot is not None
        assert snapshot.get("status") == "succeeded"
        result = snapshot.get("result") or {}
        assert result.get("status") == "ok"
        assert int(result.get("models_after") or 0) >= 3

        res = await client.get("/api/v1/models", params={"model_type": "lora"})
        assert res.status_code == 200
        payload = res.json()
        assert payload["count"] == 1
        assert payload["models"][0]["name"] == "lora/demo_lora"


@pytest.mark.anyio
async def test_models_scan_job_submit_returns_409_when_active(tmp_path: Path):
    lock_path = get_cache_paths().cache / "locks" / "models_scan.lock"
    ready_path = tmp_path / "scan_lock_ready.txt"

    repo_root = Path(__file__).resolve().parents[1]
    code = textwrap.dedent(
        """
        from pathlib import Path
        import sys
        import time

        from core.file_lock import file_lock

        lock_path = Path(sys.argv[1])
        ready_path = Path(sys.argv[2])

        with file_lock(lock_path, timeout_s=1.0):
            ready_path.write_text("ready", encoding="utf-8")
            time.sleep(2.0)
        """
    ).strip()

    proc = subprocess.Popen(
        [sys.executable, "-c", code, str(lock_path), str(ready_path)],
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    transport = ASGITransport(app=app)
    try:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if ready_path.exists():
                break
            if proc.poll() is not None:
                stdout, stderr = proc.communicate(timeout=1.0)
                pytest.fail(
                    f"Lock holder exited early (code={proc.returncode}).\n"
                    f"stdout:\n{stdout}\n\nstderr:\n{stderr}"
                )
            await asyncio.sleep(0.05)
        assert ready_path.exists()

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            first = await client.post("/api/v1/models/scan/submit", json={"replace": True})
            assert first.status_code == 200
            job_id = first.json()["job_id"]

            second = await client.post("/api/v1/models/scan/submit", json={"replace": True})
            assert second.status_code == 409
            assert second.json().get("error") == "MODELS_SCAN_JOB_IN_PROGRESS"

            snapshot: dict | None = None
            for _ in range(80):
                res = await client.get(f"/api/v1/models/scan/status/{job_id}")
                assert res.status_code == 200
                snapshot = res.json()
                if snapshot.get("status") in {"succeeded", "failed", "canceled"}:
                    break
                await asyncio.sleep(0.05)

            assert snapshot is not None
            assert snapshot.get("status") in {"succeeded", "failed"}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)
