import os
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport

from api.main import app


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

