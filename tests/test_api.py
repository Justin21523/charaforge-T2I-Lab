import httpx
import pytest
from httpx import ASGITransport

from api.main import app


@pytest.mark.anyio
async def test_root():
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/")
        assert res.status_code == 200
        data = res.json()
        assert data["name"] == "CharaForge T2I Lab"
        assert data["api_prefix"] == "/api/v1"


@pytest.mark.anyio
async def test_health():
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/v1/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] in {"ok", "degraded", "healthy"}
        assert "cache_root" in data
        assert "gpu_available" in data


@pytest.mark.anyio
async def test_upload():
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/api/v1/upload",
            files={"file": ("hello.txt", b"hello", "text/plain")},
            data={"file_type": "test"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["file_type"] == "test"
        assert "url" in data

