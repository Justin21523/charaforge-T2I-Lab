from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport
from PIL import Image

from api.main import app


@pytest.mark.anyio
async def test_datasets_list_and_validate():
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        root_res = await client.get("/api/v1/datasets/root")
        assert root_res.status_code == 200
        raw_root = Path(root_res.json()["raw_root"])

        dataset_dir = raw_root / "myset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        img_path = dataset_dir / "0001.png"
        Image.new("RGB", (8, 8), color=(255, 0, 0)).save(img_path)

        list_res = await client.get("/api/v1/datasets/list")
        assert list_res.status_code == 200
        names = {item["name"] for item in list_res.json()}
        assert "myset" in names

        validate_res = await client.post("/api/v1/datasets/validate", json={"dataset_path": "myset"})
        assert validate_res.status_code == 200
        payload = validate_res.json()
        assert payload["ok"] is True

