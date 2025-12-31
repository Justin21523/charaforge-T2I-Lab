import httpx
import pytest
from httpx import ASGITransport

import core.config as config
from api.main import create_app
from api.t2i_jobs import read_access_owner
from api.t2i_tokens import make_image_token


def _reset_settings_cache() -> None:
    config.get_settings.cache_clear()
    config.get_cache_paths.cache_clear()
    config.get_app_paths.cache_clear()
    config._settings_instance = None
    config._cache_paths_instance = None
    config._app_paths_instance = None


@pytest.fixture
def make_app(monkeypatch):
    def _make_app(**env: str):
        for key, value in env.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)
        _reset_settings_cache()
        return create_app()

    yield _make_app
    _reset_settings_cache()


def _assert_error_schema(res: httpx.Response) -> None:
    assert "X-Request-ID" in res.headers
    request_id = res.headers["X-Request-ID"]
    assert isinstance(request_id, str) and request_id

    payload = res.json()
    assert set(payload) >= {"error", "message", "details", "request_id"}
    assert payload["request_id"] == request_id


@pytest.mark.anyio
async def test_error_schema_and_request_id_for_422(make_app):
    app = make_app(API_RATE_LIMIT="0")
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/api/v1/t2i/generate", json={}, headers={"X-Request-ID": "req_422"})
        assert res.status_code == 422
        _assert_error_schema(res)
        assert res.headers["X-Request-ID"] == "req_422"


@pytest.mark.anyio
async def test_error_schema_and_request_id_for_401(make_app):
    app = make_app(API_ADMIN_KEYS="admin_key", API_KEYS="user_key", API_RATE_LIMIT="0")
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/v1/datasets/root")
        assert res.status_code == 401
        _assert_error_schema(res)
        assert res.json()["error"] == "UNAUTHORIZED"


@pytest.mark.anyio
async def test_models_scan_requires_admin_key(make_app):
    app = make_app(API_ADMIN_KEYS="admin_key", API_KEYS="user_key", API_RATE_LIMIT="0", API_SCAN_RATE_LIMIT="0")
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res_user = await client.post(
            "/api/v1/models/scan",
            json={"replace": True},
            headers={"X-API-Key": "user_key"},
        )
        assert res_user.status_code == 403
        _assert_error_schema(res_user)

        res_admin = await client.post(
            "/api/v1/models/scan",
            json={"replace": True},
            headers={"X-API-Key": "admin_key"},
        )
        assert res_admin.status_code == 200


@pytest.mark.anyio
async def test_error_schema_and_request_id_for_429(make_app):
    app = make_app(API_RATE_LIMIT="1", API_SCAN_RATE_LIMIT="0")
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.get("/api/v1/datasets/root")
        assert first.status_code == 200

        second = await client.get("/api/v1/datasets/root")
        assert second.status_code == 429
        _assert_error_schema(second)
        assert second.json()["error"] == "RATE_LIMITED"


@pytest.mark.anyio
async def test_t2i_status_is_owner_only_and_images_accept_token(make_app, tmp_path):
    app = make_app(
        API_ADMIN_KEYS="admin_key",
        API_KEYS="user_key_1,user_key_2",
        API_RATE_LIMIT="0",
        API_SCAN_RATE_LIMIT="0",
        API_T2I_WORKER_ENABLED="false",
        JWT_SECRET="test-signing-secret",
    )
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "prompt": "test",
            "model_type": "sd15",
            "width": 256,
            "height": 256,
            "steps": 1,
            "batch_size": 1,
        }
        submit = await client.post("/api/v1/t2i/submit", json=payload, headers={"X-API-Key": "user_key_1"})
        assert submit.status_code == 200
        job_id = submit.json()["job_id"]

        forbidden = await client.get(f"/api/v1/t2i/status/{job_id}", headers={"X-API-Key": "user_key_2"})
        assert forbidden.status_code == 403
        _assert_error_schema(forbidden)

        ok = await client.get(f"/api/v1/t2i/status/{job_id}", headers={"X-API-Key": "user_key_1"})
        assert ok.status_code == 200

        owner = read_access_owner(job_id)
        assert owner

        token = make_image_token(job_id=job_id, filename="dummy.png", owner=owner, ttl_seconds=3600)
        assert token

        # No API key header (simulates <img>), should still work with a token.
        # The file doesn't exist in this unit test environment; passing the token should
        # get past auth/ACL and return 404 (not 401/403).
        img = await client.get(
            f"/api/v1/t2i/images/{job_id}/dummy.png", params={"img_token": token}
        )
        assert img.status_code == 404
        _assert_error_schema(img)

        bad = await client.get(
            f"/api/v1/t2i/images/{job_id}/dummy.png", params={"img_token": token + "x"}
        )
        assert bad.status_code == 403
        _assert_error_schema(bad)
