import httpx
import pytest
from httpx import ASGITransport

import core.config as config
from api.main import create_app


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

