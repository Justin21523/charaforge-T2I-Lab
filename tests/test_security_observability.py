import json
import time

import httpx
import pytest
from httpx import ASGITransport

import core.config as config
from api.main import create_app
from api.t2i_jobs import access_meta_path, read_access_owner
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


@pytest.mark.anyio
async def test_scoped_keys_enforce_permissions(make_app):
    app = make_app(
        API_ADMIN_KEYS="admin_key",
        API_RATE_LIMIT="0",
        API_SCAN_RATE_LIMIT="0",
        API_T2I_WORKER_ENABLED="false",
    )
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post(
            "/api/v1/auth/keys",
            json={"role": "user", "scopes": ["datasets:read"], "label": "scoped"},
            headers={"X-API-Key": "admin_key"},
        )
        assert created.status_code == 200
        key_payload = created.json()
        scoped_key = key_payload["key"]
        key_id = key_payload["key_id"]

        datasets = await client.get("/api/v1/datasets/root", headers={"X-API-Key": scoped_key})
        assert datasets.status_code == 200

        blocked = await client.post("/api/v1/t2i/generate", json={}, headers={"X-API-Key": scoped_key})
        assert blocked.status_code == 403
        _assert_error_schema(blocked)
        assert blocked.json()["error"] in {"INSUFFICIENT_SCOPE", "FORBIDDEN"}

        listed = await client.get(
            "/api/v1/auth/keys",
            headers={"X-API-Key": "admin_key"},
        )
        assert listed.status_code == 200
        keys = listed.json()["keys"]
        match = next((item for item in keys if item.get("key_id") == key_id), None)
        assert match is not None
        assert match.get("last_used_at")


@pytest.mark.anyio
async def test_metrics_endpoint_when_enabled(make_app):
    app = make_app(
        PROMETHEUS_ENABLED="true",
        API_ADMIN_KEYS="admin_key",
        API_KEYS="user_key",
        API_RATE_LIMIT="0",
        API_SCAN_RATE_LIMIT="0",
        API_T2I_WORKER_ENABLED="false",
    )
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        user = await client.get("/api/v1/metrics", headers={"X-API-Key": "user_key"})
        assert user.status_code == 403
        _assert_error_schema(user)

        scoped = await client.post(
            "/api/v1/auth/keys",
            json={"role": "user", "scopes": ["metrics:read"], "label": "metrics"},
            headers={"X-API-Key": "admin_key"},
        )
        assert scoped.status_code == 200
        scoped_key = scoped.json()["key"]

        ok = await client.get("/api/v1/metrics", headers={"X-API-Key": scoped_key})
        assert ok.status_code == 200
        assert ok.headers.get("content-type", "").startswith("text/plain")
        assert "charaforge_http_requests_total" in ok.text
        assert "charaforge_auth_refresh_total" in ok.text
        assert "charaforge_auth_revoke_total" in ok.text

        res = await client.get("/api/v1/metrics", headers={"X-API-Key": "admin_key"})
        assert res.status_code == 200
        assert res.headers.get("content-type", "").startswith("text/plain")
        assert "X-Request-ID" in res.headers
        assert "charaforge_http_requests_total" in res.text


@pytest.mark.anyio
async def test_t2i_jobs_list_delete_and_cleanup(make_app):
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

        listed = await client.get("/api/v1/t2i/jobs", headers={"X-API-Key": "user_key_1"})
        assert listed.status_code == 200
        assert any(item.get("job_id") == job_id for item in listed.json().get("jobs") or [])

        other = await client.get("/api/v1/t2i/jobs", headers={"X-API-Key": "user_key_2"})
        assert other.status_code == 200
        assert not any(item.get("job_id") == job_id for item in other.json().get("jobs") or [])

        forbidden_all = await client.get(
            "/api/v1/t2i/jobs", params={"all": "true"}, headers={"X-API-Key": "user_key_1"}
        )
        assert forbidden_all.status_code == 403

        admin_all = await client.get(
            "/api/v1/t2i/jobs", params={"all": "true"}, headers={"X-API-Key": "admin_key"}
        )
        assert admin_all.status_code == 200
        assert any(item.get("job_id") == job_id for item in admin_all.json().get("jobs") or [])

        delete_forbidden = await client.delete(
            f"/api/v1/t2i/jobs/{job_id}", headers={"X-API-Key": "user_key_2"}
        )
        assert delete_forbidden.status_code == 403

        deleted = await client.delete(
            f"/api/v1/t2i/jobs/{job_id}", headers={"X-API-Key": "user_key_1"}
        )
        assert deleted.status_code == 200

        gone = await client.get(f"/api/v1/t2i/status/{job_id}", headers={"X-API-Key": "user_key_1"})
        assert gone.status_code == 404

        submit2 = await client.post("/api/v1/t2i/submit", json=payload, headers={"X-API-Key": "user_key_1"})
        assert submit2.status_code == 200
        job_id2 = submit2.json()["job_id"]

        cancelled = await client.post(
            f"/api/v1/t2i/cancel/{job_id2}", headers={"X-API-Key": "user_key_1"}
        )
        assert cancelled.status_code == 200

        meta_path = access_meta_path(job_id2)
        raw = meta_path.read_text(encoding="utf-8")
        meta = json.loads(raw)
        meta["created_at"] = time.time() - 10.0
        meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

        cleaned = await client.post(
            "/api/v1/t2i/jobs/cleanup",
            params={"ttl_seconds": 1, "dry_run": "false", "delete_records": "true"},
            headers={"X-API-Key": "user_key_1"},
        )
        assert cleaned.status_code == 200
        assert cleaned.json()["deleted_outputs"] >= 1
        assert cleaned.json()["deleted_records"] >= 1

        gone2 = await client.get(f"/api/v1/t2i/status/{job_id2}", headers={"X-API-Key": "user_key_1"})
        assert gone2.status_code == 404


@pytest.mark.anyio
async def test_jwt_token_exchange_refresh_and_bearer_auth(make_app):
    app = make_app(
        API_ADMIN_KEYS="admin_key",
        API_KEYS="user_key",
        API_RATE_LIMIT="0",
        API_SCAN_RATE_LIMIT="0",
        API_T2I_WORKER_ENABLED="false",
        JWT_SECRET="test-signing-secret",
        API_JWT_ACCESS_TTL_SECONDS="60",
        API_JWT_REFRESH_TTL_SECONDS="3600",
    )
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        issued = await client.post("/api/v1/auth/token", headers={"X-API-Key": "user_key"})
        assert issued.status_code == 200
        payload = issued.json()
        assert payload.get("access_token")
        refresh_cookie = client.cookies.get("cfr_refresh")
        csrf_cookie = client.cookies.get("cfr_csrf")
        assert isinstance(refresh_cookie, str) and refresh_cookie
        assert isinstance(csrf_cookie, str) and csrf_cookie

        bearer = {"Authorization": f"Bearer {payload['access_token']}"}
        res = await client.get("/api/v1/datasets/root", headers=bearer)
        assert res.status_code == 200

        # The refresh endpoint is usable without an API key.
        refreshed = await client.post("/api/v1/auth/refresh", headers={"X-CSRF-Token": csrf_cookie})
        assert refreshed.status_code == 200
        refreshed_payload = refreshed.json()
        assert refreshed_payload.get("access_token")
        rotated_cookie = client.cookies.get("cfr_refresh")
        assert isinstance(rotated_cookie, str) and rotated_cookie
        assert rotated_cookie != refresh_cookie

        # Old refresh token is invalid after rotation.
        reused = await client.post(
            "/api/v1/auth/refresh", json={"refresh_token": refresh_cookie}
        )
        assert reused.status_code == 401
        _assert_error_schema(reused)

        # /auth/token requires API key auth (not an access token).
        blocked = await client.post("/api/v1/auth/token", headers={"Authorization": bearer["Authorization"]})
        assert blocked.status_code == 401
        _assert_error_schema(blocked)


@pytest.mark.anyio
async def test_refresh_replay_revokes_subject_sessions(make_app):
    app = make_app(
        API_KEYS="user_key",
        API_RATE_LIMIT="0",
        API_SCAN_RATE_LIMIT="0",
        API_T2I_WORKER_ENABLED="false",
        JWT_SECRET="test-signing-secret",
        API_JWT_ACCESS_TTL_SECONDS="60",
        API_JWT_REFRESH_TTL_SECONDS="3600",
    )
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        issued = await client.post("/api/v1/auth/token", headers={"X-API-Key": "user_key"})
        assert issued.status_code == 200
        old_refresh = client.cookies.get("cfr_refresh")
        old_csrf = client.cookies.get("cfr_csrf")
        assert isinstance(old_refresh, str) and old_refresh
        assert isinstance(old_csrf, str) and old_csrf

        rotated = await client.post("/api/v1/auth/refresh", headers={"X-CSRF-Token": old_csrf})
        assert rotated.status_code == 200
        new_refresh = client.cookies.get("cfr_refresh")
        new_csrf = client.cookies.get("cfr_csrf")
        assert isinstance(new_refresh, str) and new_refresh
        assert new_refresh != old_refresh
        assert isinstance(new_csrf, str) and new_csrf

        replay = await client.post("/api/v1/auth/refresh", json={"refresh_token": old_refresh})
        assert replay.status_code == 401
        _assert_error_schema(replay)
        assert replay.json().get("error") == "REFRESH_REPLAY_DETECTED"

        # The active session is revoked after replay detection.
        after = await client.post("/api/v1/auth/refresh", headers={"X-CSRF-Token": new_csrf})
        assert after.status_code == 401
        _assert_error_schema(after)


@pytest.mark.anyio
async def test_revoking_key_revokes_refresh_tokens(make_app):
    app = make_app(
        API_ADMIN_KEYS="admin_key",
        API_RATE_LIMIT="0",
        API_SCAN_RATE_LIMIT="0",
        API_T2I_WORKER_ENABLED="false",
        JWT_SECRET="test-signing-secret",
        API_JWT_ACCESS_TTL_SECONDS="60",
        API_JWT_REFRESH_TTL_SECONDS="3600",
    )
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post(
            "/api/v1/auth/keys",
            json={"role": "user", "scopes": ["datasets:read"], "label": "revocation_test"},
            headers={"X-API-Key": "admin_key"},
        )
        assert created.status_code == 200
        created_payload = created.json()
        key_id = created_payload["key_id"]
        raw_key = created_payload["key"]

        issued = await client.post("/api/v1/auth/token", headers={"X-API-Key": raw_key})
        assert issued.status_code == 200
        refresh_token = client.cookies.get("cfr_refresh")
        assert isinstance(refresh_token, str) and refresh_token

        revoked = await client.post(
            f"/api/v1/auth/keys/{key_id}/revoke", headers={"X-API-Key": "admin_key"}
        )
        assert revoked.status_code == 200
        assert int(revoked.json().get("revoked_refresh_tokens") or 0) >= 1

        refreshed = await client.post(
            "/api/v1/auth/refresh", json={"refresh_token": refresh_token}
        )
        assert refreshed.status_code == 401
        _assert_error_schema(refreshed)
