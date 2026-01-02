"""Authentication / API key management endpoints.

Supports:
- Managed API keys (admin-only management)
- JWT access tokens (short-lived) + refresh tokens (stored server-side)
"""

from __future__ import annotations

import secrets
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from api.jwt_tokens import make_access_token
from api.key_store import APIKeyStore
from api.refresh_store import RefreshTokenStore
from core.config import get_settings

router = APIRouter(prefix="/auth", tags=["auth"])

CSRF_HEADER_NAME = "X-CSRF-Token"


def _cookie_kwargs(request: Request) -> Dict[str, Any]:
    settings = get_settings()
    secure = bool(settings.api.jwt_cookie_secure) or request.url.scheme == "https"
    samesite = str(settings.api.jwt_cookie_samesite or "lax").lower()
    if samesite not in {"lax", "strict", "none"}:
        samesite = "lax"
    return {
        "path": str(settings.api.jwt_cookie_path or "/api/v1/auth"),
        "domain": str(settings.api.jwt_cookie_domain) if settings.api.jwt_cookie_domain else None,
        "secure": secure,
        "samesite": samesite,
    }


def _set_refresh_cookie(
    request: Request,
    response: Response,
    *,
    refresh_token: str,
    refresh_ttl_seconds: int,
    csrf_token: str,
) -> None:
    settings = get_settings()
    kwargs = _cookie_kwargs(request)
    response.set_cookie(
        key=str(settings.api.jwt_refresh_cookie_name),
        value=str(refresh_token),
        max_age=int(refresh_ttl_seconds),
        httponly=True,
        **kwargs,
    )
    response.set_cookie(
        key=str(settings.api.jwt_csrf_cookie_name),
        value=str(csrf_token),
        max_age=int(refresh_ttl_seconds),
        httponly=False,
        **kwargs,
    )


def _clear_refresh_cookie(request: Request, response: Response) -> None:
    settings = get_settings()
    kwargs = _cookie_kwargs(request)
    response.delete_cookie(
        key=str(settings.api.jwt_refresh_cookie_name),
        path=kwargs.get("path"),
        domain=kwargs.get("domain"),
    )
    response.delete_cookie(
        key=str(settings.api.jwt_csrf_cookie_name),
        path=kwargs.get("path"),
        domain=kwargs.get("domain"),
    )


def _require_csrf(request: Request) -> None:
    settings = get_settings()
    expected = str(request.cookies.get(str(settings.api.jwt_csrf_cookie_name)) or "")
    provided = str(request.headers.get(CSRF_HEADER_NAME) or "")
    if not expected or not provided:
        raise HTTPException(status_code=403, detail={"error": "CSRF_FAILED", "message": "Forbidden"})
    if not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=403, detail={"error": "CSRF_FAILED", "message": "Forbidden"})


def _resolve_refresh_token(request: Request, payload: "RefreshRequest | LogoutRequest | None") -> tuple[str, str]:
    token = str(getattr(payload, "refresh_token", "") or "").strip()
    if token:
        return token, "body"
    settings = get_settings()
    token = str(request.cookies.get(str(settings.api.jwt_refresh_cookie_name)) or "").strip()
    if token:
        return token, "cookie"
    return "", ""


def _normalize_refresh_ttl_seconds(ttl_seconds: int) -> int:
    ttl_seconds = int(ttl_seconds or 0)
    if ttl_seconds <= 0:
        ttl_seconds = 30 * 24 * 3600
    return ttl_seconds


def _require_admin(request: Request) -> None:
    role = getattr(request.state, "auth_role", "anonymous")
    if role == "admin":
        return
    if role == "anonymous":
        raise HTTPException(status_code=401, detail="Unauthorized")
    raise HTTPException(status_code=403, detail="Forbidden")


def _store(request: Request) -> APIKeyStore:
    store = getattr(request.app.state, "api_key_store", None)
    if isinstance(store, APIKeyStore):
        return store
    store = APIKeyStore.default()
    request.app.state.api_key_store = store
    return store


def _refresh_store(request: Request) -> RefreshTokenStore:
    store = getattr(request.app.state, "refresh_token_store", None)
    if isinstance(store, RefreshTokenStore):
        return store
    store = RefreshTokenStore.default(redis_url=getattr(request.app.state, "redis_url", None))
    request.app.state.refresh_token_store = store
    return store


class APIKeyInfo(BaseModel):
    key_id: str
    role: str
    scopes: List[str] = Field(default_factory=list)
    label: Optional[str] = None
    created_at: Optional[str] = None
    last_used_at: Optional[str] = None
    revoked_at: Optional[str] = None


class ListKeysResponse(BaseModel):
    count: int
    keys: List[APIKeyInfo]


class CreateKeyRequest(BaseModel):
    role: str = Field(default="user", pattern="^(admin|user)$")
    scopes: List[str] = Field(default_factory=list)
    label: Optional[str] = Field(default=None, max_length=200)


class CreateKeyResponse(BaseModel):
    key_id: str
    key: str
    role: str
    scopes: List[str] = Field(default_factory=list)
    label: Optional[str] = None


class RotateKeyResponse(BaseModel):
    old_key_id: str
    key_id: str
    key: str
    role: str
    scopes: List[str] = Field(default_factory=list)
    label: Optional[str] = None


class TokenResponse(BaseModel):
    token_type: str = Field(default="bearer")
    access_token: str
    expires_at: int
    refresh_expires_at: int
    role: str
    scopes: List[str] = Field(default_factory=list)


class RefreshRequest(BaseModel):
    refresh_token: Optional[str] = None


class LogoutRequest(BaseModel):
    refresh_token: Optional[str] = None
    all: bool = Field(default=False)


@router.get("/me")
async def me(request: Request) -> Dict[str, Any]:
    return {
        "role": getattr(request.state, "auth_role", "anonymous"),
        "scopes": sorted(getattr(request.state, "auth_scopes", set()) or set()),
        "api_key_id": getattr(request.state, "api_key_id", None),
        "client_key": getattr(request.state, "client_key", None),
        "auth_source": getattr(request.state, "auth_source", "anonymous"),
    }


@router.post("/token", response_model=TokenResponse)
async def issue_token(request: Request, response: Response) -> TokenResponse:
    """Exchange an API key for JWT access + refresh tokens.

    For security, this endpoint requires API key auth (not an existing JWT).
    """

    if getattr(request.state, "auth_source", "anonymous") != "api_key":
        raise HTTPException(status_code=401, detail="Unauthorized")

    role = str(getattr(request.state, "auth_role", "user") or "user")
    scopes = set(getattr(request.state, "auth_scopes", set()) or set())
    key_id = getattr(request.state, "api_key_id", None)
    subject = str(getattr(request.state, "client_key", "") or "")
    if not subject:
        raise HTTPException(status_code=401, detail="Unauthorized")

    settings = get_settings()
    access_token, expires_at = make_access_token(
        subject=subject,
        role=role,
        scopes=scopes,
        key_id=str(key_id) if key_id else None,
        ttl_seconds=int(settings.api.jwt_access_ttl_seconds or 0),
    )

    refresh_store = _refresh_store(request)
    refresh_token, refresh_expires_at = refresh_store.issue(
        subject=subject,
        role=role,
        scopes=scopes,
        key_id=str(key_id) if key_id else None,
        ttl_seconds=int(settings.api.jwt_refresh_ttl_seconds or 0),
    )
    csrf_token = secrets.token_urlsafe(32)
    refresh_ttl_seconds = _normalize_refresh_ttl_seconds(int(settings.api.jwt_refresh_ttl_seconds or 0))
    _set_refresh_cookie(
        request,
        response,
        refresh_token=refresh_token,
        refresh_ttl_seconds=refresh_ttl_seconds,
        csrf_token=csrf_token,
    )

    return TokenResponse(
        access_token=access_token,
        expires_at=expires_at,
        refresh_expires_at=refresh_expires_at,
        role=role,
        scopes=sorted(scopes),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: Request,
    response: Response,
    payload: RefreshRequest | None = None,
) -> TokenResponse:
    """Rotate refresh token and mint a new access token."""

    refresh_store = _refresh_store(request)
    raw_refresh, source = _resolve_refresh_token(request, payload)
    if source == "cookie":
        _require_csrf(request)
    record = refresh_store.verify(raw_refresh)
    if not record:
        peeked = refresh_store.peek(raw_refresh)
        if peeked and str(peeked.get("revoked_reason") or "") == "rotated":
            subject = str(peeked.get("subject") or "")
            key_id = str(peeked.get("key_id") or "") if peeked.get("key_id") else ""
            revoked_subject = refresh_store.revoke_subject(subject) if subject else 0
            revoked_key = refresh_store.revoke_key_id(key_id) if key_id else 0
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "REFRESH_REPLAY_DETECTED",
                    "message": "Unauthorized",
                    "details": {
                        "revoked_subject_sessions": revoked_subject,
                        "revoked_key_sessions": revoked_key,
                    },
                },
            )
        raise HTTPException(status_code=401, detail="Unauthorized")

    subject = str(record.get("subject") or "")
    role = str(record.get("role") or "user")
    scopes = set(record.get("scopes") or [])
    key_id = record.get("key_id")

    api_key_store = getattr(request.app.state, "api_key_store", None)
    if key_id and isinstance(api_key_store, APIKeyStore):
        try:
            if not api_key_store.is_active_key_id(str(key_id)):
                raise HTTPException(status_code=401, detail="Unauthorized")
        except HTTPException:
            raise
        except Exception:
            pass

    settings = get_settings()
    access_token, expires_at = make_access_token(
        subject=subject,
        role=role,
        scopes=scopes,
        key_id=str(key_id) if key_id else None,
        ttl_seconds=int(settings.api.jwt_access_ttl_seconds or 0),
    )

    new_refresh_token, refresh_expires_at = refresh_store.issue(
        subject=subject,
        role=role,
        scopes=scopes,
        key_id=str(key_id) if key_id else None,
        ttl_seconds=int(settings.api.jwt_refresh_ttl_seconds or 0),
    )
    replay_window = int(settings.api.jwt_refresh_replay_window_seconds or 0)
    refresh_store.revoke(raw_refresh, reason="rotated", keep_seconds=replay_window)
    csrf_token = secrets.token_urlsafe(32)
    refresh_ttl_seconds = _normalize_refresh_ttl_seconds(int(settings.api.jwt_refresh_ttl_seconds or 0))
    _set_refresh_cookie(
        request,
        response,
        refresh_token=new_refresh_token,
        refresh_ttl_seconds=refresh_ttl_seconds,
        csrf_token=csrf_token,
    )

    return TokenResponse(
        access_token=access_token,
        expires_at=expires_at,
        refresh_expires_at=refresh_expires_at,
        role=role,
        scopes=sorted(scopes),
    )


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    payload: LogoutRequest | None = None,
) -> Dict[str, Any]:
    refresh_store = _refresh_store(request)
    raw_refresh, source = _resolve_refresh_token(request, payload)
    if source == "cookie":
        _require_csrf(request)
    deleted = 0
    all_sessions = bool(getattr(payload, "all", False))
    if all_sessions:
        record = refresh_store.verify(raw_refresh)
        if record:
            deleted = refresh_store.revoke_subject(str(record.get("subject") or ""))
    else:
        ok = refresh_store.revoke(raw_refresh)
        deleted = 1 if ok else 0
    _clear_refresh_cookie(request, response)
    return {"status": "ok", "revoked": True, "count": deleted}


@router.get("/keys", response_model=ListKeysResponse)
async def list_keys(request: Request, include_revoked: bool = False) -> ListKeysResponse:
    _require_admin(request)
    store = _store(request)
    keys = [APIKeyInfo(**k) for k in store.list_keys(include_revoked=include_revoked)]
    return ListKeysResponse(count=len(keys), keys=keys)


@router.post("/keys", response_model=CreateKeyResponse)
async def create_key(request: Request, payload: CreateKeyRequest) -> CreateKeyResponse:
    _require_admin(request)
    store = _store(request)
    try:
        key_id, raw_key = store.create_key(
            role=payload.role,
            scopes=payload.scopes,
            label=payload.label,
            actor_key_id=getattr(request.state, "api_key_id", None),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return CreateKeyResponse(
        key_id=key_id,
        key=raw_key,
        role=payload.role,
        scopes=payload.scopes,
        label=payload.label,
    )


@router.post("/keys/{key_id}/revoke")
async def revoke_key(request: Request, key_id: str) -> Dict[str, Any]:
    _require_admin(request)
    store = _store(request)
    ok = store.revoke_key(key_id, actor_key_id=getattr(request.state, "api_key_id", None))
    if not ok:
        raise HTTPException(status_code=404, detail="Key not found")
    revoked_refresh = 0
    try:
        revoked_refresh = _refresh_store(request).revoke_key_id(key_id)
    except Exception:
        revoked_refresh = 0
    return {"status": "ok", "key_id": key_id, "revoked": True, "revoked_refresh_tokens": revoked_refresh}


@router.post("/keys/{key_id}/rotate", response_model=RotateKeyResponse)
async def rotate_key(request: Request, key_id: str) -> RotateKeyResponse:
    _require_admin(request)
    store = _store(request)
    rotated = store.rotate_key(key_id, actor_key_id=getattr(request.state, "api_key_id", None))
    if not rotated:
        raise HTTPException(status_code=404, detail="Key not found")

    new_id, raw_key = rotated
    try:
        _refresh_store(request).revoke_key_id(key_id)
    except Exception:
        pass
    record = next((k for k in store.list_keys(include_revoked=True) if k.get("key_id") == new_id), {})

    return RotateKeyResponse(
        old_key_id=key_id,
        key_id=new_id,
        key=raw_key,
        role=str(record.get("role") or "user"),
        scopes=list(record.get("scopes") or []),
        label=record.get("label"),
    )
