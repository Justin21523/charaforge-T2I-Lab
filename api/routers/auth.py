"""Authentication / API key management endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from api.key_store import APIKeyStore

router = APIRouter(prefix="/auth", tags=["auth"])


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


class APIKeyInfo(BaseModel):
    key_id: str
    role: str
    scopes: List[str] = Field(default_factory=list)
    label: Optional[str] = None
    created_at: Optional[str] = None
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


@router.get("/me")
async def me(request: Request) -> Dict[str, Any]:
    return {
        "role": getattr(request.state, "auth_role", "anonymous"),
        "scopes": sorted(getattr(request.state, "auth_scopes", set()) or set()),
        "api_key_id": getattr(request.state, "api_key_id", None),
        "client_key": getattr(request.state, "client_key", None),
    }


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
    ok = store.revoke_key(key_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Key not found")
    return {"status": "ok", "key_id": key_id, "revoked": True}


@router.post("/keys/{key_id}/rotate", response_model=RotateKeyResponse)
async def rotate_key(request: Request, key_id: str) -> RotateKeyResponse:
    _require_admin(request)
    store = _store(request)
    rotated = store.rotate_key(key_id)
    if not rotated:
        raise HTTPException(status_code=404, detail="Key not found")

    new_id, raw_key = rotated
    record = next((k for k in store.list_keys(include_revoked=True) if k.get("key_id") == new_id), {})

    return RotateKeyResponse(
        old_key_id=key_id,
        key_id=new_id,
        key=raw_key,
        role=str(record.get("role") or "user"),
        scopes=list(record.get("scopes") or []),
        label=record.get("label"),
    )

