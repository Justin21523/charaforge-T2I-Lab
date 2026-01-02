"""WebSocket endpoints for realtime progress updates.

Current channels:
- Training progress: `/api/v1/ws/train/{job_id}`

Implementation:
- Workers publish JSON messages to Redis pubsub channel: `charaforge:train:{job_id}`
- API subscribes and forwards messages to connected WebSocket clients.
"""

from __future__ import annotations

import asyncio
import json
import os

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.jwt_tokens import verify_access_token
from api.security import parse_api_keys, resolve_api_key, scope_allows
from core.config import get_settings

router = APIRouter(prefix="/ws", tags=["ws"])


def _redis_url() -> str:
    settings = get_settings()
    return (
        os.getenv("REDIS_URL")
        or os.getenv("CELERY_BROKER_URL")
        or settings.redis_url
        or settings.celery.broker_url
    )


def _parse_ws_protocols(value: str | None) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


@router.websocket("/train/{job_id}")
async def ws_train_progress(websocket: WebSocket, job_id: str) -> None:
    settings = get_settings()
    admin_keys = parse_api_keys(settings.api.api_admin_keys)
    user_keys = parse_api_keys(settings.api.api_keys)
    if settings.api.api_key:
        admin_keys.add(settings.api.api_key)

    api_key_store = None
    try:
        from api.key_store import APIKeyStore

        api_key_store = APIKeyStore.default()
    except Exception:
        api_key_store = None

    auth_enabled = bool(admin_keys or user_keys or (api_key_store and api_key_store.has_active_keys()))
    if auth_enabled:
        required_scope = "train:manage"

        protocols = _parse_ws_protocols(websocket.headers.get("sec-websocket-protocol"))
        proto_access_token = ""
        proto_api_key = ""
        for proto in protocols:
            if proto.startswith("access_token."):
                proto_access_token = proto.split(".", 1)[1]
            elif proto.startswith("api_key."):
                proto_api_key = proto.split(".", 1)[1]

        access_token = proto_access_token or websocket.query_params.get("access_token")
        if access_token:
            payload = verify_access_token(access_token)
            if not payload:
                await websocket.accept()
                await websocket.send_json({"topic": "ws.error", "message": "Unauthorized"})
                await websocket.close(code=4401)
                return

            role = str(payload.get("role") or "user")
            scopes_raw = payload.get("scopes") or []
            if isinstance(scopes_raw, str):
                scopes_raw = [scopes_raw]
            scopes = {str(s).strip() for s in scopes_raw if str(s).strip()}
            if scopes and not scope_allows(scopes, required_scope):
                await websocket.accept()
                await websocket.send_json({"topic": "ws.error", "message": "Forbidden"})
                await websocket.close(code=4403)
                return
            if not scopes and role != "admin":
                # Legacy un-scoped keys are allowed (matches HTTP middleware behavior).
                pass
        else:
            header_name = settings.api.key_header or "X-API-Key"
            presented = (
                websocket.headers.get(header_name)
                or proto_api_key
                or websocket.query_params.get("api_key")
                or websocket.query_params.get("token")
            )
            auth = resolve_api_key(
                presented,
                admin_keys=admin_keys,
                user_keys=user_keys,
                key_store=api_key_store,
            )
            if not auth:
                await websocket.accept()
                await websocket.send_json({"topic": "ws.error", "message": "Unauthorized"})
                await websocket.close(code=4401)
                return

            if auth.scopes and not scope_allows(auth.scopes, required_scope):
                await websocket.accept()
                await websocket.send_json({"topic": "ws.error", "message": "Forbidden"})
                await websocket.close(code=4403)
                return

    await websocket.accept()

    try:
        import redis.asyncio as redis  # type: ignore
    except Exception:
        await websocket.send_json(
            {
                "topic": "ws.error",
                "message": "redis-py is not installed; realtime progress is unavailable",
            }
        )
        await websocket.close(code=1011)
        return

    url = _redis_url()
    channel = f"charaforge:train:{job_id}"

    client = redis.from_url(url, decode_responses=True)
    pubsub = client.pubsub()

    try:
        await pubsub.subscribe(channel)
        await websocket.send_json({"topic": "ws.subscribed", "channel": channel})

        while True:
            try:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if message and message.get("type") == "message":
                    raw = message.get("data")
                    try:
                        payload = json.loads(raw) if isinstance(raw, str) else {"data": raw}
                    except Exception:
                        payload = {"topic": "ws.message", "data": raw}
                    await websocket.send_json(payload)
                else:
                    await asyncio.sleep(0.1)
            except WebSocketDisconnect:
                break
    finally:
        try:
            await pubsub.unsubscribe(channel)
        except Exception:
            pass
        try:
            await pubsub.close()
        except Exception:
            pass
        try:
            await client.close()
        except Exception:
            pass
