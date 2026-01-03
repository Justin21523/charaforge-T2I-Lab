"""WebSocket endpoints for realtime progress updates.

Current channels:
- Training progress: `/api/v1/ws/train/{job_id}`

Implementation:
- Workers publish JSON messages to Redis pubsub channel: `charaforge:train:{job_id}`
- API subscribes and forwards messages to connected WebSocket clients.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.jwt_tokens import verify_access_token
from api.security import parse_api_keys, resolve_api_key, scope_allows
from api.train_access import read_train_access_owner
from api.ws_tickets import verify_ws_ticket
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
    metrics = getattr(getattr(websocket, "app", None), "state", None)
    metrics = getattr(metrics, "metrics", None)
    admin_keys = parse_api_keys(settings.api.api_admin_keys)
    user_keys = parse_api_keys(settings.api.api_keys)
    if settings.api.api_key:
        admin_keys.add(settings.api.api_key)

    protocols = _parse_ws_protocols(websocket.headers.get("sec-websocket-protocol"))
    selected_subprotocol = "charaforge" if "charaforge" in protocols else None

    api_key_store = None
    try:
        from api.key_store import APIKeyStore

        api_key_store = APIKeyStore.default()
    except Exception:
        api_key_store = None

    auth_enabled = bool(admin_keys or user_keys or (api_key_store and api_key_store.has_active_keys()))
    ws_ticket_payload: dict | None = None
    if auth_enabled:
        required_scope = "train:manage"

        subject = ""
        proto_access_token = ""
        proto_api_key = ""
        for proto in protocols:
            if proto.startswith("access_token."):
                proto_access_token = proto.split(".", 1)[1]
            elif proto.startswith("api_key."):
                proto_api_key = proto.split(".", 1)[1]
            elif proto.startswith("ws_ticket."):
                ws_ticket_payload = verify_ws_ticket(proto.split(".", 1)[1])

        allow_query_auth = bool(getattr(settings.api, "ws_allow_query_auth", True))
        access_token = None
        if not ws_ticket_payload:
            access_token = proto_access_token or (
                websocket.query_params.get("access_token") if allow_query_auth else None
            )
        if access_token:
            payload = verify_access_token(access_token)
            if not payload:
                if metrics is not None:
                    try:
                        metrics.inc_ws_connection(endpoint="train", outcome="unauthorized")
                    except Exception:
                        pass
                await websocket.accept(subprotocol=selected_subprotocol)
                await websocket.send_json({"topic": "ws.error", "message": "Unauthorized"})
                await websocket.close(code=4401)
                return

            role = str(payload.get("role") or "user")
            subject = str(payload.get("sub") or "")
            scopes_raw = payload.get("scopes") or []
            if isinstance(scopes_raw, str):
                scopes_raw = [scopes_raw]
            scopes = {str(s).strip() for s in scopes_raw if str(s).strip()}
            if scopes and not scope_allows(scopes, required_scope):
                if metrics is not None:
                    try:
                        metrics.inc_ws_connection(endpoint="train", outcome="forbidden")
                    except Exception:
                        pass
                await websocket.accept(subprotocol=selected_subprotocol)
                await websocket.send_json({"topic": "ws.error", "message": "Forbidden"})
                await websocket.close(code=4403)
                return
            if not scopes and role != "admin":
                # Legacy un-scoped keys are allowed (matches HTTP middleware behavior).
                pass
        elif ws_ticket_payload:
            if str(ws_ticket_payload.get("job_id") or "") != str(job_id):
                if metrics is not None:
                    try:
                        metrics.inc_ws_connection(endpoint="train", outcome="forbidden")
                    except Exception:
                        pass
                await websocket.accept(subprotocol=selected_subprotocol)
                await websocket.send_json({"topic": "ws.error", "message": "Forbidden"})
                await websocket.close(code=4403)
                return

            role = str(ws_ticket_payload.get("role") or "user")
            subject = str(ws_ticket_payload.get("sub") or "")
            scopes_raw = ws_ticket_payload.get("scopes") or []
            if isinstance(scopes_raw, str):
                scopes_raw = [scopes_raw]
            scopes = {str(s).strip() for s in scopes_raw if str(s).strip()}

            if scopes and not scope_allows(scopes, required_scope):
                if metrics is not None:
                    try:
                        metrics.inc_ws_connection(endpoint="train", outcome="forbidden")
                    except Exception:
                        pass
                await websocket.accept(subprotocol=selected_subprotocol)
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
                or (websocket.query_params.get("api_key") if allow_query_auth else None)
                or (websocket.query_params.get("token") if allow_query_auth else None)
            )
            auth = resolve_api_key(
                presented,
                admin_keys=admin_keys,
                user_keys=user_keys,
                key_store=api_key_store,
            )
            if not auth:
                if metrics is not None:
                    try:
                        metrics.inc_ws_connection(endpoint="train", outcome="unauthorized")
                    except Exception:
                        pass
                await websocket.accept(subprotocol=selected_subprotocol)
                await websocket.send_json({"topic": "ws.error", "message": "Unauthorized"})
                await websocket.close(code=4401)
                return

            presented_key = str(presented or "")
            digest = hashlib.sha256(presented_key.encode("utf-8")).hexdigest()[:32]
            subject = f"key:{digest}"
            if auth.scopes and not scope_allows(auth.scopes, required_scope):
                if metrics is not None:
                    try:
                        metrics.inc_ws_connection(endpoint="train", outcome="forbidden")
                    except Exception:
                        pass
                await websocket.accept(subprotocol=selected_subprotocol)
                await websocket.send_json({"topic": "ws.error", "message": "Forbidden"})
                await websocket.close(code=4403)
                return

            role = str(auth.role or "user")
            scopes = set(auth.scopes or set())

        if role != "admin":
            owner = read_train_access_owner(job_id)
            if not owner or owner != subject:
                if metrics is not None:
                    try:
                        metrics.inc_ws_connection(endpoint="train", outcome="forbidden")
                    except Exception:
                        pass
                await websocket.accept(subprotocol=selected_subprotocol)
                await websocket.send_json({"topic": "ws.error", "message": "Forbidden"})
                await websocket.close(code=4403)
                return

    await websocket.accept(subprotocol=selected_subprotocol)
    if metrics is not None:
        try:
            metrics.inc_ws_connection(endpoint="train", outcome="accepted")
            metrics.inc_ws_active()
        except Exception:
            pass

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
        if metrics is not None:
            try:
                metrics.dec_ws_active()
                metrics.inc_ws_disconnect(endpoint="train", code="1011")
            except Exception:
                pass
        return

    url = _redis_url()
    channel = f"charaforge:train:{job_id}"

    client = redis.from_url(url, decode_responses=True)
    pubsub = client.pubsub()

    disconnect_code: str | None = None
    try:
        if ws_ticket_payload and bool(getattr(settings.api, "ws_ticket_replay_protection", True)):
            jti = str(ws_ticket_payload.get("jti") or "")
            exp = int(ws_ticket_payload.get("exp") or 0)
            now = int(time.time())
            ttl = max(1, exp - now + 5)
            ticket_key = f"charaforge:ws_ticket:{jti}"
            try:
                ok = await client.set(ticket_key, "1", nx=True, ex=ttl)
            except Exception:
                await websocket.send_json(
                    {"topic": "ws.error", "message": "Service unavailable"}
                )
                await websocket.close(code=1011)
                return
            if not ok:
                await websocket.send_json({"topic": "ws.error", "message": "Unauthorized"})
                await websocket.close(code=4401)
                return

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
            except WebSocketDisconnect as exc:
                disconnect_code = str(getattr(exc, "code", "") or "1000")
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
        if metrics is not None:
            try:
                metrics.dec_ws_active()
                metrics.inc_ws_disconnect(endpoint="train", code=disconnect_code or "unknown")
            except Exception:
                pass
