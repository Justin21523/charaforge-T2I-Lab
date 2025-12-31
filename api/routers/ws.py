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

from api.security import get_api_key_role, parse_api_keys
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


@router.websocket("/train/{job_id}")
async def ws_train_progress(websocket: WebSocket, job_id: str) -> None:
    settings = get_settings()
    admin_keys = parse_api_keys(settings.api.api_admin_keys)
    user_keys = parse_api_keys(settings.api.api_keys)
    if settings.api.api_key:
        admin_keys.add(settings.api.api_key)

    auth_enabled = bool(admin_keys or user_keys)
    if auth_enabled:
        header_name = settings.api.key_header or "X-API-Key"
        presented = (
            websocket.headers.get(header_name)
            or websocket.query_params.get("api_key")
            or websocket.query_params.get("token")
        )
        role = get_api_key_role(presented, admin_keys=admin_keys, user_keys=user_keys)
        if not role:
            await websocket.accept()
            await websocket.send_json({"topic": "ws.error", "message": "Unauthorized"})
            await websocket.close(code=4401)
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
