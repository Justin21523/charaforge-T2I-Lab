"""Short-lived WebSocket ticket helpers.

Tickets are signed (HS256) and intended to be passed via `Sec-WebSocket-Protocol` to
avoid sending API keys / access tokens during the WebSocket handshake.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional, Set, Tuple

from api.jwt_tokens import decode_jwt, encode_jwt
from api.t2i_tokens import signing_secret


def make_ws_ticket(
    *,
    subject: str,
    role: str,
    scopes: Set[str],
    job_id: str,
    key_id: str | None,
    ttl_seconds: int,
    now: Optional[float] = None,
) -> Tuple[str, int]:
    issued_at = int(float(now if now is not None else time.time()))
    ttl_seconds = int(ttl_seconds or 0)
    if ttl_seconds <= 0:
        ttl_seconds = 30
    expires_at = int(issued_at + ttl_seconds)

    payload: Dict[str, Any] = {
        "typ": "ws_ticket",
        "aud": "ws.train",
        "sub": str(subject),
        "role": str(role or "user"),
        "scopes": sorted(set(scopes or set())),
        "job_id": str(job_id),
        "iat": issued_at,
        "exp": expires_at,
        "jti": uuid.uuid4().hex,
    }
    if key_id:
        payload["kid"] = str(key_id)

    token = encode_jwt(payload, secret=signing_secret())
    return token, expires_at


def verify_ws_ticket(
    token: str,
    *,
    now: Optional[float] = None,
) -> Dict[str, Any] | None:
    payload = decode_jwt(token, secret=signing_secret(), now=now)
    if not payload:
        return None
    if str(payload.get("typ") or "") != "ws_ticket":
        return None
    if str(payload.get("aud") or "") != "ws.train":
        return None
    if not payload.get("sub"):
        return None
    if not payload.get("job_id"):
        return None
    if not payload.get("jti"):
        return None
    return payload

