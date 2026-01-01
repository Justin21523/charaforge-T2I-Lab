"""Minimal HS256 JWT helpers (no external dependencies)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any, Dict, Optional, Set, Tuple

from api.t2i_tokens import signing_secret


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    raw = (data or "").strip().encode("utf-8")
    pad = b"=" * ((4 - (len(raw) % 4)) % 4)
    return base64.urlsafe_b64decode(raw + pad)


def _sign(message: bytes, secret: bytes) -> str:
    sig = hmac.new(secret, message, hashlib.sha256).digest()
    return _b64url_encode(sig)


def encode_jwt(payload: Dict[str, Any], *, secret: bytes) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url_encode(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )
    message = f"{header_b64}.{payload_b64}".encode("utf-8")
    signature = _sign(message, secret)
    return f"{header_b64}.{payload_b64}.{signature}"


def decode_jwt(token: str, *, secret: bytes, now: Optional[float] = None) -> Dict[str, Any] | None:
    parts = (token or "").split(".")
    if len(parts) != 3:
        return None
    header_b64, payload_b64, signature = parts
    message = f"{header_b64}.{payload_b64}".encode("utf-8")
    expected = _sign(message, secret)
    if not hmac.compare_digest(str(signature), expected):
        return None

    try:
        header = json.loads(_b64url_decode(header_b64))
        if not isinstance(header, dict):
            return None
        if str(header.get("alg") or "") != "HS256":
            return None
    except Exception:
        return None

    try:
        payload = json.loads(_b64url_decode(payload_b64))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    current = float(now if now is not None else time.time())
    exp = payload.get("exp")
    if exp is None:
        return None
    try:
        exp_ts = float(exp)
    except Exception:
        return None
    if current > exp_ts:
        return None

    return payload


def make_access_token(
    *,
    subject: str,
    role: str,
    scopes: Set[str],
    key_id: str | None,
    ttl_seconds: int,
    now: Optional[float] = None,
) -> Tuple[str, int]:
    issued_at = int(float(now if now is not None else time.time()))
    ttl_seconds = int(ttl_seconds or 0)
    if ttl_seconds <= 0:
        ttl_seconds = 900
    expires_at = int(issued_at + ttl_seconds)

    payload: Dict[str, Any] = {
        "typ": "access",
        "sub": str(subject),
        "role": str(role or "user"),
        "scopes": sorted(set(scopes or set())),
        "iat": issued_at,
        "exp": expires_at,
    }
    if key_id:
        payload["kid"] = str(key_id)

    token = encode_jwt(payload, secret=signing_secret())
    return token, expires_at


def verify_access_token(
    token: str,
    *,
    now: Optional[float] = None,
) -> Dict[str, Any] | None:
    payload = decode_jwt(token, secret=signing_secret(), now=now)
    if not payload:
        return None
    if str(payload.get("typ") or "") != "access":
        return None
    if not payload.get("sub"):
        return None
    return payload

