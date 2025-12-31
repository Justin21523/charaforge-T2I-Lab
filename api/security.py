"""API security helpers (API key + rate limiting)."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from fastapi import Request

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore

logger = logging.getLogger(__name__)

_EXEMPT_V1_PATHS = {
    "/api/v1/health",
    "/api/v1/healthz",
    "/api/v1/liveness",
    "/api/v1/readiness",
}


def is_exempt_v1_request(request: Request) -> bool:
    if request.method == "OPTIONS":
        return True
    return request.url.path in _EXEMPT_V1_PATHS


def parse_api_keys(value: str | None) -> set[str]:
    if not value:
        return set()
    keys: set[str] = set()
    for candidate in re.split(r"[,\s]+", value):
        key = candidate.strip()
        if key:
            keys.add(key)
    return keys


if TYPE_CHECKING:
    from api.key_store import APIKeyStore


@dataclass(frozen=True)
class APIKeyAuth:
    role: str
    scopes: set[str]
    key_id: Optional[str] = None
    source: str = "env"


def resolve_api_key(
    presented: str | None,
    *,
    admin_keys: set[str],
    user_keys: set[str],
    key_store: "APIKeyStore | None" = None,
) -> Optional[APIKeyAuth]:
    if not presented:
        return None
    if presented in admin_keys:
        return APIKeyAuth(role="admin", scopes={"*"}, key_id=None, source="env")
    if presented in user_keys:
        return APIKeyAuth(role="user", scopes=set(), key_id=None, source="env")
    if key_store is not None:
        verified = key_store.verify(presented)
        if verified:
            return APIKeyAuth(
                role=str(verified.role),
                scopes=set(verified.scopes),
                key_id=str(verified.key_id),
                source="store",
            )
    return None


def get_api_key_role(
    presented: str | None,
    *,
    admin_keys: set[str],
    user_keys: set[str],
    key_store: "APIKeyStore | None" = None,
) -> Optional[str]:
    resolved = resolve_api_key(
        presented, admin_keys=admin_keys, user_keys=user_keys, key_store=key_store
    )
    return resolved.role if resolved else None


def get_client_key(request: Request, api_key: str | None = None) -> str:
    if api_key:
        digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:32]
        return f"key:{digest}"

    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        candidate = forwarded.split(",")[0].strip()
        if candidate:
            return f"ip:{candidate}"
    if request.client and request.client.host:
        return f"ip:{request.client.host}"
    return "ip:unknown"


def extract_api_key(request: Request, header_name: str) -> Optional[str]:
    return (
        request.headers.get(header_name)
        or request.query_params.get("api_key")
        or request.query_params.get("token")
    )


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    reset_epoch: int


class RateLimiter:
    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self._memory: dict[str, tuple[int, int]] = {}

        self._redis_client = None
        self._redis_url: Optional[str] = None
        self._redis_unavailable_until = 0.0

    def check(self, key: str, limit: int, redis_url: Optional[str] = None) -> RateLimitResult:
        if limit <= 0:
            return RateLimitResult(allowed=True, limit=0, remaining=0, reset_epoch=0)

        now = time.time()
        window_id = int(now // self.window_seconds)
        reset_epoch = int((window_id + 1) * self.window_seconds)

        if redis_url:
            count = self._check_redis(redis_url, key, window_id)
            if count is not None:
                remaining = max(0, limit - count)
                return RateLimitResult(
                    allowed=count <= limit,
                    limit=limit,
                    remaining=remaining,
                    reset_epoch=reset_epoch,
                )

        count = self._check_memory(key, window_id)
        remaining = max(0, limit - count)
        return RateLimitResult(
            allowed=count <= limit,
            limit=limit,
            remaining=remaining,
            reset_epoch=reset_epoch,
        )

    def _check_memory(self, key: str, window_id: int) -> int:
        current = self._memory.get(key)
        if current and current[0] == window_id:
            count = current[1] + 1
        else:
            count = 1
        self._memory[key] = (window_id, count)

        if len(self._memory) > 20_000:
            keep_window = window_id
            self._memory = {
                k: v for k, v in self._memory.items() if v[0] >= keep_window - 1
            }

        return count

    def _get_redis_client(self, redis_url: str):
        if redis is None:
            return None

        now = time.time()
        if now < self._redis_unavailable_until:
            return None

        if redis_url != self._redis_url:
            self._redis_url = redis_url
            self._redis_client = None

        if self._redis_client is None:
            try:
                self._redis_client = redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=0.05,
                    socket_timeout=0.05,
                )
            except Exception:
                self._redis_unavailable_until = now + 10.0
                self._redis_client = None
                return None

        return self._redis_client

    def _check_redis(self, redis_url: str, key: str, window_id: int) -> Optional[int]:
        client = self._get_redis_client(redis_url)
        if client is None:
            return None

        redis_key = f"charaforge:ratelimit:{key}:{window_id}"
        try:
            count = int(client.incr(redis_key))
            client.expire(redis_key, self.window_seconds + 1)
            return count
        except Exception as exc:
            logger.debug("Rate limiter Redis unavailable: %s", exc)
            self._redis_unavailable_until = time.time() + 10.0
            self._redis_client = None
            return None
