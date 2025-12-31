"""Signed token helpers for T2I download URLs."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from pathlib import Path
from typing import Optional

from core.config import get_cache_paths


def _secret_path() -> Path:
    cache_paths = get_cache_paths()
    return cache_paths.cache / "auth" / "signing_secret.txt"


def _load_or_create_secret_file(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = path.read_bytes()
        if data:
            return data
    except FileNotFoundError:
        pass
    except Exception:
        pass

    secret = secrets.token_bytes(32)
    for _ in range(10):
        try:
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            try:
                data = path.read_bytes()
                if data:
                    return data
            except Exception:
                pass
            time.sleep(0.01)
            continue
        except Exception:
            break
        else:
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(secret)
            finally:
                try:
                    os.close(fd)
                except Exception:
                    pass
            return secret

    try:
        data = path.read_bytes()
        if data:
            return data
    except Exception:
        pass
    try:
        path.write_bytes(secret)
    except Exception:
        pass
    return secret


def signing_secret() -> bytes:
    env = os.getenv("JWT_SECRET") or os.getenv("API_SIGNING_SECRET") or ""
    env = env.strip()
    if env and env.lower() not in {"changeme", "your-secret-key-here"}:
        return env.encode("utf-8")
    return _load_or_create_secret_file(_secret_path())


def make_image_token(
    *,
    job_id: str,
    filename: str,
    owner: str,
    ttl_seconds: int,
    now: Optional[float] = None,
) -> str:
    ttl_seconds = int(ttl_seconds or 0)
    if ttl_seconds <= 0:
        return ""

    issued_at = float(now if now is not None else time.time())
    expires_at = int(issued_at + ttl_seconds)
    payload = f"{job_id}:{filename}:{owner}:{expires_at}".encode("utf-8")
    sig = hmac.new(signing_secret(), payload, hashlib.sha256).hexdigest()
    return f"{expires_at}.{sig}"


def verify_image_token(
    token: str,
    *,
    job_id: str,
    filename: str,
    owner: str,
    now: Optional[float] = None,
) -> bool:
    if not token or "." not in token:
        return False
    exp_raw, sig = token.split(".", 1)
    try:
        expires_at = int(exp_raw)
    except Exception:
        return False

    current = float(now if now is not None else time.time())
    if current > float(expires_at):
        return False

    payload = f"{job_id}:{filename}:{owner}:{expires_at}".encode("utf-8")
    expected = hmac.new(signing_secret(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(str(sig), expected)
