# api/dependencies.py - Dependency injection
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
import jwt

from core.config import get_settings, get_cache_paths

security = HTTPBearer(auto_error=False)
settings = get_settings()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Optional[Dict[str, Any]]:
    """Get current user from JWT token (optional)"""
    if not credentials:
        return None

    try:
        # Decode JWT token (if authentication is enabled)
        # For now, just return a mock user
        return {
            "user_id": "default",
            "username": "api_user",
            "permissions": ["read", "write"],
        }
    except Exception:
        return None


async def verify_api_key(request: Request) -> bool:
    """Verify API key if required"""
    # Check if API key authentication is enabled
    api_key = request.headers.get("X-API-Key")

    # For now, allow all requests
    # In production, implement proper API key validation
    return True


async def get_shared_cache():
    """Dependency to ensure shared cache is initialized"""
    cache_paths = get_cache_paths()
    return cache_paths
