"""
CharaForge T2I Lab - FastAPI entrypoint

API contract:
- All public endpoints are mounted under `/api/v1/*`.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from api.routers.health import router as health_router
from api.security import RateLimiter, extract_api_key, get_client_key, is_exempt_v1_request
from core.config import bootstrap_config, get_settings
from core.exceptions import CharaForgeError

logger = logging.getLogger(__name__)

API_V1_PREFIX = "/api/v1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    bootstrap_config(verbose=False)
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="CharaForge T2I Lab",
        description="Text-to-Image generation + LoRA fine-tuning API",
        version="0.2.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.state.rate_limiter = RateLimiter()
    app.state.redis_url = (
        os.getenv("REDIS_URL")
        or os.getenv("CELERY_BROKER_URL")
        or settings.redis_url
        or settings.celery.broker_url
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.api.cors_origins.split(",") if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Routers (mounted under /api/v1)
    app.include_router(health_router, prefix=API_V1_PREFIX)

    optional_routers = [
        ("models", "api.routers.models", "router"),
        ("t2i", "api.routers.t2i", "router"),
        ("controlnet", "api.routers.controlnet", "router"),
        ("lora", "api.routers.lora", "router"),
        ("batch", "api.routers.batch", "router"),
        ("datasets", "api.routers.datasets", "router"),
        ("finetune", "api.routers.finetune", "router"),
        ("upload", "api.routers.upload", "router"),
        ("ws", "api.routers.ws", "router"),
    ]

    for name, module_path, attr in optional_routers:
        try:
            module = __import__(module_path, fromlist=[attr])
            router = getattr(module, attr)
            app.include_router(router, prefix=API_V1_PREFIX)
        except Exception as exc:
            logger.warning("Router '%s' disabled: %s", name, exc)

    @app.get("/")
    async def root():
        return {
            "name": "CharaForge T2I Lab",
            "version": "0.2.0",
            "api_prefix": API_V1_PREFIX,
            "docs": "/docs",
        }

    @app.middleware("http")
    async def request_timing(request: Request, call_next):
        started = time.time()
        response = await call_next(request)
        response.headers["X-Process-Time"] = f"{time.time() - started:.6f}"
        return response

    @app.middleware("http")
    async def auth_and_rate_limit(request: Request, call_next):
        path = request.url.path
        if not path.startswith(API_V1_PREFIX) or is_exempt_v1_request(request):
            return await call_next(request)

        rate_limit = int(settings.api.rate_limit or 0)
        rate_result = None
        if rate_limit > 0:
            key = get_client_key(request)
            rate_result = app.state.rate_limiter.check(
                key=key,
                limit=rate_limit,
                redis_url=app.state.redis_url,
            )
            if not rate_result.allowed:
                retry_after = max(0, rate_result.reset_epoch - int(time.time()))
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Rate limit exceeded",
                        "limit": rate_result.limit,
                        "remaining": rate_result.remaining,
                        "reset": rate_result.reset_epoch,
                    },
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(rate_result.limit),
                        "X-RateLimit-Remaining": str(rate_result.remaining),
                        "X-RateLimit-Reset": str(rate_result.reset_epoch),
                    },
                )

        if settings.api.api_key:
            header_name = settings.api.key_header or "X-API-Key"
            presented = extract_api_key(request, header_name)
            if not presented or presented != settings.api.api_key:
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

        response = await call_next(request)

        if rate_result is not None:
            response.headers["X-RateLimit-Limit"] = str(rate_result.limit)
            response.headers["X-RateLimit-Remaining"] = str(rate_result.remaining)
            response.headers["X-RateLimit-Reset"] = str(rate_result.reset_epoch)

        return response

    @app.exception_handler(CharaForgeError)
    async def charaforge_error_handler(request: Request, exc: CharaForgeError):
        return JSONResponse(status_code=500, content=exc.to_dict())

    return app


app = create_app()
