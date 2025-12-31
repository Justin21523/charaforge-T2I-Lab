"""
CharaForge T2I Lab - FastAPI entrypoint

API contract:
- All public endpoints are mounted under `/api/v1/*`.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.routers.health import router as health_router
from api.security import (
    RateLimiter,
    extract_api_key,
    get_api_key_role,
    get_client_key,
    is_exempt_v1_request,
    parse_api_keys,
)
from core.config import bootstrap_config, get_settings
from core.exceptions import CharaForgeError

logger = logging.getLogger(__name__)

API_V1_PREFIX = "/api/v1"
REQUEST_ID_HEADER = "X-Request-ID"


def _get_request_id(request: Request) -> str:
    existing = getattr(request.state, "request_id", None)
    if isinstance(existing, str) and existing:
        return existing

    incoming = request.headers.get(REQUEST_ID_HEADER)
    request_id = incoming or uuid.uuid4().hex
    request.state.request_id = request_id
    return request_id


def _error_payload(
    request: Request,
    *,
    code: str,
    message: str,
    details: dict | None = None,
) -> dict:
    return {
        "error": code,
        "message": message,
        "details": details or {},
        "request_id": _get_request_id(request),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    bootstrap_config(verbose=False)
    yield
    manager = getattr(app.state, "t2i_job_manager", None)
    try:
        from api.t2i_jobs import T2IJobManager

        if isinstance(manager, T2IJobManager):
            manager.shutdown()
    except Exception:
        pass


def create_app() -> FastAPI:
    settings = get_settings()
    header_name = settings.api.key_header or "X-API-Key"
    admin_keys = parse_api_keys(settings.api.api_admin_keys)
    user_keys = parse_api_keys(settings.api.api_keys)
    if settings.api.api_key:
        admin_keys.add(settings.api.api_key)
    auth_enabled = bool(admin_keys or user_keys)

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
    try:
        from api.t2i_jobs import T2IJobManager

        app.state.t2i_job_manager = T2IJobManager()
    except Exception as exc:
        logger.warning("T2I async jobs disabled: %s", exc)

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
    async def auth_and_rate_limit(request: Request, call_next):
        path = request.url.path
        if not path.startswith(API_V1_PREFIX) or is_exempt_v1_request(request):
            return await call_next(request)

        is_scan_request = path == f"{API_V1_PREFIX}/models/scan"
        presented = extract_api_key(request, header_name)
        role = get_api_key_role(presented, admin_keys=admin_keys, user_keys=user_keys)
        request.state.auth_role = role or "anonymous"

        rate_limit = int(
            (settings.api.scan_rate_limit if is_scan_request else settings.api.rate_limit) or 0
        )
        rate_result = None
        if rate_limit > 0:
            rate_client_key = get_client_key(request, api_key=presented if role else None)
            bucket = "models_scan" if is_scan_request else "default"
            rate_result = app.state.rate_limiter.check(
                key=f"{bucket}:{rate_client_key}",
                limit=rate_limit,
                redis_url=app.state.redis_url,
            )
            if not rate_result.allowed:
                retry_after = max(0, rate_result.reset_epoch - int(time.time()))
                return JSONResponse(
                    status_code=429,
                    content=_error_payload(
                        request,
                        code="RATE_LIMITED",
                        message="Rate limit exceeded",
                        details={
                            "limit": rate_result.limit,
                            "remaining": rate_result.remaining,
                            "reset": rate_result.reset_epoch,
                        },
                    ),
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(rate_result.limit),
                        "X-RateLimit-Remaining": str(rate_result.remaining),
                        "X-RateLimit-Reset": str(rate_result.reset_epoch),
                    },
                )

        if auth_enabled:
            if not role:
                return JSONResponse(
                    status_code=401,
                    content=_error_payload(
                        request,
                        code="UNAUTHORIZED",
                        message="Unauthorized",
                    ),
                )
            if is_scan_request and role != "admin":
                return JSONResponse(
                    status_code=403,
                    content=_error_payload(
                        request,
                        code="FORBIDDEN",
                        message="Forbidden",
                    ),
                )

        response = await call_next(request)

        if rate_result is not None:
            response.headers["X-RateLimit-Limit"] = str(rate_result.limit)
            response.headers["X-RateLimit-Remaining"] = str(rate_result.remaining)
            response.headers["X-RateLimit-Reset"] = str(rate_result.reset_epoch)

        return response

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        started = time.time()
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        request.state.request_id = request_id

        try:
            response = await call_next(request)
        except Exception as exc:
            logger.exception("Unhandled exception: %s", exc)
            details = {"type": exc.__class__.__name__}
            if settings.debug:
                details["message"] = str(exc)
            response = JSONResponse(
                status_code=500,
                content=_error_payload(
                    request,
                    code="INTERNAL_SERVER_ERROR",
                    message="Internal server error",
                    details=details,
                ),
            )
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers["X-Process-Time"] = f"{time.time() - started:.6f}"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                request,
                code="VALIDATION_ERROR",
                message="Request validation failed",
                details={"errors": exc.errors()},
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        status_code = exc.status_code
        code = "HTTP_ERROR"
        if status_code == 401:
            code = "UNAUTHORIZED"
        elif status_code == 403:
            code = "FORBIDDEN"
        elif status_code == 404:
            code = "NOT_FOUND"
        elif status_code == 405:
            code = "METHOD_NOT_ALLOWED"
        elif status_code == 429:
            code = "RATE_LIMITED"

        detail = exc.detail
        if isinstance(detail, str):
            message = detail
            details = {}
        else:
            message = "HTTP request failed"
            details = {"detail": detail}

        return JSONResponse(
            status_code=status_code,
            content=_error_payload(request, code=code, message=message, details=details),
            headers=exc.headers,
        )

    @app.exception_handler(CharaForgeError)
    async def charaforge_error_handler(request: Request, exc: CharaForgeError):
        return JSONResponse(
            status_code=500,
            content=_error_payload(
                request,
                code=exc.error_code,
                message=exc.message,
                details=exc.details,
            ),
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.api.cors_origins.split(",") if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = create_app()
