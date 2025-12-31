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
    get_client_key,
    is_exempt_v1_request,
    parse_api_keys,
    resolve_api_key,
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
    api_key_store = None
    try:
        from api.key_store import APIKeyStore

        api_key_store = APIKeyStore.default()
    except Exception as exc:
        logger.warning("API key store unavailable: %s", exc)

    store_enabled = bool(api_key_store and api_key_store.has_active_keys())
    auth_enabled = bool(admin_keys or user_keys or store_enabled)

    app = FastAPI(
        title="CharaForge T2I Lab",
        description="Text-to-Image generation + LoRA fine-tuning API",
        version="0.2.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.state.rate_limiter = RateLimiter()
    app.state.auth_enabled = auth_enabled
    app.state.redis_url = (
        os.getenv("REDIS_URL")
        or os.getenv("CELERY_BROKER_URL")
        or settings.redis_url
        or settings.celery.broker_url
    )
    app.state.api_key_store = api_key_store
    try:
        from api.t2i_jobs import T2IJobManager

        app.state.t2i_job_manager = T2IJobManager(
            redis_url=app.state.redis_url,
            worker_enabled=bool(settings.api.t2i_worker_enabled),
            job_ttl_seconds=int(settings.api.t2i_job_ttl_seconds or 0),
            stale_seconds=int(settings.api.t2i_job_stale_seconds or 0),
            max_attempts=int(settings.api.t2i_job_max_attempts or 1),
            max_concurrent_per_owner=int(settings.api.t2i_max_concurrent or 1),
        )
    except Exception as exc:
        logger.warning("T2I async jobs disabled: %s", exc)

    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Routers (mounted under /api/v1)
    app.include_router(health_router, prefix=API_V1_PREFIX)

    optional_routers = [
        ("models", "api.routers.models", "router"),
        ("auth", "api.routers.auth", "router"),
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
        is_t2i_image_request = path.startswith(f"{API_V1_PREFIX}/t2i/images/")
        is_controlnet_image_request = path.startswith(f"{API_V1_PREFIX}/controlnet/images/")
        has_image_token = bool(request.query_params.get("img_token"))
        presented = extract_api_key(request, header_name)
        auth = resolve_api_key(
            presented,
            admin_keys=admin_keys,
            user_keys=user_keys,
            key_store=api_key_store,
        )
        role = auth.role if auth else None
        request.state.auth_role = role or "anonymous"
        request.state.auth_scopes = auth.scopes if auth else set()
        request.state.api_key_id = auth.key_id if auth else None
        request.state.client_key = get_client_key(request, api_key=presented if auth else None)

        rate_client_key = request.state.client_key
        global_rate_limit = int(settings.api.rate_limit or 0)

        special_bucket = None
        special_limit = 0
        if is_scan_request:
            special_bucket = "models_scan"
            special_limit = int(settings.api.scan_rate_limit or 0)
        elif path.startswith(f"{API_V1_PREFIX}/upload"):
            special_bucket = "upload"
            special_limit = int(settings.api.upload_rate_limit or 0)
        elif path.startswith(f"{API_V1_PREFIX}/datasets"):
            special_bucket = "datasets"
            special_limit = int(settings.api.datasets_rate_limit or 0)

        global_result = None
        if global_rate_limit > 0:
            global_result = app.state.rate_limiter.check(
                key=f"global:{rate_client_key}",
                limit=global_rate_limit,
                redis_url=app.state.redis_url,
            )
            if not global_result.allowed:
                retry_after = max(0, global_result.reset_epoch - int(time.time()))
                return JSONResponse(
                    status_code=429,
                    content=_error_payload(
                        request,
                        code="RATE_LIMITED",
                        message="Rate limit exceeded",
                        details={
                            "bucket": "global",
                            "limit": global_result.limit,
                            "remaining": global_result.remaining,
                            "reset": global_result.reset_epoch,
                        },
                    ),
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Bucket": "global",
                        "X-RateLimit-Limit": str(global_result.limit),
                        "X-RateLimit-Remaining": str(global_result.remaining),
                        "X-RateLimit-Reset": str(global_result.reset_epoch),
                    },
                )

        bucket_result = None
        if special_bucket and special_limit > 0:
            bucket_result = app.state.rate_limiter.check(
                key=f"{special_bucket}:{rate_client_key}",
                limit=special_limit,
                redis_url=app.state.redis_url,
            )
            if not bucket_result.allowed:
                retry_after = max(0, bucket_result.reset_epoch - int(time.time()))
                headers: dict[str, str] = {
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Bucket": special_bucket,
                    "X-RateLimit-Bucket-Limit": str(bucket_result.limit),
                    "X-RateLimit-Bucket-Remaining": str(bucket_result.remaining),
                    "X-RateLimit-Bucket-Reset": str(bucket_result.reset_epoch),
                }
                if global_result is not None:
                    headers["X-RateLimit-Limit"] = str(global_result.limit)
                    headers["X-RateLimit-Remaining"] = str(global_result.remaining)
                    headers["X-RateLimit-Reset"] = str(global_result.reset_epoch)
                return JSONResponse(
                    status_code=429,
                    content=_error_payload(
                        request,
                        code="RATE_LIMITED",
                        message="Rate limit exceeded",
                        details={
                            "bucket": special_bucket,
                            "limit": bucket_result.limit,
                            "remaining": bucket_result.remaining,
                            "reset": bucket_result.reset_epoch,
                        },
                    ),
                    headers=headers,
                )

        if auth_enabled:
            if not auth:
                if (is_t2i_image_request or is_controlnet_image_request) and has_image_token:
                    return await call_next(request)
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

        if global_result is not None:
            response.headers["X-RateLimit-Limit"] = str(global_result.limit)
            response.headers["X-RateLimit-Remaining"] = str(global_result.remaining)
            response.headers["X-RateLimit-Reset"] = str(global_result.reset_epoch)
        if bucket_result is not None and special_bucket:
            response.headers["X-RateLimit-Bucket"] = str(special_bucket)
            response.headers["X-RateLimit-Bucket-Limit"] = str(bucket_result.limit)
            response.headers["X-RateLimit-Bucket-Remaining"] = str(bucket_result.remaining)
            response.headers["X-RateLimit-Bucket-Reset"] = str(bucket_result.reset_epoch)

        return response

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        started = time.time()
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        request.state.request_id = request_id

        try:
            response = await call_next(request)
        except Exception as exc:
            logger.exception(
                "Unhandled exception request_id=%s method=%s path=%s",
                request_id,
                request.method,
                request.url.path,
            )
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

        elapsed = time.time() - started
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers["X-Process-Time"] = f"{elapsed:.6f}"

        try:
            role = getattr(request.state, "auth_role", "unknown")
            logger.info(
                "request request_id=%s method=%s path=%s status=%s latency_ms=%s auth_role=%s",
                request_id,
                request.method,
                request.url.path,
                response.status_code,
                int(elapsed * 1000),
                role,
            )
        except Exception:
            pass

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
        elif isinstance(detail, dict):
            if "error" in detail and detail.get("error"):
                code = str(detail.get("error"))

            if "message" in detail and detail.get("message"):
                message = str(detail.get("message"))
            elif "detail" in detail and detail.get("detail"):
                message = str(detail.get("detail"))
            else:
                message = "HTTP request failed"

            if isinstance(detail.get("details"), dict):
                details = dict(detail.get("details") or {})
            else:
                details = {k: v for k, v in detail.items() if k not in {"error", "message"}}
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
