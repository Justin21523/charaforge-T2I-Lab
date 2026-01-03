"""
CharaForge T2I Lab - FastAPI entrypoint

API contract:
- All public endpoints are mounted under `/api/v1/*`.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.jwt_tokens import verify_access_token
from api.routers.health import router as health_router
from api.security import (
    RateLimiter,
    extract_api_key,
    get_client_key,
    is_exempt_v1_request,
    parse_api_keys,
    resolve_api_key,
    scope_allows,
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


def _required_scope(path: str, method: str) -> str | None:
    if path == f"{API_V1_PREFIX}/metrics":
        return "metrics:read"
    if path.startswith(f"{API_V1_PREFIX}/models/scan"):
        return "models:scan"
    if path.startswith(f"{API_V1_PREFIX}/models/"):
        return "models:read"
    if path == f"{API_V1_PREFIX}/auth/ws_ticket":
        return "train:manage"
    if path.startswith(f"{API_V1_PREFIX}/auth/keys"):
        return "auth:manage"
    if path.startswith(f"{API_V1_PREFIX}/t2i/"):
        if path.startswith(f"{API_V1_PREFIX}/t2i/images/") or path.startswith(
            f"{API_V1_PREFIX}/t2i/status/"
        ):
            return "t2i:read"
        if path.startswith(f"{API_V1_PREFIX}/t2i/jobs"):
            if method in {"GET", "HEAD"}:
                return "t2i:read"
            return "t2i:manage"
        if path.startswith(f"{API_V1_PREFIX}/t2i/cancel/"):
            return "t2i:cancel"
        return "t2i:generate"
    if path.startswith(f"{API_V1_PREFIX}/controlnet/"):
        if path.startswith(f"{API_V1_PREFIX}/controlnet/images/") or path == (
            f"{API_V1_PREFIX}/controlnet/types"
        ):
            return "controlnet:read"
        return "controlnet:generate"
    if path.startswith(f"{API_V1_PREFIX}/upload"):
        if method in {"GET", "HEAD"}:
            return "upload:read"
        return "upload:write"
    if path.startswith(f"{API_V1_PREFIX}/datasets"):
        if method in {"GET", "HEAD"}:
            return "datasets:read"
        return "datasets:write"
    if path.startswith(f"{API_V1_PREFIX}/lora"):
        if method in {"GET", "HEAD"}:
            return "lora:read"
        return "lora:manage"
    if path.startswith(f"{API_V1_PREFIX}/finetune"):
        return "train:manage"
    if path.startswith(f"{API_V1_PREFIX}/batch"):
        return "batch:manage"
    return None


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
    scan_manager = getattr(app.state, "model_scan_job_manager", None)
    try:
        from api.model_scan_jobs import ModelScanJobManager

        if isinstance(scan_manager, ModelScanJobManager):
            scan_manager.shutdown()
    except Exception:
        pass


def create_app() -> FastAPI:
    settings = get_settings()
    json_logs = os.getenv("LOG_JSON", "").strip().lower() in {"1", "true", "yes", "on"}
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

    sentry_dsn = os.getenv("SENTRY_DSN", "").strip()
    if sentry_dsn:
        try:
            import sentry_sdk  # type: ignore
            from sentry_sdk.integrations.asgi import SentryAsgiMiddleware  # type: ignore

            sentry_sdk.init(
                dsn=sentry_dsn,
                environment=getattr(settings, "environment", None) or "development",
                traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0") or 0.0),
            )
            app.add_middleware(SentryAsgiMiddleware)
        except Exception as exc:
            logger.warning("Sentry disabled: %s", exc)

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

        dispatch_mode = str(settings.api.t2i_dispatch_mode or "redis").lower()
        worker_enabled = bool(settings.api.t2i_worker_enabled) and dispatch_mode != "celery"

        app.state.t2i_job_manager = T2IJobManager(
            redis_url=app.state.redis_url,
            worker_enabled=worker_enabled,
            dispatch_mode=dispatch_mode,
            job_ttl_seconds=int(settings.api.t2i_job_ttl_seconds or 0),
            stale_seconds=int(settings.api.t2i_job_stale_seconds or 0),
            max_attempts=int(settings.api.t2i_job_max_attempts or 1),
            max_concurrent_per_owner=int(settings.api.t2i_max_concurrent or 1),
            max_global_concurrent=int(settings.api.t2i_max_global_concurrent or 0),
        )
    except Exception as exc:
        logger.warning("T2I async jobs disabled: %s", exc)

    try:
        from api.model_scan_jobs import ModelScanJobManager

        app.state.model_scan_job_manager = ModelScanJobManager(
            redis_url=app.state.redis_url,
            worker_enabled=bool(settings.api.models_scan_worker_enabled),
            job_ttl_seconds=int(settings.api.models_scan_job_ttl_seconds or 0),
        )
    except Exception as exc:
        logger.warning("Model scan jobs disabled: %s", exc)

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

    prometheus_enabled = os.getenv("PROMETHEUS_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if prometheus_enabled:
        try:
            from api.metrics import Metrics

            app.state.metrics = Metrics()

            @app.get(f"{API_V1_PREFIX}/metrics", include_in_schema=False)
            async def metrics_endpoint(request: Request):
                metrics = getattr(request.app.state, "metrics", None)
                if metrics is None:
                    return PlainTextResponse("", status_code=404)

                body = metrics.render_prometheus()
                try:
                    manager = getattr(request.app.state, "t2i_job_manager", None)
                    queued = 0
                    running = 0
                    if manager is not None and hasattr(manager, "global_counts"):
                        counts = manager.global_counts()
                        queued = int(counts.get("queued", 0))
                        running = int(counts.get("running", 0))
                        active = queued + running
                        max_queue = int(settings.api.t2i_max_global_queue or 0)
                        body += (
                            "# HELP charaforge_t2i_jobs_queued Queued T2I jobs.\n"
                            "# TYPE charaforge_t2i_jobs_queued gauge\n"
                            f"charaforge_t2i_jobs_queued {queued}\n"
                            "# HELP charaforge_t2i_jobs_running Running T2I jobs.\n"
                            "# TYPE charaforge_t2i_jobs_running gauge\n"
                            f"charaforge_t2i_jobs_running {running}\n"
                            "# HELP charaforge_t2i_jobs_active Active (queued+running) T2I jobs.\n"
                            "# TYPE charaforge_t2i_jobs_active gauge\n"
                            f"charaforge_t2i_jobs_active {active}\n"
                            "# HELP charaforge_t2i_jobs_queue_max Max allowed active T2I jobs (0=unlimited).\n"
                            "# TYPE charaforge_t2i_jobs_queue_max gauge\n"
                            f"charaforge_t2i_jobs_queue_max {max_queue}\n"
                        )
                    if manager is not None and hasattr(manager, "global_slot_usage"):
                        usage = manager.global_slot_usage()
                        used = int(usage.get("used", 0))
                        total = int(usage.get("max", 0))
                        if total <= 0:
                            total = int(settings.api.t2i_max_global_concurrent or 0)
                            used = running
                        body += (
                            "# HELP charaforge_t2i_gpu_slots_used Used T2I GPU slots.\n"
                            "# TYPE charaforge_t2i_gpu_slots_used gauge\n"
                            f"charaforge_t2i_gpu_slots_used {used}\n"
                            "# HELP charaforge_t2i_gpu_slots_total Configured T2I GPU slots.\n"
                            "# TYPE charaforge_t2i_gpu_slots_total gauge\n"
                            f"charaforge_t2i_gpu_slots_total {total}\n"
                        )

                    scan_manager = getattr(request.app.state, "model_scan_job_manager", None)
                    if scan_manager is not None and hasattr(scan_manager, "global_counts"):
                        scan_counts = scan_manager.global_counts()
                        scan_queued = int(scan_counts.get("queued", 0))
                        scan_running = int(scan_counts.get("running", 0))
                        scan_active = scan_queued + scan_running
                        scan_lease = int(scan_counts.get("lease_active", 0))
                        body += (
                            "# HELP charaforge_models_scan_jobs_queued Queued model scan jobs.\n"
                            "# TYPE charaforge_models_scan_jobs_queued gauge\n"
                            f"charaforge_models_scan_jobs_queued {scan_queued}\n"
                            "# HELP charaforge_models_scan_jobs_running Running model scan jobs.\n"
                            "# TYPE charaforge_models_scan_jobs_running gauge\n"
                            f"charaforge_models_scan_jobs_running {scan_running}\n"
                            "# HELP charaforge_models_scan_jobs_active Active (queued+running) model scan jobs.\n"
                            "# TYPE charaforge_models_scan_jobs_active gauge\n"
                            f"charaforge_models_scan_jobs_active {scan_active}\n"
                            "# HELP charaforge_models_scan_active_lease_present Active scan lease present.\n"
                            "# TYPE charaforge_models_scan_active_lease_present gauge\n"
                            f"charaforge_models_scan_active_lease_present {scan_lease}\n"
                            "# HELP charaforge_models_scan_jobs_queue_max Max allowed active scan jobs.\n"
                            "# TYPE charaforge_models_scan_jobs_queue_max gauge\n"
                            "charaforge_models_scan_jobs_queue_max 1\n"
                        )
                        if hasattr(scan_manager, "completed_counts"):
                            completed = scan_manager.completed_counts()
                            succeeded = int(completed.get("succeeded", 0))
                            failed = int(completed.get("failed", 0))
                            canceled = int(completed.get("canceled", 0))
                            body += (
                                "# HELP charaforge_models_scan_jobs_completed_total Completed model scan jobs.\n"
                                "# TYPE charaforge_models_scan_jobs_completed_total counter\n"
                                'charaforge_models_scan_jobs_completed_total{result="succeeded"} '
                                f"{succeeded}\n"
                                'charaforge_models_scan_jobs_completed_total{result="failed"} '
                                f"{failed}\n"
                                'charaforge_models_scan_jobs_completed_total{result="canceled"} '
                                f"{canceled}\n"
                            )
                except Exception:
                    pass

                return PlainTextResponse(body, media_type="text/plain; version=0.0.4")
        except Exception as exc:
            logger.warning("Prometheus metrics disabled: %s", exc)

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

        allow_anonymous = path in {
            f"{API_V1_PREFIX}/auth/refresh",
            f"{API_V1_PREFIX}/auth/logout",
        }

        is_auth_token_request = path == f"{API_V1_PREFIX}/auth/token"
        is_auth_refresh_request = path == f"{API_V1_PREFIX}/auth/refresh"
        is_scan_trigger_request = path in {
            f"{API_V1_PREFIX}/models/scan",
            f"{API_V1_PREFIX}/models/scan/submit",
        }
        is_t2i_image_request = path.startswith(f"{API_V1_PREFIX}/t2i/images/")
        is_controlnet_image_request = path.startswith(f"{API_V1_PREFIX}/controlnet/images/")
        has_image_token = bool(request.query_params.get("img_token"))

        request.state.auth_source = "anonymous"
        request.state.auth_role = "anonymous"
        request.state.auth_scopes = set()
        request.state.api_key_id = None
        request.state.client_key = get_client_key(request, api_key=None)

        authorization = request.headers.get("Authorization", "").strip()
        if authorization.lower().startswith("bearer "):
            token = authorization.split(" ", 1)[1].strip()
            payload = verify_access_token(token)
            if not payload:
                return JSONResponse(
                    status_code=401,
                    content=_error_payload(
                        request,
                        code="INVALID_TOKEN",
                        message="Unauthorized",
                    ),
                )

            role = str(payload.get("role") or "user")
            scopes = payload.get("scopes") or []
            if isinstance(scopes, str):
                scopes = [scopes]
            scope_set = {str(s).strip() for s in scopes if str(s).strip()}
            subject = str(payload.get("sub") or "")
            if not subject:
                return JSONResponse(
                    status_code=401,
                    content=_error_payload(
                        request,
                        code="INVALID_TOKEN",
                        message="Unauthorized",
                    ),
                )

            request.state.auth_source = "jwt"
            request.state.auth_role = role or "user"
            request.state.auth_scopes = scope_set
            request.state.api_key_id = payload.get("kid")
            request.state.client_key = subject
            if request.state.api_key_id and api_key_store is not None:
                try:
                    api_key_store.mark_used(str(request.state.api_key_id))
                except Exception:
                    pass
        else:
            presented = extract_api_key(request, header_name)
            auth = resolve_api_key(
                presented,
                admin_keys=admin_keys,
                user_keys=user_keys,
                key_store=api_key_store,
            )
            role = auth.role if auth else None
            request.state.auth_source = "api_key" if auth else "anonymous"
            request.state.auth_role = role or "anonymous"
            request.state.auth_scopes = auth.scopes if auth else set()
            request.state.api_key_id = auth.key_id if auth else None
            request.state.client_key = get_client_key(
                request, api_key=presented if auth else None
            )
            if auth and auth.key_id and api_key_store is not None:
                try:
                    api_key_store.mark_used(auth.key_id)
                except Exception:
                    pass

        metrics = getattr(request.app.state, "metrics", None)

        rate_client_key = request.state.client_key
        global_rate_limit = int(settings.api.rate_limit or 0)

        special_bucket = None
        special_limit = 0
        if is_auth_token_request:
            special_bucket = "auth_token"
            special_limit = int(settings.api.auth_token_rate_limit or 0)
        elif is_auth_refresh_request:
            special_bucket = "auth_refresh"
            special_limit = int(settings.api.auth_refresh_rate_limit or 0)
        elif is_scan_trigger_request:
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
                if metrics is not None:
                    try:
                        metrics.inc_rate_limited(bucket="global")
                    except Exception:
                        pass
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
                if metrics is not None:
                    try:
                        metrics.inc_rate_limited(bucket=str(special_bucket))
                    except Exception:
                        pass
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

        authenticated = getattr(request.state, "auth_role", "anonymous") != "anonymous"
        if auth_enabled:
            if not authenticated and not allow_anonymous:
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

            if authenticated:
                required_scope = _required_scope(path, request.method)
                if required_scope:
                    scopes = getattr(request.state, "auth_scopes", set()) or set()
                    if scopes:
                        if not scope_allows(scopes, required_scope):
                            return JSONResponse(
                                status_code=403,
                                content=_error_payload(
                                    request,
                                    code="INSUFFICIENT_SCOPE",
                                    message="Forbidden",
                                    details={"required": required_scope},
                                ),
                            )
                    elif required_scope in {"models:scan", "metrics:read"} and getattr(request.state, "auth_role", "") != "admin":
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

        metrics = getattr(request.app.state, "metrics", None)
        if metrics is not None:
            try:
                metrics.inc_in_flight()
            except Exception:
                pass

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
            route = request.scope.get("route")
            route_path = getattr(route, "path", request.url.path)
        except Exception:
            route_path = request.url.path

        if metrics is not None:
            try:
                metrics.observe_request(
                    method=request.method,
                    route=str(route_path),
                    status_code=int(response.status_code),
                    duration_s=elapsed,
                )
            except Exception:
                pass
            try:
                metrics.dec_in_flight()
            except Exception:
                pass

        try:
            role = getattr(request.state, "auth_role", "unknown")
            api_key_id = getattr(request.state, "api_key_id", None)
            if json_logs:
                logger.info(
                    json.dumps(
                        {
                            "event": "request",
                            "request_id": request_id,
                            "method": request.method,
                            "path": request.url.path,
                            "route": str(route_path),
                            "status": int(response.status_code),
                            "latency_ms": int(elapsed * 1000),
                            "auth_role": role,
                            "api_key_id": api_key_id,
                        },
                        ensure_ascii=False,
                    )
                )
            else:
                logger.info(
                    "request request_id=%s method=%s path=%s status=%s latency_ms=%s auth_role=%s api_key_id=%s",
                    request_id,
                    request.method,
                    request.url.path,
                    response.status_code,
                    int(elapsed * 1000),
                    role,
                    api_key_id,
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
