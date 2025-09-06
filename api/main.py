# api/main.py - Updated FastAPI Application
"""
SagaForge T2I Lab 主要 API 入口 - 重構版
整合統一的配置、快取、路由和錯誤處理系統
"""

import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# 確保可以導入專案模組
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# 導入核心模組
try:
    from core.config import get_settings, bootstrap_config
    from core.shared_cache import bootstrap_cache
    from core.performance import get_resource_monitor, get_system_performance_summary
    from core.exceptions import CharaForgeError, handle_errors, global_error_reporter

    CORE_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Core modules import failed: {e}")
    CORE_MODULES_AVAILABLE = False

# 導入 API 路由 (有容錯處理)
available_routers = {}

try:
    from api.routers.health import router as health_router

    available_routers["health"] = health_router
except ImportError as e:
    print(f"⚠️  Health router not available: {e}")

try:
    from api.routers.t2i import router as t2i_router

    available_routers["t2i"] = t2i_router
except ImportError as e:
    print(f"⚠️  T2I router not available: {e}")

try:
    from api.routers.finetune import router as finetune_router

    available_routers["finetune"] = finetune_router
except ImportError as e:
    print(f"⚠️  Finetune router not available: {e}")

try:
    from api.routers.batch import router as batch_router

    available_routers["batch"] = batch_router
except ImportError as e:
    print(f"⚠️  Batch router not available: {e}")

try:
    from api.routers.export import router as export_router

    available_routers["export"] = export_router
except ImportError as e:
    print(f"⚠️  Export router not available: {e}")

try:
    from api.routers.safety import router as safety_router

    available_routers["safety"] = safety_router
except ImportError as e:
    print(f"⚠️  Safety router not available: {e}")

try:
    from api.routers.monitoring import router as monitoring_router

    available_routers["monitoring"] = monitoring_router
except ImportError as e:
    print(f"⚠️  Monitoring router not available: {e}")

# 設定日誌
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用程式生命週期管理"""
    logger.info("🚀 Starting SagaForge T2I Lab API...")

    startup_status = {
        "core_modules": False,
        "config": False,
        "cache": False,
        "monitoring": False,
        "celery_connection": False,
    }

    # 1. 檢查核心模組
    if CORE_MODULES_AVAILABLE:
        startup_status["core_modules"] = True
        logger.info("✅ Core modules loaded successfully")

        # 2. 初始化配置系統
        try:
            config_summary = bootstrap_config(verbose=True)
            startup_status["config"] = True
            logger.info("✅ Configuration system initialized")
        except Exception as e:
            logger.error(f"❌ Configuration initialization failed: {e}")
            startup_status["config"] = False

        # 3. 初始化共用快取
        try:
            cache = bootstrap_cache(verbose=True)
            startup_status["cache"] = True
            logger.info("✅ Shared cache initialized successfully")
        except Exception as e:
            logger.error(f"❌ Cache initialization failed: {e}")
            startup_status["cache"] = False

        # 4. 啟動效能監控
        try:
            monitor = get_resource_monitor()
            startup_status["monitoring"] = True
            logger.info("✅ Performance monitoring started")
        except Exception as e:
            logger.error(f"❌ Performance monitoring failed: {e}")
            startup_status["monitoring"] = False

        # 5. 檢查 Celery 連接
        try:
            from workers.celery_app import health_check

            celery_health = health_check()
            startup_status["celery_connection"] = celery_health["status"] == "healthy"

            if startup_status["celery_connection"]:
                logger.info("✅ Celery workers available")
            else:
                logger.warning("⚠️  Celery workers not available - async tasks disabled")
        except Exception as e:
            logger.warning(f"⚠️  Celery connection check failed: {e}")
            startup_status["celery_connection"] = False

    else:
        logger.error("❌ Core modules not available - running in degraded mode")

    # 6. 檢查可用路由
    logger.info(f"📍 Available routers: {list(available_routers.keys())}")

    # 7. 系統狀態摘要
    app.state.startup_status = startup_status
    app.state.available_routers = list(available_routers.keys())

    if startup_status["core_modules"]:
        try:
            perf_summary = get_system_performance_summary()
            logger.info(f"💻 System Status: {perf_summary['current']['health_status']}")
        except Exception:
            pass

    # 8. 系統健康狀態
    healthy_components = sum(startup_status.values())
    total_components = len(startup_status)
    health_percentage = (healthy_components / total_components) * 100

    if health_percentage >= 80:
        logger.info(
            f"🎯 SagaForge T2I Lab API startup completed ({health_percentage:.0f}% healthy)"
        )
    else:
        logger.warning(
            f"⚠️  SagaForge T2I Lab API running in degraded mode ({health_percentage:.0f}% healthy)"
        )

    yield

    # 清理
    logger.info("🛑 Shutting down SagaForge T2I Lab API...")

    if CORE_MODULES_AVAILABLE:
        try:
            from core.performance import cleanup_performance_monitoring

            cleanup_performance_monitoring()
            logger.info("✅ Performance monitoring cleaned up")
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")


# 建立 FastAPI 應用程式
app = FastAPI(
    title="SagaForge T2I Lab",
    description="Text-to-Image generation and LoRA fine-tuning API with advanced training capabilities",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# 載入設定
if CORE_MODULES_AVAILABLE:
    settings = get_settings()
else:
    # 基本設定回退
    class BasicSettings:
        api = type(
            "api",
            (),
            {
                "cors_origins": "http://localhost:3000,http://127.0.0.1:3000",
                "host": "0.0.0.0",
                "port": 8000,
            },
        )()
        debug = False

    settings = BasicSettings()

# CORS 中介軟體
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins.split(","),  # type: ignore
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 壓縮中介軟體
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 包含可用的路由
for router_name, router in available_routers.items():
    if router_name == "health":
        app.include_router(router, tags=["health"])
    else:
        app.include_router(router, tags=[router_name])


# 根路由
@app.get("/")
async def root():
    """API 根端點"""
    return {
        "name": "SagaForge T2I Lab",
        "version": "0.2.0",
        "description": "Text-to-Image generation and LoRA fine-tuning API",
        "available_endpoints": {
            "health": "/healthz",
            "docs": "/docs",
            "t2i": "/t2i/*",
            "finetune": "/finetune/*",
            "monitoring": "/monitoring/*",
        },
        "status": "running",
        "available_routers": getattr(app.state, "available_routers", []),
    }


# 健康檢查端點
@app.get("/healthz")
async def health_check():
    """系統健康檢查"""
    status = {
        "status": "healthy",
        "timestamp": "2025-01-07T00:00:00Z",  # 會被實際時間替換
        "version": "0.2.0",
        "environment": "development",
        "components": {},
    }

    if CORE_MODULES_AVAILABLE:
        try:
            from datetime import datetime

            status["timestamp"] = datetime.now().isoformat()
            status["environment"] = settings.environment  # type: ignore

            # 核心模組狀態
            startup_status = getattr(app.state, "startup_status", {})
            status["components"]["core_modules"] = startup_status.get(
                "core_modules", False
            )
            status["components"]["config"] = startup_status.get("config", False)
            status["components"]["cache"] = startup_status.get("cache", False)
            status["components"]["monitoring"] = startup_status.get("monitoring", False)
            status["components"]["celery"] = startup_status.get(
                "celery_connection", False
            )

            # 系統效能
            try:
                perf_summary = get_system_performance_summary()
                status["performance"] = perf_summary["current"]
            except Exception:
                status["performance"] = {"status": "unknown"}

            # 快取資訊
            try:
                cache = bootstrap_cache(verbose=False)
                cache_stats = cache.get_cache_stats()
                status["cache"] = {
                    "total_size_gb": cache_stats.get("total_size_gb", 0),
                    "registered_models": cache_stats.get("registered_models", {}).get(
                        "total", 0
                    ),
                }
            except Exception:
                status["cache"] = {"status": "unknown"}

            # 錯誤統計
            try:
                error_summary = global_error_reporter.get_error_summary()
                status["errors"] = error_summary
            except Exception:
                status["errors"] = {"total_errors": 0}

            # 判斷整體健康狀態
            healthy_components = sum(status["components"].values())
            total_components = len(status["components"])

            if healthy_components < total_components * 0.8:
                status["status"] = "degraded"
            elif healthy_components < total_components * 0.5:
                status["status"] = "unhealthy"

        except Exception as e:
            status["error"] = f"Failed to get system status: {e}"
            status["status"] = "error"

    return status


# 全域例外處理
@app.exception_handler(CharaForgeError)
async def charaforge_error_handler(request: Request, exc: CharaForgeError):
    """CharaForge 例外處理"""
    logger.error(f"CharaForge error: {exc}")

    # 報告錯誤
    if CORE_MODULES_AVAILABLE:
        global_error_reporter.report_error(exc)

    return JSONResponse(
        status_code=getattr(exc, "status_code", 500), content=exc.to_dict()
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP 例外處理"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP_ERROR",
            "message": exc.detail,
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """一般例外處理"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)

    # 包裝為 CharaForge 錯誤並報告
    if CORE_MODULES_AVAILABLE:
        try:
            wrapped_error = CharaForgeError(
                f"Internal server error: {str(exc)[:100]}", error_code="INTERNAL_ERROR"
            )
            global_error_reporter.report_error(wrapped_error)
        except Exception:
            pass

    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
            "detail": str(exc)[:200] if settings.debug else "Internal server error",
        },
    )


# 中介軟體：請求日誌
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """記錄 HTTP 請求"""
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time

    logger.info(
        f"{request.method} {request.url.path} "
        f"- {response.status_code} "
        f"- {process_time:.3f}s"
    )

    return response


# CLI 運行支援
def create_app():
    """創建應用程式實例 (供 ASGI 伺服器使用)"""
    return app


if __name__ == "__main__":
    # 直接運行開發伺服器
    import time

    if CORE_MODULES_AVAILABLE:
        host = settings.api.host  # type: ignore
        port = settings.api.port  # type: ignore
        debug = settings.debug
    else:
        host = "0.0.0.0"
        port = 8000
        debug = True

    print(f"🚀 Starting SagaForge T2I Lab API on {host}:{port}")
    print(f"📖 API docs available at: http://{host}:{port}/docs")

    uvicorn.run("api.main:app", host=host, port=port, reload=debug, log_level="info")
