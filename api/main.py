# api/main.py - CharaForge T2I Lab Main API
"""
CharaForge T2I Lab 主要 API 入口
整合統一的配置、快取、效能監控和例外處理系統
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
    from core.config import get_settings, get_cache_paths, validate_cache_setup
    from core.shared_cache import bootstrap_cache
    from core.performance import get_resource_monitor, get_system_performance_summary
    from core.exceptions import (
        CharaForgeError,
        ServiceUnavailableError,
        handle_errors,
        global_error_reporter,
    )

    CORE_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Core modules import failed: {e}")
    CORE_MODULES_AVAILABLE = False

# 導入 API 路由（有容錯處理）
available_routers = {}

try:
    from api.routers import health

    available_routers["health"] = health.router
except ImportError as e:
    print(f"⚠️  Health router not available: {e}")

try:
    from api.routers import t2i

    available_routers["t2i"] = t2i.router
except ImportError as e:
    print(f"⚠️  T2I router not available: {e}")

try:
    from api.routers import finetune

    available_routers["finetune"] = finetune.router
except ImportError as e:
    print(f"⚠️  Finetune router not available: {e}")

try:
    from api.routers import batch

    available_routers["batch"] = batch.router
except ImportError as e:
    print(f"⚠️  Batch router not available: {e}")

try:
    from api.routers import export

    available_routers["export"] = export.router
except ImportError as e:
    print(f"⚠️  Export router not available: {e}")

try:
    from api.routers import safety

    available_routers["safety"] = safety.router
except ImportError as e:
    print(f"⚠️  Safety router not available: {e}")

try:
    from api.routers import monitoring

    available_routers["monitoring"] = monitoring.router
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
    logger.info("🚀 Starting CharaForge T2I Lab API...")

    startup_status = {"core_modules": False, "cache": False, "monitoring": False}

    # 1. 檢查核心模組
    if CORE_MODULES_AVAILABLE:
        startup_status["core_modules"] = True
        logger.info("✅ Core modules loaded successfully")

        # 2. 初始化共用快取
        try:
            cache = bootstrap_cache(verbose=True)
            validation = validate_cache_setup()
            startup_status["cache"] = validation.get("status") == "healthy"

            if startup_status["cache"]:
                logger.info("✅ Shared cache initialized successfully")
            else:
                logger.warning("⚠️  Cache initialization issues detected")

        except Exception as e:
            logger.error(f"❌ Cache initialization failed: {e}")
            startup_status["cache"] = False

        # 3. 啟動效能監控
        try:
            monitor = get_resource_monitor()
            startup_status["monitoring"] = True
            logger.info("✅ Performance monitoring started")
        except Exception as e:
            logger.error(f"❌ Performance monitoring failed: {e}")
            startup_status["monitoring"] = False

    else:
        logger.error("❌ Core modules not available - running in degraded mode")

    # 4. 檢查可用路由
    logger.info(f"📍 Available routers: {list(available_routers.keys())}")

    # 5. 系統狀態摘要
    app.state.startup_status = startup_status
    app.state.available_routers = list(available_routers.keys())

    if startup_status["core_modules"]:
        try:
            perf_summary = get_system_performance_summary()
            logger.info(f"💻 System Status: {perf_summary['current']['health_status']}")
        except Exception:
            pass

    logger.info("🎯 CharaForge T2I Lab API startup completed")

    yield

    # 清理
    logger.info("🛑 Shutting down CharaForge T2I Lab API...")

    if CORE_MODULES_AVAILABLE:
        try:
            from core.performance import cleanup_performance_monitoring

            cleanup_performance_monitoring()
            logger.info("✅ Performance monitoring cleaned up")
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")


# 建立 FastAPI 應用程式
app = FastAPI(
    title="CharaForge T2I Lab",
    description="Text-to-Image generation and LoRA fine-tuning API",
    version="0.1.0",
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
        api_cors_origins = "http://localhost:3000,http://127.0.0.1:3000"
        debug = False

    settings = BasicSettings()

# CORS 中介軟體
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api_cors_origins.split(","),
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
        app.include_router(router, prefix="/api/v1", tags=[router_name])
    logger.info(f"📌 Included router: {router_name}")


@app.get("/")
async def root():
    """根端點 - 系統資訊"""
    system_info = {
        "name": "CharaForge T2I Lab",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
        "health": "/healthz",
        "available_features": list(available_routers.keys()),
        "core_modules_available": CORE_MODULES_AVAILABLE,
    }

    # 添加啟動狀態資訊
    if hasattr(app.state, "startup_status"):
        system_info["startup_status"] = app.state.startup_status

    # 添加系統效能資訊（如果可用）
    if CORE_MODULES_AVAILABLE:
        try:
            perf_summary = get_system_performance_summary()
            system_info["system_health"] = perf_summary["current"]["health_status"]
            system_info["memory_usage"] = (
                f"{perf_summary['current']['memory_percent']:.1f}%"
            )

            if perf_summary["current"]["gpu_memory_total_gb"] > 0:
                gpu_usage = (
                    perf_summary["current"]["gpu_memory_used_gb"]
                    / perf_summary["current"]["gpu_memory_total_gb"]
                ) * 100
                system_info["gpu_memory_usage"] = f"{gpu_usage:.1f}%"
        except Exception:
            pass

    return system_info


@app.get("/status")
async def get_system_status():
    """系統狀態詳細資訊"""
    status = {
        "api_version": "0.1.0",
        "core_modules": CORE_MODULES_AVAILABLE,
        "available_routers": list(available_routers.keys()),
        "timestamp": "2024-01-01T00:00:00Z",  # 實際應用中應使用真實時間戳
    }

    if CORE_MODULES_AVAILABLE:
        try:
            # 效能資訊
            perf_summary = get_system_performance_summary()
            status["performance"] = perf_summary["current"]

            # 快取資訊
            cache_paths = get_cache_paths()
            status["cache"] = {
                "root": str(cache_paths.root),
                "models_dir": str(cache_paths.models),
                "datasets_dir": str(cache_paths.datasets),
                "outputs_dir": str(cache_paths.outputs),
            }

            # 錯誤統計
            error_summary = global_error_reporter.get_error_summary()
            status["errors"] = error_summary

        except Exception as e:
            status["error"] = f"Failed to get system status: {e}"

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
        from core.exceptions import CharaForgeError

        wrapped_error = CharaForgeError(
            f"Internal server error: {str(exc)}",
            "INTERNAL_ERROR",
            {"request_path": str(request.url.path)},
        )
        global_error_reporter.report_error(wrapped_error)

    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
            "debug_info": str(exc) if getattr(settings, "debug", False) else None,
        },
    )


# 回退健康檢查端點（如果健康路由不可用）
if "health" not in available_routers:

    @app.get("/healthz")
    async def fallback_health():
        """回退健康檢查"""
        status = {
            "status": "running",
            "message": "API is running but health router not available",
            "core_modules": CORE_MODULES_AVAILABLE,
            "available_features": list(available_routers.keys()),
        }

        if CORE_MODULES_AVAILABLE:
            try:
                perf_summary = get_system_performance_summary()
                status["health"] = perf_summary["current"]["health_status"]
            except Exception:
                status["health"] = "unknown"

        return status


def create_app():
    """工廠函數 - 建立應用程式實例"""
    return app


# 主程式入口
if __name__ == "__main__":
    # 載入設定
    host = getattr(settings, "api_host", "0.0.0.0")
    port = getattr(settings, "api_port", 8000)
    debug = getattr(settings, "debug", False)

    logger.info(f"🌟 Starting CharaForge T2I Lab API on {host}:{port}")
    logger.info(f"📚 API Documentation: http://{host}:{port}/docs")
    logger.info(f"❤️  Health Check: http://{host}:{port}/healthz")

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug",
    )
