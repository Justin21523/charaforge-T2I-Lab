# utils/__init__.py
"""
SagaForge T2I Lab - 統一工具模組
提供跨模組共用的核心功能
"""

import os
import pathlib
import logging
from typing import Dict, Any, Optional
import torch

# 共享快取初始化 (每個模組都需要)
AI_CACHE_ROOT = os.getenv("AI_CACHE_ROOT", "../ai_warehouse/cache")
for k, v in {
    "HF_HOME": f"{AI_CACHE_ROOT}/hf",
    "TRANSFORMERS_CACHE": f"{AI_CACHE_ROOT}/hf/transformers",
    "HF_DATASETS_CACHE": f"{AI_CACHE_ROOT}/hf/datasets",
    "HUGGINGFACE_HUB_CACHE": f"{AI_CACHE_ROOT}/hf/hub",
    "TORCH_HOME": f"{AI_CACHE_ROOT}/torch",
}.items():
    os.environ[k] = v
    pathlib.Path(v).mkdir(parents=True, exist_ok=True)

# 模組匯入
from .logging import setup_logger, get_logger, PerformanceLogger
from .file_operations import (
    SafeFileHandler,
    validate_path,
    ensure_directory,
    get_file_hash,
    safe_json_load,
    safe_json_save,
)
from .security import TokenManager, ContentValidator, secure_filename, sanitize_input
from .calculator import (
    MemoryCalculator,
    PerformanceMonitor,
    estimate_vram_usage,
    optimize_batch_size,
)
from .web_search import SearchManager, validate_url

# 全域實例 (單例模式)
_token_manager: Optional[TokenManager] = None
_performance_monitor: Optional[PerformanceMonitor] = None
_safe_file_handler: Optional[SafeFileHandler] = None


def get_token_manager() -> TokenManager:
    """取得全域 Token 管理器"""
    global _token_manager
    if _token_manager is None:
        _token_manager = TokenManager()
    return _token_manager


def get_performance_monitor() -> PerformanceMonitor:
    """取得全域效能監控器"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def get_file_handler() -> SafeFileHandler:
    """取得全域檔案處理器"""
    global _safe_file_handler
    if _safe_file_handler is None:
        _safe_file_handler = SafeFileHandler(base_path=AI_CACHE_ROOT)
    return _safe_file_handler


# 快速存取函數
def log_performance(func_name: str, duration: float, **kwargs) -> None:
    """記錄效能資訊"""
    monitor = get_performance_monitor()
    monitor.log_operation(func_name, duration, **kwargs)


def get_cache_info() -> Dict[str, Any]:
    """取得快取資訊"""
    return {
        "cache_root": AI_CACHE_ROOT,
        "hf_cache": os.environ.get("HF_HOME"),
        "torch_cache": os.environ.get("TORCH_HOME"),
    }


# 初始化檢查
def validate_utils_setup() -> Dict[str, Any]:
    """驗證 utils 模組設定"""
    import torch

    result = {
        "cache_root": AI_CACHE_ROOT,
        "cache_accessible": pathlib.Path(AI_CACHE_ROOT).exists(),
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "modules_loaded": True,
    }

    try:
        # 測試各模組是否正常載入
        get_token_manager()
        get_performance_monitor()
        get_file_handler()
        result["singletons_ready"] = True
    except Exception as e:
        result["singletons_ready"] = False
        result["error"] = str(e)

    return result


# 版本資訊
__version__ = "0.2.0"
__all__ = [
    # 日誌
    "setup_logger",
    "get_logger",
    "PerformanceLogger",
    # 檔案操作
    "SafeFileHandler",
    "validate_path",
    "ensure_directory",
    "get_file_hash",
    "safe_json_load",
    "safe_json_save",
    # 安全性
    "TokenManager",
    "ContentValidator",
    "secure_filename",
    "sanitize_input",
    # 計算
    "MemoryCalculator",
    "PerformanceMonitor",
    "estimate_vram_usage",
    "optimize_batch_size",
    # 網路搜尋
    "SearchManager",
    "validate_url",
    # 全域存取
    "get_token_manager",
    "get_performance_monitor",
    "get_file_handler",
    "log_performance",
    "get_cache_info",
    "validate_utils_setup",
]

# 模組初始化日誌
logger = setup_logger("utils", level=logging.INFO)
logger.info(f"[cache] {AI_CACHE_ROOT} | GPU: {torch.cuda.is_available()}")
logger.info(f"SagaForge utils v{__version__} 載入完成")
