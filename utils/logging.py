# utils/logging.py
"""
統一日誌系統 - 支援結構化日誌、效能監控、多模組管理
"""

import os
import logging
import time
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from functools import wraps

# 日誌設定
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class StructuredFormatter(logging.Formatter):
    """結構化日誌格式器"""

    def format(self, record: logging.LogRecord) -> str:
        # 基本資訊
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }

        # 添加額外資料
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        # 效能資料
        if hasattr(record, "duration"):
            log_data["duration_ms"] = round(record.duration * 1000, 2)

        # GPU 資訊
        if hasattr(record, "gpu_memory"):
            log_data["gpu_memory_mb"] = record.gpu_memory

        return json.dumps(log_data, ensure_ascii=False)


class PerformanceLogger:
    """效能監控日誌器"""

    def __init__(self, logger_name: str = "performance"):
        self.logger = logging.getLogger(logger_name)
        self.operations: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    @contextmanager
    def measure(self, operation: str, **kwargs):
        """使用 context manager 測量執行時間"""
        start_time = time.time()
        gpu_memory_start = self._get_gpu_memory()

        try:
            yield
        finally:
            duration = time.time() - start_time
            gpu_memory_end = self._get_gpu_memory()

            self.log_operation(
                operation,
                duration,
                gpu_memory_start=gpu_memory_start,
                gpu_memory_end=gpu_memory_end,
                **kwargs,
            )

    def log_operation(self, operation: str, duration: float, **kwargs):
        """記錄操作效能"""
        with self._lock:
            perf_data = {
                "operation": operation,
                "duration": duration,
                "timestamp": datetime.now().isoformat(),
                **kwargs,
            }

            self.operations.append(perf_data)

            # 記錄到日誌
            extra = logging.LogRecord.__dict__.copy()
            extra.update({"duration": duration, "extra_data": perf_data})

            if duration > 1.0:  # 超過 1 秒的操作
                self.logger.warning(f"Slow operation: {operation}", extra=extra)
            else:
                self.logger.info(f"Operation: {operation}", extra=extra)

    def get_stats(self) -> Dict[str, Any]:
        """取得效能統計"""
        if not self.operations:
            return {"total_operations": 0}

        durations = [op["duration"] for op in self.operations]

        return {
            "total_operations": len(self.operations),
            "avg_duration": sum(durations) / len(durations),
            "max_duration": max(durations),
            "min_duration": min(durations),
            "total_time": sum(durations),
            "slow_operations": len([d for d in durations if d > 1.0]),
        }

    def _get_gpu_memory(self) -> Optional[float]:
        """取得 GPU 記憶體使用量 (MB)"""
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024
        except:
            pass
        return None


def performance_monitor(operation_name: str = None):
    """效能監控裝飾器"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            perf_logger = PerformanceLogger()

            with perf_logger.measure(op_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    structured: bool = False,
    console: bool = True,
) -> logging.Logger:
    """設定日誌器"""

    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVELS.get(level.upper(), logging.INFO))

    # 避免重複添加 handler
    if logger.handlers:
        return logger

    # 選擇格式器
    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # 控制台輸出
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 檔案輸出
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """取得已設定的日誌器"""
    logger = logging.getLogger(name)

    # 如果沒有 handler，使用預設設定
    if not logger.handlers:
        return setup_logger(name)

    return logger


class ModuleLogger:
    """模組專用日誌器"""

    def __init__(self, module_name: str, cache_root: str = None):
        self.module_name = module_name
        self.cache_root = cache_root or os.getenv(
            "AI_CACHE_ROOT", "../ai_warehouse/cache"
        )

        # 建立日誌目錄
        log_dir = Path(self.cache_root) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # 設定日誌檔案
        log_file = log_dir / f"{module_name}.log"

        self.logger = setup_logger(
            module_name,
            level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=str(log_file),
            structured=os.getenv("STRUCTURED_LOGS", "false").lower() == "true",
        )

        self.perf_logger = PerformanceLogger(f"{module_name}.performance")

    def info(self, message: str, **kwargs):
        """資訊日誌"""
        if kwargs:
            extra = logging.LogRecord.__dict__.copy()
            extra.update({"extra_data": kwargs})
            self.logger.info(message, extra=extra)
        else:
            self.logger.info(message)

    def warning(self, message: str, **kwargs):
        """警告日誌"""
        if kwargs:
            extra = logging.LogRecord.__dict__.copy()
            extra.update({"extra_data": kwargs})
            self.logger.warning(message, extra=extra)
        else:
            self.logger.warning(message)

    def error(self, message: str, **kwargs):
        """錯誤日誌"""
        if kwargs:
            extra = logging.LogRecord.__dict__.copy()
            extra.update({"extra_data": kwargs})
            self.logger.error(message, extra=extra)
        else:
            self.logger.error(message)

    def debug(self, message: str, **kwargs):
        """除錯日誌"""
        if kwargs:
            extra = logging.LogRecord.__dict__.copy()
            extra.update({"extra_data": kwargs})
            self.logger.debug(message, extra=extra)
        else:
            self.logger.debug(message)

    @contextmanager
    def measure_performance(self, operation: str, **kwargs):
        """測量效能"""
        with self.perf_logger.measure(operation, **kwargs):
            yield

    def get_performance_stats(self) -> Dict[str, Any]:
        """取得效能統計"""
        return self.perf_logger.get_stats()


# 全域函數
def log_system_info():
    """記錄系統資訊"""
    logger = get_logger("system")

    try:
        import torch
        import psutil

        # GPU 資訊
        gpu_info = {
            "available": torch.cuda.is_available(),
            "count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }

        if torch.cuda.is_available():
            gpu_info["devices"] = [
                {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory
                    / 1024**3,
                }
                for i in range(torch.cuda.device_count())
            ]

        # 系統資訊
        system_info = {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024**3,
            "gpu": gpu_info,
        }

        logger.info("System information", extra={"extra_data": system_info})

    except Exception as e:
        logger.error(f"Failed to log system info: {e}")


# 預設日誌器
default_logger = setup_logger("sagaforge")

# 模組初始化
if __name__ != "__main__":
    log_system_info()
