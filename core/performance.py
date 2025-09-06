# core/performance.py - Performance Monitoring & Optimization
"""
效能監控與最佳化系統
提供記憶體監控、GPU 使用率追蹤、模型快取最佳化等功能
"""

import time
import gc
import threading
import asyncio
import hashlib
import json
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from collections import defaultdict
import psutil

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """效能指標資料類別"""

    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_available_gb: float = 0.0
    gpu_memory_allocated_gb: float = 0.0
    gpu_memory_reserved_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    gpu_utilization_percent: float = 0.0

    @property
    def gpu_memory_free_gb(self) -> float:
        return max(0, self.gpu_memory_total_gb - self.gpu_memory_reserved_gb)

    @property
    def memory_pressure(self) -> str:
        """記憶體壓力等級"""
        if self.memory_percent > 90:
            return "critical"
        elif self.memory_percent > 80:
            return "high"
        elif self.memory_percent > 60:
            return "medium"
        else:
            return "low"

    @property
    def gpu_memory_pressure(self) -> str:
        """GPU 記憶體壓力等級"""
        if self.gpu_memory_total_gb == 0:
            return "n/a"

        usage_percent = (self.gpu_memory_reserved_gb / self.gpu_memory_total_gb) * 100
        if usage_percent > 95:
            return "critical"
        elif usage_percent > 85:
            return "high"
        elif usage_percent > 60:
            return "medium"
        else:
            return "low"


@dataclass
class CacheMetrics:
    """快取效能指標"""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total_requests if self.total_requests > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        return 1.0 - self.hit_rate


class ResourceMonitor:
    """系統資源監控器"""

    def __init__(
        self, collection_interval: float = 5.0, max_history_points: int = 720
    ):  # 1小時的歷史記錄 (5s * 720 = 3600s)
        self.collection_interval = collection_interval
        self.max_history_points = max_history_points

        self.metrics_history: List[PerformanceMetrics] = []
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # 警報閾值
        self.alert_thresholds = {
            "memory_percent": 85.0,
            "gpu_memory_percent": 90.0,
            "cpu_percent": 90.0,
        }

        self.alert_callbacks: List[Callable] = []

    def start_monitoring(self):
        """開始監控"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitor_thread.start()
        logger.info("Resource monitoring started")

    def stop_monitoring(self):
        """停止監控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Resource monitoring stopped")

    def _monitoring_loop(self):
        """監控主迴圈"""
        while self.is_monitoring:
            try:
                metrics = self.collect_current_metrics()

                with self._lock:
                    self.metrics_history.append(metrics)

                    # 限制歷史記錄長度
                    if len(self.metrics_history) > self.max_history_points:
                        self.metrics_history.pop(0)

                # 檢查警報
                self._check_alerts(metrics)

                time.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.collection_interval)

    def collect_current_metrics(self) -> PerformanceMetrics:
        """收集當前效能指標"""
        metrics = PerformanceMetrics()

        try:
            # CPU 和記憶體
            metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            metrics.memory_percent = memory.percent
            metrics.memory_used_gb = (memory.total - memory.available) / (1024**3)
            metrics.memory_available_gb = memory.available / (1024**3)

            # GPU 資訊
            try:
                import torch

                if torch.cuda.is_available():
                    metrics.gpu_memory_allocated_gb = torch.cuda.memory_allocated() / (
                        1024**3
                    )
                    metrics.gpu_memory_reserved_gb = torch.cuda.memory_reserved() / (
                        1024**3
                    )
                    metrics.gpu_memory_total_gb = torch.cuda.get_device_properties(
                        0
                    ).total_memory / (1024**3)

                    # GPU 使用率 (需要 nvidia-ml-py 套件)
                    try:
                        import pynvml

                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        metrics.gpu_utilization_percent = utilization.gpu  # type: ignore
                    except ImportError:
                        pass

            except ImportError:
                pass

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")

        return metrics

    def get_current_metrics(self) -> PerformanceMetrics:
        """取得最新的效能指標"""
        if self.metrics_history:
            with self._lock:
                return self.metrics_history[-1]
        else:
            return self.collect_current_metrics()

    def get_metrics_history(
        self, minutes: Optional[int] = None
    ) -> List[PerformanceMetrics]:
        """取得歷史效能指標"""
        with self._lock:
            if minutes is None:
                return self.metrics_history.copy()

            # 取得指定分鐘數內的記錄
            cutoff_time = time.time() - (minutes * 60)
            return [m for m in self.metrics_history if m.timestamp >= cutoff_time]

    def get_performance_summary(self, minutes: int = 30) -> Dict[str, Any]:
        """取得效能摘要"""
        history = self.get_metrics_history(minutes)

        if not history:
            return {"error": "No metrics available"}

        # 計算統計資料
        cpu_values = [m.cpu_percent for m in history]
        memory_values = [m.memory_percent for m in history]
        gpu_memory_values = [
            m.gpu_memory_reserved_gb for m in history if m.gpu_memory_total_gb > 0
        ]

        summary = {
            "period_minutes": minutes,
            "data_points": len(history),
            "cpu": {
                "current": cpu_values[-1] if cpu_values else 0,
                "average": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                "max": max(cpu_values) if cpu_values else 0,
                "min": min(cpu_values) if cpu_values else 0,
            },
            "memory": {
                "current_percent": memory_values[-1] if memory_values else 0,
                "average_percent": (
                    sum(memory_values) / len(memory_values) if memory_values else 0
                ),
                "max_percent": max(memory_values) if memory_values else 0,
                "current_gb": history[-1].memory_used_gb if history else 0,
                "available_gb": history[-1].memory_available_gb if history else 0,
            },
        }

        # GPU 摘要
        if gpu_memory_values:
            latest = history[-1]
            summary["gpu"] = {
                "memory_allocated_gb": latest.gpu_memory_allocated_gb,
                "memory_reserved_gb": latest.gpu_memory_reserved_gb,
                "memory_total_gb": latest.gpu_memory_total_gb,
                "memory_free_gb": latest.gpu_memory_free_gb,
                "utilization_percent": latest.gpu_utilization_percent,
                "memory_pressure": latest.gpu_memory_pressure,
            }
        else:
            summary["gpu"] = {"available": False}

        # 系統健康狀態
        current = history[-1]
        summary["health"] = {
            "status": self._determine_health_status(current),
            "memory_pressure": current.memory_pressure,
            "gpu_memory_pressure": current.gpu_memory_pressure,
            "warnings": self._generate_warnings(current),
        }

        return summary

    def _determine_health_status(self, metrics: PerformanceMetrics) -> str:
        """判斷系統健康狀態"""
        if metrics.memory_percent > 95 or (
            metrics.gpu_memory_total_gb > 0
            and (metrics.gpu_memory_reserved_gb / metrics.gpu_memory_total_gb) > 0.98
        ):
            return "critical"
        elif metrics.memory_percent > 85 or (
            metrics.gpu_memory_total_gb > 0
            and (metrics.gpu_memory_reserved_gb / metrics.gpu_memory_total_gb) > 0.90
        ):
            return "warning"
        elif metrics.memory_percent > 70 or (
            metrics.gpu_memory_total_gb > 0
            and (metrics.gpu_memory_reserved_gb / metrics.gpu_memory_total_gb) > 0.80
        ):
            return "moderate"
        else:
            return "healthy"

    def _generate_warnings(self, metrics: PerformanceMetrics) -> List[str]:
        """產生警告訊息"""
        warnings = []

        if metrics.memory_percent > 90:
            warnings.append(f"High memory usage: {metrics.memory_percent:.1f}%")

        if metrics.gpu_memory_total_gb > 0:
            gpu_usage_percent = (
                metrics.gpu_memory_reserved_gb / metrics.gpu_memory_total_gb
            ) * 100
            if gpu_usage_percent > 95:
                warnings.append(f"Critical GPU memory usage: {gpu_usage_percent:.1f}%")
            elif gpu_usage_percent > 85:
                warnings.append(f"High GPU memory usage: {gpu_usage_percent:.1f}%")

        if metrics.cpu_percent > 90:
            warnings.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")

        return warnings

    def _check_alerts(self, metrics: PerformanceMetrics):
        """檢查並觸發警報"""
        alerts = []

        if metrics.memory_percent > self.alert_thresholds["memory_percent"]:
            alerts.append(f"Memory usage alert: {metrics.memory_percent:.1f}%")

        if metrics.gpu_memory_total_gb > 0:
            gpu_usage = (
                metrics.gpu_memory_reserved_gb / metrics.gpu_memory_total_gb
            ) * 100
            if gpu_usage > self.alert_thresholds["gpu_memory_percent"]:
                alerts.append(f"GPU memory usage alert: {gpu_usage:.1f}%")

        if metrics.cpu_percent > self.alert_thresholds["cpu_percent"]:
            alerts.append(f"CPU usage alert: {metrics.cpu_percent:.1f}%")

        # 觸發警報回調
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert, metrics)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")

    def add_alert_callback(self, callback: Callable):
        """添加警報回調函數"""
        self.alert_callbacks.append(callback)

    def cleanup_old_metrics(self, keep_hours: int = 24):
        """清理舊的效能指標"""
        cutoff_time = time.time() - (keep_hours * 3600)

        with self._lock:
            original_count = len(self.metrics_history)
            self.metrics_history = [
                m for m in self.metrics_history if m.timestamp >= cutoff_time
            ]
            removed_count = original_count - len(self.metrics_history)

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old performance metrics")


class MemoryOptimizer:
    """記憶體最佳化器"""

    @staticmethod
    def cleanup_gpu_memory():
        """清理 GPU 記憶體"""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # 強制垃圾回收
                gc.collect()

                logger.debug("GPU memory cleaned up")
        except ImportError:
            pass

    @staticmethod
    def cleanup_system_memory():
        """清理系統記憶體"""
        gc.collect()
        logger.debug("System memory cleaned up")

    @staticmethod
    def get_memory_pressure() -> Dict[str, Any]:
        """取得記憶體壓力資訊"""
        pressure_info = {
            "system_memory_gb": 0,
            "system_memory_percent": 0,
            "gpu_memory_gb": 0,
            "gpu_memory_percent": 0,
            "recommendation": "normal",
        }

        try:
            # 系統記憶體
            memory = psutil.virtual_memory()
            pressure_info["system_memory_gb"] = memory.used / (1024**3)
            pressure_info["system_memory_percent"] = memory.percent

            # GPU 記憶體
            try:
                import torch

                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated()
                    total = torch.cuda.get_device_properties(0).total_memory
                    pressure_info["gpu_memory_gb"] = allocated / (1024**3)
                    pressure_info["gpu_memory_percent"] = (allocated / total) * 100
            except ImportError:
                pass

            # 建議
            if (
                pressure_info["system_memory_percent"] > 85
                or pressure_info["gpu_memory_percent"] > 85
            ):
                pressure_info["recommendation"] = "cleanup_required"
            elif (
                pressure_info["system_memory_percent"] > 70
                or pressure_info["gpu_memory_percent"] > 70
            ):
                pressure_info["recommendation"] = "monitor_closely"

        except Exception as e:
            pressure_info["error"] = str(e)

        return pressure_info

    @classmethod
    def auto_cleanup_if_needed(cls, threshold_percent: float = 80.0):
        """自動清理記憶體（如果需要）"""
        pressure = cls.get_memory_pressure()

        if (
            pressure["system_memory_percent"] > threshold_percent
            or pressure["gpu_memory_percent"] > threshold_percent
        ):

            logger.info(f"Memory pressure detected, performing cleanup...")
            cls.cleanup_system_memory()
            cls.cleanup_gpu_memory()

            # 再次檢查
            new_pressure = cls.get_memory_pressure()
            logger.info(
                f"Cleanup completed. Memory usage: "
                f"System {new_pressure['system_memory_percent']:.1f}%, "
                f"GPU {new_pressure['gpu_memory_percent']:.1f}%"
            )


def memory_cleanup_decorator(cleanup_after: bool = True, cleanup_before: bool = False):
    """記憶體清理裝飾器"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if cleanup_before:
                MemoryOptimizer.auto_cleanup_if_needed()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                if cleanup_after:
                    MemoryOptimizer.auto_cleanup_if_needed()

        return wrapper

    return decorator


def performance_monitor_decorator(monitor_name: str = None):  # type: ignore
    """效能監控裝飾器"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            name = monitor_name or f"{func.__module__}.{func.__name__}"

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                logger.debug(f"Performance [{name}]: {duration:.3f}s")
                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Performance [{name}]: {duration:.3f}s (FAILED: {e})")
                raise

        return wrapper

    return decorator


# ===== 全域監控器實例 =====
_global_monitor = None


def get_resource_monitor() -> ResourceMonitor:
    """取得全域資源監控器"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = ResourceMonitor()
        _global_monitor.start_monitoring()
    return _global_monitor


def stop_global_monitoring():
    """停止全域監控"""
    global _global_monitor
    if _global_monitor:
        _global_monitor.stop_monitoring()


# ===== 效能分析工具 =====


class PerformanceProfiler:
    """效能分析器"""

    def __init__(self):
        self.profiles: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def profile_function(self, name: str):
        """函數效能分析裝飾器"""

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.perf_counter() - start_time
                    with self._lock:
                        self.profiles[name].append(duration)

            return wrapper

        return decorator

    def get_profile_stats(self, name: str) -> Dict[str, float]:
        """取得分析統計"""
        with self._lock:
            if name not in self.profiles or not self.profiles[name]:
                return {}

            durations = self.profiles[name]
            return {
                "count": len(durations),
                "total_time": sum(durations),
                "average_time": sum(durations) / len(durations),
                "min_time": min(durations),
                "max_time": max(durations),
                "last_time": durations[-1],
            }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """取得所有分析統計"""
        return {name: self.get_profile_stats(name) for name in self.profiles.keys()}

    def clear_profiles(self, name: Optional[str] = None):
        """清除分析資料"""
        with self._lock:
            if name:
                self.profiles.pop(name, None)
            else:
                self.profiles.clear()


# ===== 批次處理最佳化 =====


class BatchProcessor:
    """批次處理最佳化器"""

    def __init__(self, max_batch_size: int = 8, memory_threshold_gb: float = 1.0):
        self.max_batch_size = max_batch_size
        self.memory_threshold_gb = memory_threshold_gb

    def calculate_optimal_batch_size(self, item_memory_mb: float) -> int:
        """計算最佳批次大小"""
        try:
            # 取得可用記憶體
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)

            # GPU 記憶體檢查
            gpu_available_gb = available_gb
            try:
                import torch

                if torch.cuda.is_available():
                    gpu_total = torch.cuda.get_device_properties(0).total_memory / (
                        1024**3
                    )
                    gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
                    gpu_available_gb = min(
                        available_gb, gpu_total - gpu_reserved - 1.0
                    )  # 保留 1GB
            except ImportError:
                pass

            # 計算批次大小
            item_memory_gb = item_memory_mb / 1024
            max_by_memory = (
                int(gpu_available_gb / item_memory_gb)
                if item_memory_gb > 0
                else self.max_batch_size
            )

            optimal_size = min(self.max_batch_size, max_by_memory, 32)  # 硬限制 32

            return max(1, optimal_size)

        except Exception as e:
            logger.warning(f"Failed to calculate optimal batch size: {e}")
            return 1

    def should_reduce_batch_size(self) -> bool:
        """是否應該減少批次大小"""
        pressure = MemoryOptimizer.get_memory_pressure()
        return (
            pressure["system_memory_percent"] > 85
            or pressure["gpu_memory_percent"] > 85
        )


# ===== 模型載入最佳化 =====


class ModelLoadOptimizer:
    """模型載入最佳化器"""

    @staticmethod
    def get_optimal_device_config() -> Dict[str, Any]:
        """取得最佳裝置配置"""
        config = {
            "device": "cpu",
            "torch_dtype": "float32",
            "device_map": None,
            "low_cpu_mem_usage": True,
            "use_4bit": False,
            "use_8bit": False,
        }

        try:
            import torch

            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )

                config["device"] = "cuda"
                config["torch_dtype"] = "float16"

                # 根據 GPU 記憶體調整策略
                if gpu_memory_gb < 6:
                    # 小於 6GB: 積極最佳化
                    config.update(
                        {
                            "device_map": "auto",
                            "use_4bit": True,
                            "use_8bit": False,
                            "torch_dtype": "float16",
                        }
                    )
                elif gpu_memory_gb < 12:
                    # 6-12GB: 適度最佳化
                    config.update(
                        {
                            "device_map": "auto",
                            "use_4bit": False,
                            "use_8bit": True,
                            "torch_dtype": "float16",
                        }
                    )
                else:
                    # 12GB+: 最小最佳化
                    config.update(
                        {
                            "device_map": None,
                            "use_4bit": False,
                            "use_8bit": False,
                            "torch_dtype": "float16",
                        }
                    )

                logger.info(f"GPU config for {gpu_memory_gb:.1f}GB: {config}")

        except ImportError:
            logger.info("PyTorch not available, using CPU config")

        return config

    @staticmethod
    def optimize_model_for_inference(model, enable_optimizations: bool = True):
        """最佳化模型以進行推理"""
        if not enable_optimizations:
            return model

        try:
            # 設定為評估模式
            if hasattr(model, "eval"):
                model.eval()

            # 禁用梯度計算
            if hasattr(model, "requires_grad_"):
                model.requires_grad_(False)

            # 嘗試編譯模型 (PyTorch 2.0+)
            try:
                import torch

                if hasattr(torch, "compile") and hasattr(model, "forward"):
                    model = torch.compile(model, mode="reduce-overhead")
                    logger.info("Model compiled with torch.compile")
            except Exception:
                pass

            # Memory layout optimization
            if hasattr(model, "to") and hasattr(model, "device"):
                if str(model.device).startswith("cuda"):
                    try:
                        # Channels last memory format for conv models
                        if hasattr(model, "to"):
                            memory_format = getattr(torch, "channels_last", None)
                            if memory_format:
                                model = model.to(memory_format=memory_format)
                    except Exception:
                        pass

            return model

        except Exception as e:
            logger.warning(f"Model optimization failed: {e}")
            return model


# ===== 全域效能分析器 =====
global_profiler = PerformanceProfiler()


# ===== 便利函數 =====


def profile_function(name: str):
    """全域函數分析裝飾器"""
    return global_profiler.profile_function(name)


def get_system_performance_summary() -> Dict[str, Any]:
    """取得系統效能摘要"""
    monitor = get_resource_monitor()
    current_metrics = monitor.get_current_metrics()
    summary = monitor.get_performance_summary(minutes=10)

    return {
        "current": {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": current_metrics.cpu_percent,
            "memory_percent": current_metrics.memory_percent,
            "memory_used_gb": current_metrics.memory_used_gb,
            "gpu_memory_used_gb": current_metrics.gpu_memory_reserved_gb,
            "gpu_memory_total_gb": current_metrics.gpu_memory_total_gb,
            "health_status": summary["health"]["status"],
        },
        "summary_10min": summary,
        "memory_pressure": MemoryOptimizer.get_memory_pressure(),
        "profiler_stats": global_profiler.get_all_stats(),
    }


def optimize_for_low_memory():
    """低記憶體模式最佳化"""
    logger.info("Applying low-memory optimizations...")

    # 清理記憶體
    MemoryOptimizer.cleanup_system_memory()
    MemoryOptimizer.cleanup_gpu_memory()

    # 調整監控頻率
    monitor = get_resource_monitor()
    monitor.collection_interval = 10.0  # 減少監控頻率
    monitor.max_history_points = 360  # 減少歷史記錄

    # 清理分析器
    global_profiler.clear_profiles()

    logger.info("Low-memory optimizations applied")


def get_memory_recommendations() -> List[str]:
    """取得記憶體使用建議"""
    recommendations = []
    pressure = MemoryOptimizer.get_memory_pressure()

    if pressure["system_memory_percent"] > 85:
        recommendations.append(
            "System memory usage is high. Consider closing other applications."
        )

    if pressure["gpu_memory_percent"] > 85:
        recommendations.extend(
            [
                "GPU memory usage is high. Consider:",
                "- Reducing batch size",
                "- Using gradient checkpointing",
                "- Enabling CPU offloading",
                "- Using lower precision (fp16/bf16)",
            ]
        )

    if pressure["system_memory_percent"] > 70 and pressure["gpu_memory_percent"] > 70:
        recommendations.append(
            "Both system and GPU memory are under pressure. Consider enabling low-memory mode."
        )

    if not recommendations:
        recommendations.append("Memory usage is healthy.")

    return recommendations


# ===== 清理與關閉 =====


def cleanup_performance_monitoring():
    """清理效能監控資源"""
    try:
        stop_global_monitoring()
        global_profiler.clear_profiles()
        MemoryOptimizer.cleanup_system_memory()
        MemoryOptimizer.cleanup_gpu_memory()
        logger.info("Performance monitoring cleanup completed")
    except Exception as e:
        logger.error(f"Error during performance cleanup: {e}")


# ===== 測試與示例 =====

if __name__ == "__main__":
    print("=== Performance Monitoring Test ===")

    # 啟動監控
    monitor = get_resource_monitor()

    # 等待一些資料
    time.sleep(2)

    # 取得摘要
    summary = get_system_performance_summary()
    print(f"System Status: {summary['current']['health_status']}")
    print(f"Memory: {summary['current']['memory_percent']:.1f}%")

    if summary["current"]["gpu_memory_total_gb"] > 0:
        gpu_percent = (
            summary["current"]["gpu_memory_used_gb"]
            / summary["current"]["gpu_memory_total_gb"]
        ) * 100
        print(f"GPU Memory: {gpu_percent:.1f}%")

    # 測試記憶體建議
    recommendations = get_memory_recommendations()
    print("\nRecommendations:")
    for rec in recommendations:
        print(f"- {rec}")

    # 測試效能分析器
    @profile_function("test_function")
    def test_function():
        time.sleep(0.1)
        return "test"

    # 執行測試函數幾次
    for _ in range(3):
        test_function()

    # 顯示分析結果
    stats = global_profiler.get_profile_stats("test_function")
    if stats:
        print(f"\nFunction Profile Stats:")
        print(f"Count: {stats['count']}")
        print(f"Average: {stats['average_time']:.3f}s")
        print(f"Total: {stats['total_time']:.3f}s")

    # 清理
    cleanup_performance_monitoring()
    print("\n✅ Performance monitoring test completed")
