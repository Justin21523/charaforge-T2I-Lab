# utils/calculator.py
"""
計算與效能工具 - VRAM 估算、批次大小最佳化、效能監控
"""

import os
import math
import time
import psutil
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelSpecs:
    """模型規格"""

    name: str
    params_millions: float
    base_vram_mb: float
    fp16_multiplier: float = 0.5
    fp32_multiplier: float = 1.0
    int8_multiplier: float = 0.25
    int4_multiplier: float = 0.125


@dataclass
class SystemInfo:
    """系統資訊"""

    total_ram_gb: float
    available_ram_gb: float
    gpu_count: int
    gpu_memory_gb: List[float]
    cpu_count: int


class MemoryCalculator:
    """記憶體與 VRAM 計算器"""

    def __init__(self):
        self.model_registry = {
            # Stable Diffusion 模型
            "sd_1_5": ModelSpecs("SD 1.5", 860, 3800),
            "sd_2_1": ModelSpecs("SD 2.1", 865, 4200),
            "sdxl_base": ModelSpecs("SDXL Base", 3500, 7200),
            "sdxl_refiner": ModelSpecs("SDXL Refiner", 2300, 5100),
            # LLM 模型
            "llama_7b": ModelSpecs("LLaMA 7B", 7000, 14000),
            "llama_13b": ModelSpecs("LLaMA 13B", 13000, 26000),
            "llama_30b": ModelSpecs("LLaMA 30B", 30000, 60000),
            # 多模態模型
            "clip_vit_l": ModelSpecs("CLIP ViT-L", 428, 1700),
            "clip_vit_g": ModelSpecs("CLIP ViT-G", 1300, 2600),
        }

    def estimate_model_vram(
        self,
        model_name: str,
        precision: str = "fp16",
        batch_size: int = 1,
        sequence_length: int = 512,
        include_training: bool = False,
    ) -> Dict[str, float]:
        """
        估算模型 VRAM 使用量

        Args:
            model_name: 模型名稱
            precision: 精度 (fp32, fp16, int8, int4)
            batch_size: 批次大小
            sequence_length: 序列長度 (對文字模型)
            include_training: 是否包含訓練記憶體

        Returns:
            Dict: VRAM 使用量估算
        """
        if model_name not in self.model_registry:
            logger.warning(f"Unknown model: {model_name}")
            return {"error": f"Unknown model: {model_name}"}

        model_spec = self.model_registry[model_name]

        # 精度倍數
        precision_multipliers = {
            "fp32": model_spec.fp32_multiplier,
            "fp16": model_spec.fp16_multiplier,
            "bf16": model_spec.fp16_multiplier,
            "int8": model_spec.int8_multiplier,
            "int4": model_spec.int4_multiplier,
        }

        multiplier = precision_multipliers.get(precision, model_spec.fp16_multiplier)

        # 基礎模型記憶體
        base_memory = model_spec.base_vram_mb * multiplier

        # 批次大小影響
        batch_factor = 1.0 + (batch_size - 1) * 0.7  # 非線性增長

        # 序列長度影響 (主要對 LLM)
        if "llama" in model_name.lower() or "gpt" in model_name.lower():
            seq_factor = sequence_length / 512  # 以 512 為基準
            batch_factor *= seq_factor

        # 訓練額外記憶體 (優化器狀態、梯度等)
        training_multiplier = 3.0 if include_training else 1.0

        # 計算總記憶體
        total_vram = base_memory * batch_factor * training_multiplier

        # 加入緩衝區
        overhead = total_vram * 0.2  # 20% 緩衝

        result = {
            "model_name": model_spec.name,
            "precision": precision,
            "batch_size": batch_size,
            "base_vram_mb": base_memory,
            "batch_factor": batch_factor,
            "training_multiplier": training_multiplier,
            "overhead_mb": overhead,
            "total_vram_mb": total_vram + overhead,
            "total_vram_gb": (total_vram + overhead) / 1024,
        }

        return result

    def optimize_batch_size(
        self,
        model_name: str,
        available_vram_gb: float,
        precision: str = "fp16",
        sequence_length: int = 512,
        safety_margin: float = 0.9,
        include_training: bool = False,
    ) -> Dict[str, Any]:
        """
        最佳化批次大小

        Args:
            model_name: 模型名稱
            available_vram_gb: 可用 VRAM (GB)
            precision: 精度
            sequence_length: 序列長度
            safety_margin: 安全邊際 (0.9 = 90% 利用率)
            include_training: 是否包含訓練

        Returns:
            Dict: 最佳批次大小建議
        """
        available_vram_mb = available_vram_gb * 1024 * safety_margin

        # 二分搜尋最佳批次大小
        min_batch = 1
        max_batch = 64
        optimal_batch = 1

        for batch_size in range(min_batch, max_batch + 1):
            estimation = self.estimate_model_vram(
                model_name=model_name,
                precision=precision,
                batch_size=batch_size,
                sequence_length=sequence_length,
                include_training=include_training,
            )

            if "error" in estimation:
                break

            if estimation["total_vram_mb"] <= available_vram_mb:
                optimal_batch = batch_size
            else:
                break

        # 計算最終估算
        final_estimation = self.estimate_model_vram(
            model_name=model_name,
            precision=precision,
            batch_size=optimal_batch,
            sequence_length=sequence_length,
            include_training=include_training,
        )

        # 效能建議
        recommendations = []

        if optimal_batch == 1:
            recommendations.append(
                "Consider using int8 or int4 precision for larger batches"
            )

        utilization = final_estimation["total_vram_mb"] / available_vram_mb
        if utilization < 0.5:
            recommendations.append(
                "VRAM underutilized - could increase resolution or use higher precision"
            )
        elif utilization > 0.9:
            recommendations.append("High VRAM utilization - monitor for OOM errors")

        return {
            "optimal_batch_size": optimal_batch,
            "vram_utilization": utilization,
            "estimated_vram_gb": final_estimation["total_vram_gb"],
            "available_vram_gb": available_vram_gb,
            "safety_margin": safety_margin,
            "recommendations": recommendations,
            "estimation_details": final_estimation,
        }


class PerformanceMonitor:
    """效能監控器"""

    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.operations: List[Dict[str, Any]] = []
        self.system_snapshots: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

        # 啟動背景監控
        self._monitoring_active = False
        self._monitor_thread = None

    def start_monitoring(self, interval_seconds: float = 30.0):
        """啟動系統監控"""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._background_monitor, args=(interval_seconds,), daemon=True
        )
        self._monitor_thread.start()
        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """停止系統監控"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Performance monitoring stopped")

    def _background_monitor(self, interval: float):
        """背景監控執行緒"""
        while self._monitoring_active:
            try:
                snapshot = self.get_system_snapshot()

                with self._lock:
                    self.system_snapshots.append(snapshot)
                    if len(self.system_snapshots) > self.history_size:
                        self.system_snapshots.pop(0)

                time.sleep(interval)

            except Exception as e:
                logger.error(f"Background monitoring error: {e}")
                time.sleep(interval)

    def log_operation(self, operation: str, duration: float, **kwargs):
        """記錄操作效能"""
        with self._lock:
            operation_data = {
                "operation": operation,
                "duration": duration,
                "timestamp": datetime.now().isoformat(),
                "system_info": self.get_system_snapshot(),
                **kwargs,
            }

            self.operations.append(operation_data)
            if len(self.operations) > self.history_size:
                self.operations.pop(0)

    def get_system_snapshot(self) -> Dict[str, Any]:
        """取得系統即時狀態"""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
        }

        # GPU 資訊
        try:
            import torch

            if torch.cuda.is_available():
                gpu_info = []
                for i in range(torch.cuda.device_count()):
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory
                    gpu_allocated = torch.cuda.memory_allocated(i)
                    gpu_reserved = torch.cuda.memory_reserved(i)

                    gpu_info.append(
                        {
                            "device_id": i,
                            "name": torch.cuda.get_device_name(i),
                            "total_memory_gb": gpu_memory / (1024**3),
                            "allocated_memory_gb": gpu_allocated / (1024**3),
                            "reserved_memory_gb": gpu_reserved / (1024**3),
                            "utilization_percent": (gpu_allocated / gpu_memory) * 100,
                        }
                    )

                snapshot["gpu"] = gpu_info
        except Exception as e:
            snapshot["gpu_error"] = str(e)

        return snapshot

    def get_performance_stats(self) -> Dict[str, Any]:
        """取得效能統計"""
        with self._lock:
            if not self.operations:
                return {"total_operations": 0}

            durations = [op["duration"] for op in self.operations]

            # 按操作類型分組
            operations_by_type = {}
            for op in self.operations:
                op_type = op["operation"]
                if op_type not in operations_by_type:
                    operations_by_type[op_type] = []
                operations_by_type[op_type].append(op["duration"])

            type_stats = {}
            for op_type, type_durations in operations_by_type.items():
                type_stats[op_type] = {
                    "count": len(type_durations),
                    "avg_duration": sum(type_durations) / len(type_durations),
                    "max_duration": max(type_durations),
                    "min_duration": min(type_durations),
                }

            return {
                "total_operations": len(self.operations),
                "avg_duration": sum(durations) / len(durations),
                "max_duration": max(durations),
                "min_duration": min(durations),
                "slow_operations": len([d for d in durations if d > 5.0]),  # 超過 5 秒
                "operations_by_type": type_stats,
                "monitoring_active": self._monitoring_active,
                "system_snapshots_count": len(self.system_snapshots),
            }

    def get_system_trends(self, hours: int = 24) -> Dict[str, Any]:
        """取得系統趨勢"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        with self._lock:
            recent_snapshots = [
                snapshot
                for snapshot in self.system_snapshots
                if datetime.fromisoformat(snapshot["timestamp"]) > cutoff_time
            ]

        if not recent_snapshots:
            return {"error": "No recent data available"}

        # 計算趨勢
        cpu_values = [s["cpu_percent"] for s in recent_snapshots]
        memory_values = [s["memory_percent"] for s in recent_snapshots]

        trends = {
            "period_hours": hours,
            "snapshots_count": len(recent_snapshots),
            "cpu": {
                "avg": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values),
            },
            "memory": {
                "avg": sum(memory_values) / len(memory_values),
                "max": max(memory_values),
                "min": min(memory_values),
            },
        }

        # GPU 趨勢
        gpu_utilizations = []
        for snapshot in recent_snapshots:
            if "gpu" in snapshot and isinstance(snapshot["gpu"], list):
                for gpu in snapshot["gpu"]:
                    gpu_utilizations.append(gpu["utilization_percent"])

        if gpu_utilizations:
            trends["gpu"] = {
                "avg_utilization": sum(gpu_utilizations) / len(gpu_utilizations),
                "max_utilization": max(gpu_utilizations),
                "min_utilization": min(gpu_utilizations),
            }

        return trends


class ResourceOptimizer:
    """資源最佳化建議"""

    @staticmethod
    def analyze_system() -> Dict[str, Any]:
        """分析系統並提供最佳化建議"""
        system_info = get_system_info()
        recommendations = []
        warnings = []

        # RAM 分析
        if system_info.available_ram_gb < 4:
            warnings.append("Low available RAM - consider closing other applications")
        elif system_info.available_ram_gb < 8:
            recommendations.append("Consider increasing batch size or model precision")

        # GPU 分析
        if system_info.gpu_count == 0:
            warnings.append("No GPU detected - operations will use CPU (much slower)")
        else:
            for i, gpu_memory in enumerate(system_info.gpu_memory_gb):
                if gpu_memory < 4:
                    warnings.append(f"GPU {i} has limited VRAM ({gpu_memory:.1f}GB)")
                elif gpu_memory >= 12:
                    recommendations.append(
                        f"GPU {i} can handle larger models or batches"
                    )

        # CPU 分析
        if system_info.cpu_count < 4:
            recommendations.append("Consider enabling CPU fallback for some operations")
        elif system_info.cpu_count >= 8:
            recommendations.append("High CPU count - good for parallel processing")

        return {
            "system_info": system_info,
            "recommendations": recommendations,
            "warnings": warnings,
            "optimization_score": len(recommendations) / max(1, len(warnings)),
        }


def get_system_info() -> SystemInfo:
    """取得系統資訊"""
    memory = psutil.virtual_memory()

    gpu_count = 0
    gpu_memory = []

    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                total_memory = torch.cuda.get_device_properties(i).total_memory
                gpu_memory.append(total_memory / (1024**3))
    except:
        pass

    return SystemInfo(
        total_ram_gb=memory.total / (1024**3),
        available_ram_gb=memory.available / (1024**3),
        gpu_count=gpu_count,
        gpu_memory_gb=gpu_memory,
        cpu_count=psutil.cpu_count(),  # type: ignore
    )


def estimate_vram_usage(model_name: str, **kwargs) -> Dict[str, float]:
    """估算 VRAM 使用量 (獨立函數)"""
    calculator = MemoryCalculator()
    return calculator.estimate_model_vram(model_name, **kwargs)


def optimize_batch_size(
    model_name: str, available_vram_gb: float, **kwargs
) -> Dict[str, Any]:
    """最佳化批次大小 (獨立函數)"""
    calculator = MemoryCalculator()
    return calculator.optimize_batch_size(model_name, available_vram_gb, **kwargs)


def calculate_training_resources(
    model_name: str, dataset_size: int, epochs: int = 5, precision: str = "fp16"
) -> Dict[str, Any]:
    """計算訓練所需資源"""
    calculator = MemoryCalculator()

    # 基礎估算
    base_estimation = calculator.estimate_model_vram(
        model_name=model_name, precision=precision, batch_size=1, include_training=True
    )

    if "error" in base_estimation:
        return base_estimation

    # 訓練時間估算 (粗略)
    samples_per_second = {
        "sd_1_5": 0.5,
        "sdxl_base": 0.2,
        "llama_7b": 2.0,
        "llama_13b": 1.0,
    }.get(model_name, 1.0)

    total_samples = dataset_size * epochs
    estimated_time_hours = (total_samples / samples_per_second) / 3600

    return {
        "model_name": model_name,
        "dataset_size": dataset_size,
        "epochs": epochs,
        "total_samples": total_samples,
        "estimated_vram_gb": base_estimation["total_vram_gb"],
        "estimated_time_hours": estimated_time_hours,
        "estimated_time_days": estimated_time_hours / 24,
        "base_estimation": base_estimation,
    }
