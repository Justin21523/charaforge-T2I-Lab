# core/shared_cache.py - Shared Cache Manager (Updated)
"""
共用模型/資料倉儲管理系統 - 與統一配置系統整合
確保所有模組使用一致的快取目錄結構和環境設定
"""

import os
import json
import hashlib
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import threading
import psutil

# Import unified config system
from core.config import (
    get_settings,
    get_cache_paths,
    get_app_paths,
    CachePaths,
    AppPaths,
)

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """快取統計資訊"""

    hits: int = 0
    misses: int = 0
    total_size_gb: float = 0.0
    file_count: int = 0
    last_updated: Optional[str] = None

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


@dataclass
class ModelInfo:
    """模型資訊結構"""

    model_id: str
    model_type: str  # "sd15", "sdxl", "lora", "controlnet", "embedding"
    path: Path
    size_mb: float
    cached_at: str
    metadata: Dict[str, Any]

    @property
    def size_gb(self) -> float:
        return self.size_mb / 1024


class SharedCache:
    """共用模型與資料倉儲管理器 - 整合統一配置系統"""

    def __init__(self):
        # Use unified config system
        self.settings = get_settings()
        self.cache_paths = get_cache_paths()
        self.app_paths = get_app_paths()

        # Memory cache
        self._memory_cache: Dict[str, Any] = {}
        self._cache_lock = threading.Lock()

        # Statistics
        self.stats = {
            "models": CacheStats(),
            "datasets": CacheStats(),
            "outputs": CacheStats(),
            "memory": CacheStats(),
        }

        # Model registry
        self._model_registry: Dict[str, ModelInfo] = {}
        self._load_model_registry()

    def _load_model_registry(self) -> None:
        """載入模型註冊表"""
        registry_file = self.app_paths.models_sd15.parent / "registry.json"

        if registry_file.exists():
            try:
                with open(registry_file, "r", encoding="utf-8") as f:
                    registry_data = json.load(f)

                for model_id, data in registry_data.items():
                    self._model_registry[model_id] = ModelInfo(
                        model_id=data["model_id"],
                        model_type=data["model_type"],
                        path=Path(data["path"]),
                        size_mb=data["size_mb"],
                        cached_at=data["cached_at"],
                        metadata=data.get("metadata", {}),
                    )

                logger.info(f"Loaded {len(self._model_registry)} models from registry")

            except Exception as e:
                logger.error(f"Failed to load model registry: {e}")

    def _save_model_registry(self) -> None:
        """儲存模型註冊表"""
        registry_file = self.app_paths.models_sd15.parent / "registry.json"
        registry_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            registry_data = {}
            for model_id, info in self._model_registry.items():
                registry_data[model_id] = {
                    "model_id": info.model_id,
                    "model_type": info.model_type,
                    "path": str(info.path),
                    "size_mb": info.size_mb,
                    "cached_at": info.cached_at,
                    "metadata": info.metadata,
                }

            with open(registry_file, "w", encoding="utf-8") as f:
                json.dump(registry_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")

    def get_model_path(self, model_type: str, model_name: str) -> Path:
        """取得模型路徑"""
        model_paths = {
            "sd15": self.app_paths.models_sd15,
            "sdxl": self.app_paths.models_sdxl,
            "controlnet": self.app_paths.models_controlnet,
            "lora": self.app_paths.lora_weights,
            "embedding": self.app_paths.embeddings,
        }

        if model_type not in model_paths:
            raise ValueError(f"Unknown model type: {model_type}")

        return model_paths[model_type] / model_name

    def cache_model_info(self, model_id: str, metadata: Dict[str, Any]) -> None:
        """快取模型資訊"""
        with self._cache_lock:
            self._memory_cache[f"model_info:{model_id}"] = {
                "metadata": metadata,
                "cached_at": datetime.now().isoformat(),
                "access_count": self._memory_cache.get(
                    f"model_info:{model_id}", {}
                ).get("access_count", 0)
                + 1,
            }
            self.stats["memory"].hits += 1

    def get_cached_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """取得快取的模型資訊"""
        with self._cache_lock:
            cache_key = f"model_info:{model_id}"
            if cache_key in self._memory_cache:
                self.stats["memory"].hits += 1
                self._memory_cache[cache_key]["access_count"] += 1
                return self._memory_cache[cache_key]["metadata"]
            else:
                self.stats["memory"].misses += 1
                return None

    def register_model(
        self,
        model_id: str,
        model_type: str,
        local_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """註冊模型到快取系統"""
        try:
            local_path = Path(local_path)
            if not local_path.exists():
                logger.error(f"Model path does not exist: {local_path}")
                return False

            # Calculate model size
            if local_path.is_file():
                size_mb = local_path.stat().st_size / (1024 * 1024)
            else:
                size_mb = sum(
                    f.stat().st_size for f in local_path.rglob("*") if f.is_file()
                ) / (1024 * 1024)

            # Create model info
            model_info = ModelInfo(
                model_id=model_id,
                model_type=model_type,
                path=local_path,
                size_mb=size_mb,
                cached_at=datetime.now().isoformat(),
                metadata=metadata or {},
            )

            # Register model
            self._model_registry[model_id] = model_info
            self._save_model_registry()

            # Cache metadata
            if metadata:
                self.cache_model_info(model_id, metadata)

            logger.info(f"Registered model: {model_type}/{model_id} ({size_mb:.1f}MB)")
            return True

        except Exception as e:
            logger.error(f"Failed to register model {model_id}: {e}")
            return False

    def get_registered_models(
        self, model_type: Optional[str] = None
    ) -> List[ModelInfo]:
        """取得已註冊的模型列表"""
        models = list(self._model_registry.values())

        if model_type:
            models = [m for m in models if m.model_type == model_type]

        return sorted(models, key=lambda x: x.cached_at, reverse=True)

    def is_model_cached(self, model_id: str) -> bool:
        """檢查模型是否已快取"""
        return model_id in self._model_registry

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """取得模型資訊"""
        return self._model_registry.get(model_id)

    def cleanup_old_cache(self, max_age_days: int = 30) -> Dict[str, int]:
        """清理舊的快取檔案"""
        cleanup_stats = {"files": 0, "memory": 0, "size_freed_mb": 0}
        cutoff_time = time.time() - (max_age_days * 24 * 3600)

        # Clean memory cache
        with self._cache_lock:
            expired_keys = []
            for key, value in self._memory_cache.items():
                if isinstance(value, dict) and "cached_at" in value:
                    try:
                        cached_time = datetime.fromisoformat(
                            value["cached_at"]
                        ).timestamp()
                        if cached_time < cutoff_time:
                            expired_keys.append(key)
                    except Exception:
                        continue

            for key in expired_keys:
                del self._memory_cache[key]
                cleanup_stats["memory"] += 1

        # Clean old model files (optional - be careful!)
        # Only clean temporary/cache files, not registered models
        temp_dirs = [self.cache_paths.cache / "temp", self.cache_paths.outputs / "temp"]

        for temp_dir in temp_dirs:
            if temp_dir.exists():
                for file_path in temp_dir.rglob("*"):
                    if file_path.is_file():
                        try:
                            file_time = file_path.stat().st_mtime
                            if file_time < cutoff_time:
                                size_mb = file_path.stat().st_size / (1024 * 1024)
                                file_path.unlink()
                                cleanup_stats["files"] += 1
                                cleanup_stats["size_freed_mb"] += size_mb
                        except Exception:
                            continue

        if cleanup_stats["memory"] + cleanup_stats["files"] > 0:
            logger.info(f"Cache cleanup completed: {cleanup_stats}")

        return cleanup_stats

    def get_cache_stats(self) -> Dict[str, Any]:
        """取得快取統計資訊"""
        try:
            stats = {
                "cache_root": str(self.cache_paths.root),
                "total_size_gb": 0.0,
                "directories": {},
                "memory_cache": {
                    "items": len(self._memory_cache),
                    "hits": self.stats["memory"].hits,
                    "misses": self.stats["memory"].misses,
                    "hit_rate": self.stats["memory"].hit_rate,
                },
                "registered_models": {
                    "total": len(self._model_registry),
                    "by_type": {},
                },
                "system": self._get_system_stats(),
            }

            # Calculate directory sizes
            cache_dirs = {
                "models": self.cache_paths.models,
                "datasets": self.cache_paths.datasets,
                "outputs": self.cache_paths.outputs,
                "cache": self.cache_paths.cache,
                "runs": self.cache_paths.runs,
            }

            for name, path in cache_dirs.items():
                if path.exists():
                    dir_stats = self._calculate_directory_stats(path)
                    stats["directories"][name] = dir_stats
                    stats["total_size_gb"] += dir_stats["size_gb"]

            # Model statistics by type
            model_types = {}
            for model_info in self._model_registry.values():
                model_type = model_info.model_type
                if model_type not in model_types:
                    model_types[model_type] = {"count": 0, "size_gb": 0.0}
                model_types[model_type]["count"] += 1
                model_types[model_type]["size_gb"] += model_info.size_gb

            stats["registered_models"]["by_type"] = model_types

            return stats

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}

    def _calculate_directory_stats(self, directory: Path) -> Dict[str, Any]:
        """計算目錄統計資訊"""
        try:
            total_size = 0
            file_count = 0

            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1

            return {
                "size_gb": round(total_size / (1024**3), 3),
                "size_mb": round(total_size / (1024**2), 1),
                "file_count": file_count,
                "exists": True,
            }
        except Exception as e:
            return {"exists": False, "error": str(e)}

    def _get_system_stats(self) -> Dict[str, Any]:
        """取得系統統計資訊"""
        try:
            # GPU 資訊
            gpu_info = {"available": False}
            try:
                import torch

                if torch.cuda.is_available():
                    gpu_info = {
                        "available": True,
                        "device_count": torch.cuda.device_count(),
                        "current_device": torch.cuda.current_device(),
                        "device_name": torch.cuda.get_device_name(),
                        "memory_allocated_gb": round(
                            torch.cuda.memory_allocated() / (1024**3), 2
                        ),
                        "memory_reserved_gb": round(
                            torch.cuda.memory_reserved() / (1024**3), 2
                        ),
                    }
            except ImportError:
                pass

            # CPU 和記憶體資訊
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage(str(self.cache_paths.root))

            return {
                "gpu": gpu_info,
                "cpu_percent": cpu_percent,
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_percent": memory.percent,
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "used_percent": round((disk.used / disk.total) * 100, 1),
                },
            }
        except Exception as e:
            return {"error": str(e)}

    def get_device_config(self) -> Dict[str, Any]:
        """取得設備配置資訊"""
        try:
            import torch

            device_config = {
                "cuda_available": torch.cuda.is_available(),
                "device_count": (
                    torch.cuda.device_count() if torch.cuda.is_available() else 0
                ),
                "current_device": (
                    torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
                ),
                "low_vram_mode": self.settings.model.low_vram_mode,
                "use_fp16": self.settings.model.use_fp16,
                "use_bf16": self.settings.model.use_bf16,
                "enable_xformers": self.settings.model.enable_xformers,
                "cpu_offload": self.settings.model.enable_cpu_offload,
            }

            if torch.cuda.is_available():
                device_config.update(
                    {
                        "device_name": torch.cuda.get_device_name(),
                        "compute_capability": torch.cuda.get_device_capability(),
                        "memory_total_gb": round(
                            torch.cuda.get_device_properties(0).total_memory
                            / (1024**3),
                            2,
                        ),
                    }
                )

            return device_config

        except ImportError:
            return {"cuda_available": False, "error": "PyTorch not available"}

    def get_summary(self) -> Dict[str, Any]:
        """取得快取系統摘要"""
        return {
            "cache_root": str(self.cache_paths.root),
            "app_root": str(self.app_paths.root),
            "status": "healthy",
            "environment": self.settings.environment,
            "debug_mode": self.settings.debug,
            "directories": {
                "cache_paths": {
                    name: str(getattr(self.cache_paths, name))
                    for name in [
                        "root",
                        "models",
                        "datasets",
                        "outputs",
                        "cache",
                        "runs",
                    ]
                },
                "app_paths": {
                    name: str(getattr(self.app_paths, name))
                    for name in [
                        "models_sd15",
                        "models_sdxl",
                        "lora_weights",
                        "training_runs",
                    ]
                },
            },
            "stats": self.get_cache_stats(),
            "device_config": self.get_device_config(),
            "registered_models": len(self._model_registry),
        }


# ===== Global Instance Management =====

_shared_cache_instance: Optional[SharedCache] = None


def get_shared_cache() -> SharedCache:
    """取得或建立共用快取實例"""
    global _shared_cache_instance
    if _shared_cache_instance is None:
        _shared_cache_instance = SharedCache()
    return _shared_cache_instance


def bootstrap_cache(verbose: bool = False) -> SharedCache:
    """Bootstrap shared cache system"""
    if verbose:
        print("🗂️  Bootstrapping shared cache system...")

    cache = get_shared_cache()

    if verbose:
        summary = cache.get_summary()
        print(f"✅ Shared cache initialized")
        print(f"   Cache root: {summary['cache_root']}")
        print(f"   Registered models: {summary['registered_models']}")
        print(f"   Device: {summary['device_config'].get('device_name', 'CPU')}")

        if summary["device_config"]["cuda_available"]:
            print(
                f"   GPU memory: {summary['device_config'].get('memory_total_gb', 0)}GB"
            )
            print(f"   Low VRAM mode: {summary['device_config']['low_vram_mode']}")

    return cache


def reset_shared_cache() -> None:
    """重置共用快取實例 (測試用)"""
    global _shared_cache_instance
    _shared_cache_instance = None
