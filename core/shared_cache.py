# core/shared_cache.py - Shared Model/Data Warehouse Management
"""
共用模型/資料倉儲管理系統
確保所有模組使用一致的快取目錄結構和環境設定
"""

import os
import json
import hashlib
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import threading
import psutil

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


class SharedCache:
    """共用模型與資料倉儲管理器"""

    def __init__(self, cache_root: Optional[str] = None):
        self.cache_root = cache_root or os.getenv(
            "AI_CACHE_ROOT", "../ai_warehouse/cache"
        )
        self.cache_root_path = Path(self.cache_root)

        # 記憶體快取
        self._memory_cache: Dict[str, Any] = {}
        self._cache_lock = threading.Lock()

        # 統計資訊
        self.stats = {
            "models": CacheStats(),
            "datasets": CacheStats(),
            "outputs": CacheStats(),
            "memory": CacheStats(),
        }

        self._setup_environment()
        self._create_app_directories()

    def _setup_environment(self) -> None:
        """設定 AI 框架快取環境變數"""
        # Shared Cache Bootstrap
        cache_mappings = {
            "HF_HOME": f"{self.cache_root}/hf",
            "TRANSFORMERS_CACHE": f"{self.cache_root}/hf/transformers",
            "HF_DATASETS_CACHE": f"{self.cache_root}/hf/datasets",
            "HUGGINGFACE_HUB_CACHE": f"{self.cache_root}/hf/hub",
            "TORCH_HOME": f"{self.cache_root}/torch",
            "PYTORCH_KERNEL_CACHE_PATH": f"{self.cache_root}/torch/kernels",
        }

        for env_key, cache_path in cache_mappings.items():
            os.environ[env_key] = cache_path
            Path(cache_path).mkdir(parents=True, exist_ok=True)

        logger.info(f"Shared cache environment configured: {self.cache_root}")

    def _create_app_directories(self) -> None:
        """建立應用程式專用目錄結構"""
        self.app_dirs = {
            # === T2I 模型目錄 ===
            "models_sd": self.cache_root_path / "models" / "sd",
            "models_sdxl": self.cache_root_path / "models" / "sdxl",
            "models_controlnet": self.cache_root_path / "models" / "controlnet",
            "models_lora": self.cache_root_path / "models" / "lora",
            "models_ipadapter": self.cache_root_path / "models" / "ipadapter",
            # === 多模態模型 ===
            "models_llm": self.cache_root_path / "models" / "llm",
            "models_vlm": self.cache_root_path / "models" / "vlm",
            "models_embedding": self.cache_root_path / "models" / "embedding",
            "models_safety": self.cache_root_path / "models" / "safety",
            "models_tts": self.cache_root_path / "models" / "tts",
            "models_enhancement": self.cache_root_path / "models" / "enhancement",
            # === 資料集 ===
            "datasets_raw": self.cache_root_path / "datasets" / "raw",
            "datasets_processed": self.cache_root_path / "datasets" / "processed",
            "datasets_metadata": self.cache_root_path / "datasets" / "metadata",
            # === 輸出 ===
            "outputs_t2i": self.cache_root_path / "outputs" / "t2i",
            "outputs_training": self.cache_root_path / "outputs" / "training",
            "outputs_batch": self.cache_root_path / "outputs" / "batch",
            "outputs_exports": self.cache_root_path / "outputs" / "exports",
            # === 執行記錄 ===
            "runs": self.cache_root_path / "runs",
            "logs": self.cache_root_path / "logs",
            # === 註冊表 ===
            "registry": self.cache_root_path / "registry",
        }

        # 建立所有目錄
        for dir_path in self.app_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created {len(self.app_dirs)} application directories")

    def get_path(self, key: str) -> Path:
        """取得目錄路徑"""
        if key not in self.app_dirs:
            raise KeyError(f"Unknown cache directory: {key}")
        return self.app_dirs[key]

    def get_model_path(self, model_type: str, model_name: str) -> Path:
        """取得標準化模型路徑"""
        model_dir_key = f"models_{model_type}"
        if model_dir_key not in self.app_dirs:
            raise ValueError(f"Unknown model type: {model_type}")

        return self.app_dirs[model_dir_key] / model_name

    def get_dataset_path(self, dataset_type: str, dataset_name: str) -> Path:
        """取得標準化資料集路徑"""
        dataset_dir_key = f"datasets_{dataset_type}"
        if dataset_dir_key not in self.app_dirs:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        return self.app_dirs[dataset_dir_key] / dataset_name

    def get_output_path(self, output_type: str, sub_path: str = "") -> Path:
        """取得輸出目錄路徑"""
        output_dir_key = f"outputs_{output_type}"
        if output_dir_key not in self.app_dirs:
            raise ValueError(f"Unknown output type: {output_type}")

        output_path = self.app_dirs[output_dir_key]
        if sub_path:
            output_path = output_path / sub_path
            output_path.mkdir(parents=True, exist_ok=True)

        return output_path

    def cache_model_info(self, model_key: str, model_info: Dict[str, Any]) -> None:
        """快取模型資訊"""
        try:
            cache_info = {
                **model_info,
                "cached_at": datetime.now().isoformat(),
                "cache_key": model_key,
                "cache_version": "1.0",
            }

            # 決定快取檔案位置
            if "lora" in model_key.lower():
                cache_dir = self.app_dirs["models_lora"]
            elif "controlnet" in model_key.lower():
                cache_dir = self.app_dirs["models_controlnet"]
            else:
                cache_dir = self.app_dirs["registry"]

            info_file = cache_dir / f"{model_key}_info.json"

            with open(info_file, "w", encoding="utf-8") as f:
                json.dump(cache_info, f, indent=2, ensure_ascii=False)

            logger.debug(f"Cached model info: {model_key}")

        except Exception as e:
            logger.error(f"Failed to cache model info for {model_key}: {e}")

    def get_model_info(self, model_key: str) -> Optional[Dict[str, Any]]:
        """取得快取的模型資訊"""
        # 在多個位置搜尋
        search_dirs = [
            self.app_dirs["models_lora"],
            self.app_dirs["models_controlnet"],
            self.app_dirs["registry"],
        ]

        for search_dir in search_dirs:
            info_file = search_dir / f"{model_key}_info.json"
            if info_file.exists():
                try:
                    with open(info_file, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load model info from {info_file}: {e}")

        return None

    def set_memory_cache(self, key: str, value: Any, ttl_minutes: int = 60) -> None:
        """設定記憶體快取（含 TTL）"""
        with self._cache_lock:
            expires_at = datetime.now() + timedelta(minutes=ttl_minutes)
            self._memory_cache[key] = {
                "value": value,
                "expires_at": expires_at,
                "created_at": datetime.now(),
            }

            self.stats["memory"].hits += 1

    def get_memory_cache(self, key: str) -> Optional[Any]:
        """取得記憶體快取值"""
        with self._cache_lock:
            if key not in self._memory_cache:
                self.stats["memory"].misses += 1
                return None

            cache_item = self._memory_cache[key]

            # 檢查是否過期
            if datetime.now() > cache_item["expires_at"]:
                del self._memory_cache[key]
                self.stats["memory"].misses += 1
                return None

            self.stats["memory"].hits += 1
            return cache_item["value"]

    def clear_memory_cache(self, pattern: Optional[str] = None) -> int:
        """清除記憶體快取"""
        with self._cache_lock:
            if pattern is None:
                count = len(self._memory_cache)
                self._memory_cache.clear()
                logger.info(f"Cleared all memory cache ({count} items)")
                return count
            else:
                # 按模式清除
                keys_to_remove = [k for k in self._memory_cache.keys() if pattern in k]
                for key in keys_to_remove:
                    del self._memory_cache[key]
                logger.info(
                    f"Cleared memory cache pattern '{pattern}' ({len(keys_to_remove)} items)"
                )
                return len(keys_to_remove)

    def cleanup_expired_cache(self) -> Dict[str, int]:
        """清理過期快取"""
        cleanup_stats = {"memory": 0, "files": 0}

        # 清理記憶體快取
        with self._cache_lock:
            now = datetime.now()
            expired_keys = [
                k for k, v in self._memory_cache.items() if now > v["expires_at"]
            ]
            for key in expired_keys:
                del self._memory_cache[key]
            cleanup_stats["memory"] = len(expired_keys)

        # 清理舊的暫存檔案 (超過 7 天)
        cutoff_time = datetime.now() - timedelta(days=7)
        temp_dirs = [self.app_dirs["outputs_batch"], self.app_dirs["outputs_exports"]]

        for temp_dir in temp_dirs:
            if temp_dir.exists():
                for file_path in temp_dir.rglob("*"):
                    if file_path.is_file():
                        try:
                            file_time = datetime.fromtimestamp(
                                file_path.stat().st_mtime
                            )
                            if file_time < cutoff_time:
                                file_path.unlink()
                                cleanup_stats["files"] += 1
                        except Exception:
                            continue

        if cleanup_stats["memory"] + cleanup_stats["files"] > 0:
            logger.info(f"Cleanup completed: {cleanup_stats}")

        return cleanup_stats

    def get_cache_stats(self) -> Dict[str, Any]:
        """取得快取統計資訊"""
        try:
            stats = {
                "cache_root": str(self.cache_root),
                "total_size_gb": 0.0,
                "directories": {},
                "memory_cache": {
                    "items": len(self._memory_cache),
                    "hits": self.stats["memory"].hits,
                    "misses": self.stats["memory"].misses,
                    "hit_rate": self.stats["memory"].hit_rate,
                },
                "system": self._get_system_stats(),
            }

            # 計算各目錄大小
            for name, path in self.app_dirs.items():
                if path.exists():
                    dir_stats = self._calculate_directory_stats(path)
                    stats["directories"][name] = dir_stats
                    stats["total_size_gb"] += dir_stats["size_gb"]

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
                        "memory_allocated_gb": round(
                            torch.cuda.memory_allocated() / (1024**3), 3
                        ),
                        "memory_reserved_gb": round(
                            torch.cuda.memory_reserved() / (1024**3), 3
                        ),
                        "memory_total_gb": round(
                            torch.cuda.get_device_properties(0).total_memory
                            / (1024**3),
                            1,
                        ),
                    }
            except ImportError:
                pass

            # 系統資源
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage(str(self.cache_root_path.parent))

            return {
                "gpu": gpu_info,
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 1),
                    "available_gb": round(memory.available / (1024**3), 1),
                    "percent_used": memory.percent,
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 1),
                    "free_gb": round(disk.free / (1024**3), 1),
                    "percent_used": round((disk.used / disk.total) * 100, 1),
                },
            }
        except Exception as e:
            return {"error": str(e)}

    def get_device_config(self, device: str = "auto") -> Dict[str, Any]:
        """取得裝置配置建議"""
        try:
            import torch

            config = {
                "device": (
                    device
                    if device != "auto"
                    else ("cuda" if torch.cuda.is_available() else "cpu")
                ),
                "torch_dtype": "float16" if torch.cuda.is_available() else "float32",
                "low_vram_mode": False,
            }

            # 低 VRAM 最佳化
            if torch.cuda.is_available():
                try:
                    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (
                        1024**3
                    )
                    if gpu_memory_gb < 8:
                        config.update(
                            {
                                "low_vram_mode": True,
                                "enable_attention_slicing": True,
                                "enable_vae_slicing": True,
                                "enable_cpu_offload": True,
                                "torch_dtype": "float16",
                            }
                        )
                        logger.info(
                            f"Enabled low-VRAM optimizations for {gpu_memory_gb:.1f}GB GPU"
                        )
                    elif gpu_memory_gb < 12:
                        config.update(
                            {
                                "enable_attention_slicing": True,
                                "enable_vae_slicing": False,
                                "enable_cpu_offload": False,
                            }
                        )
                except Exception:
                    pass

            return config

        except ImportError:
            return {"device": "cpu", "torch_dtype": "float32", "low_vram_mode": True}

    def create_model_symlink(
        self, source_path: str, target_model_type: str, target_name: str
    ) -> bool:
        """建立模型符號連結（避免重複儲存）"""
        try:
            source = Path(source_path)
            if not source.exists():
                logger.error(f"Source model not found: {source}")
                return False

            target = self.get_model_path(target_model_type, target_name)
            target.parent.mkdir(parents=True, exist_ok=True)

            if target.exists():
                if target.is_symlink():
                    target.unlink()  # 移除舊連結
                else:
                    logger.warning(
                        f"Target already exists and is not a symlink: {target}"
                    )
                    return False

            # 建立符號連結
            target.symlink_to(source.resolve())
            logger.info(f"Created symlink: {target} -> {source}")
            return True

        except Exception as e:
            logger.error(f"Failed to create symlink: {e}")
            return False

    def register_model(
        self, model_type: str, model_name: str, metadata: Dict[str, Any]
    ) -> bool:
        """註冊模型到快取系統"""
        try:
            model_path = self.get_model_path(model_type, model_name)

            # 擴充元資料
            full_metadata = {
                "model_type": model_type,
                "model_name": model_name,
                "model_path": str(model_path),
                "registered_at": datetime.now().isoformat(),
                "cache_version": "1.0",
                **metadata,
            }

            # 快取模型資訊
            cache_key = f"{model_type}_{model_name}"
            self.cache_model_info(cache_key, full_metadata)

            # 更新模型註冊表
            registry_file = self.app_dirs["registry"] / "models.json"
            registry = {}

            if registry_file.exists():
                try:
                    with open(registry_file, "r", encoding="utf-8") as f:
                        registry = json.load(f)
                except Exception:
                    pass

            registry[cache_key] = full_metadata

            with open(registry_file, "w", encoding="utf-8") as f:
                json.dump(registry, f, indent=2, ensure_ascii=False)

            logger.info(f"Registered model: {model_type}/{model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to register model {model_type}/{model_name}: {e}")
            return False

    def list_registered_models(
        self, model_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """列出已註冊的模型"""
        try:
            registry_file = self.app_dirs["registry"] / "models.json"
            if not registry_file.exists():
                return []

            with open(registry_file, "r", encoding="utf-8") as f:
                registry = json.load(f)

            models = list(registry.values())

            if model_type:
                models = [m for m in models if m.get("model_type") == model_type]

            return sorted(models, key=lambda x: x.get("registered_at", ""))

        except Exception as e:
            logger.error(f"Failed to list registered models: {e}")
            return []

    def get_summary(self) -> Dict[str, Any]:
        """取得快取系統摘要"""
        return {
            "cache_root": str(self.cache_root),
            "status": "healthy",
            "directories": {name: str(path) for name, path in self.app_dirs.items()},
            "stats": self.get_cache_stats(),
            "device_config": self.get_device_config(),
            "env_vars": {
                "HF_HOME": os.environ.get("HF_HOME"),
                "TORCH_HOME": os.environ.get("TORCH_HOME"),
                "CUDA_VISIBLE_DEVICES": os.environ.get(
                    "CUDA_VISIBLE_DEVICES", "not_set"
                ),
            },
        }


# ===== 全域實例 =====
_shared_cache_instance = None


def get_shared_cache(cache_root: Optional[str] = None) -> SharedCache:
    """取得或建立共用快取實例"""
    global _shared_cache_instance
    if _shared_cache_instance is None:
        _shared_cache_instance = SharedCache(cache_root)
    return _shared_cache_instance


def bootstrap_cache(
    cache_root: Optional[str] = None, verbose: bool = True
) -> SharedCache:
    """初始化共用快取並顯示摘要"""
    cache = get_shared_cache(cache_root)

    if verbose:
        device_config = cache.get_device_config()
        system_stats = cache._get_system_stats()

        print(f"🎮 [SharedCache] Root: {cache.cache_root}")
        print(f"🖥️  [GPU] Available: {system_stats['gpu']['available']}")

        if system_stats["gpu"]["available"]:
            gpu = system_stats["gpu"]
            print(
                f"💾 [VRAM] {gpu['memory_allocated_gb']:.1f}GB used / {gpu['memory_total_gb']:.1f}GB total"
            )
            if device_config.get("low_vram_mode"):
                print("⚡ [Optimization] Low-VRAM mode enabled")

        memory = system_stats["memory"]
        print(
            f"🧠 [RAM] {memory['available_gb']:.1f}GB free / {memory['total_gb']:.1f}GB total"
        )

    return cache


def clear_all_cache():
    """清除所有快取（開發用）"""
    cache = get_shared_cache()
    cleanup_stats = cache.cleanup_expired_cache()
    memory_cleared = cache.clear_memory_cache()

    print(
        f"🧹 Cache cleanup: {cleanup_stats['memory']} expired + {memory_cleared} memory items"
    )


# ===== 便利函數 =====
def get_app_path(key: str) -> Path:
    """快速取得應用程式路徑"""
    return get_shared_cache().get_path(key)


def cache_model_quick(model_key: str, model_info: Dict[str, Any]):
    """快速快取模型資訊"""
    get_shared_cache().cache_model_info(model_key, model_info)


def get_model_quick(model_key: str) -> Optional[Dict[str, Any]]:
    """快速取得模型資訊"""
    return get_shared_cache().get_model_info(model_key)


if __name__ == "__main__":
    # 測試初始化
    print("=== SharedCache Bootstrap Test ===")
    cache = bootstrap_cache()

    # 顯示摘要
    summary = cache.get_summary()
    print(f"\n📊 Cache Summary:")
    print(f"Status: {summary['status']}")
    print(f"Directories: {len(summary['directories'])}")

    # 測試模型註冊
    test_metadata = {"description": "Test model", "size_gb": 1.5, "source": "test"}

    success = cache.register_model("lora", "test_model", test_metadata)
    print(f"Model registration test: {'✅' if success else '❌'}")

    # 列出模型
    models = cache.list_registered_models("lora")
    print(f"Registered LoRA models: {len(models)}")

    print("\n✅ SharedCache initialization completed")
