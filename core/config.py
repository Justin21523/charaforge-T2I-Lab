# core/config.py - Unified Configuration Management
"""
統一配置管理系統
整合環境變數、YAML 設定檔、共用倉儲路徑管理
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CachePaths:
    """標準化共用倉儲路徑管理"""

    def __init__(self, cache_root: str):
        self.root = Path(cache_root)

        # 主要分類目錄
        self.models = self.root / "models"
        self.datasets = self.root / "datasets"
        self.cache = self.root / "cache"
        self.runs = self.root / "runs"
        self.outputs = self.root / "outputs"

        # Model 子目錄 (T2I 專用)
        self.models_sd = self.models / "sd"
        self.models_sdxl = self.models / "sdxl"
        self.models_controlnet = self.models / "controlnet"
        self.models_lora = self.models / "lora"
        self.models_ipadapter = self.models / "ipadapter"

        # 多模態擴展
        self.models_llm = self.models / "llm"
        self.models_vlm = self.models / "vlm"
        self.models_embedding = self.models / "embedding"
        self.models_safety = self.models / "safety"
        self.models_tts = self.models / "tts"
        self.models_enhancement = self.models / "enhancement"

        # 確保目錄存在
        self._ensure_dirs()

    def _ensure_dirs(self):
        """建立所有必要目錄"""
        dirs = [
            # 主要目錄
            self.models,
            self.datasets,
            self.cache,
            self.runs,
            self.outputs,
            # T2I 模型目錄
            self.models_sd,
            self.models_sdxl,
            self.models_controlnet,
            self.models_lora,
            self.models_ipadapter,
            # 多模態模型目錄
            self.models_llm,
            self.models_vlm,
            self.models_embedding,
            self.models_safety,
            self.models_tts,
            self.models_enhancement,
            # 資料目錄
            self.datasets / "raw",
            self.datasets / "processed",
            self.datasets / "metadata",
            # 輸出目錄
            self.outputs / "t2i",
            self.outputs / "training",
            self.outputs / "batch",
            # HF 快取目錄
            self.cache / "hf" / "transformers",
            self.cache / "hf" / "datasets",
            self.cache / "hf" / "hub",
            self.cache / "torch",
        ]

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        logger.info(f"Warehouse initialized: {self.root}")


class Settings(BaseSettings):
    """主要應用程式設定"""

    # === 快取與路徑 ===
    ai_cache_root: str = Field(default="../ai_warehouse/cache", env="AI_CACHE_ROOT")

    # === API 設定 ===
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_cors_origins: str = Field(
        default="http://localhost:3000", env="API_CORS_ORIGINS"
    )
    debug: bool = Field(default=False, env="DEBUG")

    # === GPU/CUDA 設定 ===
    cuda_visible_devices: str = Field(default="0", env="CUDA_VISIBLE_DEVICES")
    device_map: str = Field(default="auto", env="DEVICE_MAP")
    torch_dtype: str = Field(default="float16", env="TORCH_DTYPE")
    use_4bit_loading: bool = Field(default=True, env="USE_4BIT")
    enable_cpu_offload: bool = Field(default=True, env="CPU_OFFLOAD")

    # === Redis/Celery ===
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    celery_broker_url: str = Field(
        default="redis://localhost:6379/0", env="CELERY_BROKER_URL"
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/0", env="CELERY_RESULT_BACKEND"
    )

    # === 安全設定 ===
    enable_nsfw_filter: bool = Field(default=True, env="ENABLE_NSFW_FILTER")
    enable_watermark: bool = Field(default=True, env="ENABLE_WATERMARK")
    blocked_terms: str = Field(default="", env="BLOCKED_TERMS")

    # === 效能設定 ===
    max_workers: int = Field(default=2, env="MAX_WORKERS")
    max_batch_size: int = Field(default=4, env="MAX_BATCH_SIZE")
    enable_xformers: bool = Field(default=True, env="ENABLE_XFORMERS")
    enable_attention_slicing: bool = Field(default=True, env="ATTENTION_SLICING")

    def setup_hf_cache(self):
        """設定 HuggingFace 快取環境變數"""
        cache_paths = self.get_cache_paths()
        hf_cache = cache_paths.cache / "hf"

        env_vars = {
            "HF_HOME": str(hf_cache),
            "TRANSFORMERS_CACHE": str(hf_cache / "transformers"),
            "HF_DATASETS_CACHE": str(hf_cache / "datasets"),
            "HUGGINGFACE_HUB_CACHE": str(hf_cache / "hub"),
            "TORCH_HOME": str(cache_paths.cache / "torch"),
        }

        for k, v in env_vars.items():
            os.environ[k] = v
            Path(v).mkdir(parents=True, exist_ok=True)

        logger.info("HuggingFace cache environment configured")

    def get_cache_paths(self) -> CachePaths:
        """取得標準化快取路徑"""
        return CachePaths(self.ai_cache_root)

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.api_cors_origins.split(",")]

    @property
    def blocked_terms_list(self) -> List[str]:
        if not self.blocked_terms:
            return []
        return [term.strip().lower() for term in self.blocked_terms.split(",")]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class ConfigManager:
    """YAML 配置檔案管理器"""

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self._cache = {}
        self._ensure_config_dir()

    def _ensure_config_dir(self):
        """確保配置目錄存在並建立預設檔案"""
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # 建立預設 app.yaml
        app_config_file = self.config_dir / "app.yaml"
        if not app_config_file.exists():
            self._create_default_app_config(app_config_file)

    def _create_default_app_config(self, config_file: Path):
        """建立預設應用程式配置"""
        default_config = {
            "app": {
                "name": "CharaForge T2I Lab",
                "version": "0.1.0",
                "description": "Text-to-Image generation and LoRA fine-tuning lab",
            },
            "features": {
                "enable_t2i": True,
                "enable_lora": True,
                "enable_controlnet": True,
                "enable_safety": True,
                "enable_watermark": True,
                "enable_batch": True,
                "enable_monitoring": True,
            },
            "limits": {
                "max_image_size": 2048,
                "max_batch_size": 50,
                "max_training_time_hours": 24,
                "request_timeout_seconds": 300,
            },
            "performance": {
                "low_vram_mode": True,
                "mixed_precision": True,
                "gradient_checkpointing": True,
                "model_cache_size_gb": 10,
            },
        }

        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)

    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """載入 YAML 配置檔案（含快取）"""
        if filename in self._cache:
            return self._cache[filename]

        file_path = self.config_dir / filename
        if not file_path.exists():
            logger.warning(f"Config file not found: {file_path}")
            return {}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config {file_path}: {e}")
            return {}

        self._cache[filename] = config
        return config

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """取得模型配置"""
        models_config = self.load_yaml("models.yaml")
        return models_config.get(model_name, {})

    def get_train_config(self, config_name: str) -> Dict[str, Any]:
        """取得訓練配置"""
        return self.load_yaml(f"train/{config_name}")

    def get_preset_config(self, preset_name: str) -> Dict[str, Any]:
        """取得風格預設配置"""
        return self.load_yaml(f"presets/{preset_name}")

    def get_app_config(self) -> Dict[str, Any]:
        """取得應用程式配置"""
        return self.load_yaml("app.yaml")


# ===== 全域實例 =====
settings = Settings()
config_manager = ConfigManager()

# 初始化 HF 快取
settings.setup_hf_cache()

# 導出快取路徑以便直接使用
cache_paths = settings.get_cache_paths()


# ===== 便利函數 =====
def get_settings() -> Settings:
    """取得全域設定實例"""
    return settings


def get_config_manager() -> ConfigManager:
    """取得全域配置管理器實例"""
    return config_manager


def get_cache_paths() -> CachePaths:
    """取得全域快取路徑實例"""
    return cache_paths


def get_model_path(model_type: str, model_name: str) -> Path:
    """取得標準化模型路徑"""
    paths = get_cache_paths()

    model_dirs = {
        "sd": paths.models_sd,
        "sdxl": paths.models_sdxl,
        "controlnet": paths.models_controlnet,
        "lora": paths.models_lora,
        "ipadapter": paths.models_ipadapter,
        "llm": paths.models_llm,
        "vlm": paths.models_vlm,
        "embedding": paths.models_embedding,
        "safety": paths.models_safety,
    }

    if model_type not in model_dirs:
        raise ValueError(f"Unknown model type: {model_type}")

    return model_dirs[model_type] / model_name


def get_run_output_dir(run_id: str) -> Path:
    """取得訓練執行輸出目錄"""
    paths = get_cache_paths()
    run_dir = paths.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_dataset_path(dataset_name: str) -> Path:
    """取得資料集目錄路徑"""
    paths = get_cache_paths()
    return paths.datasets / dataset_name


def get_output_path(output_type: str = "t2i") -> Path:
    """取得輸出目錄路徑"""
    paths = get_cache_paths()
    output_dir = paths.outputs / output_type
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ===== 快取初始化驗證 =====
def validate_cache_setup() -> Dict[str, Any]:
    """驗證快取設定是否正確"""
    try:
        paths = get_cache_paths()

        validation = {
            "cache_root_exists": paths.root.exists(),
            "hf_cache_configured": bool(os.environ.get("HF_HOME")),
            "torch_cache_configured": bool(os.environ.get("TORCH_HOME")),
            "gpu_available": False,
            "directories_created": True,
        }

        # 檢查 GPU
        try:
            import torch

            validation["gpu_available"] = torch.cuda.is_available()
            if validation["gpu_available"]:
                validation["gpu_count"] = torch.cuda.device_count()
                validation["gpu_name"] = torch.cuda.get_device_name(0)
        except ImportError:
            logger.warning("PyTorch not available for GPU check")

        # 檢查關鍵目錄
        required_dirs = [paths.models, paths.datasets, paths.cache, paths.outputs]
        for directory in required_dirs:
            if not directory.exists():
                validation["directories_created"] = False
                break

        validation["status"] = (
            "healthy"
            if all(
                [
                    validation["cache_root_exists"],
                    validation["hf_cache_configured"],
                    validation["directories_created"],
                ]
            )
            else "degraded"
        )

        return validation

    except Exception as e:
        logger.error(f"Cache validation failed: {e}")
        return {"status": "failed", "error": str(e)}


if __name__ == "__main__":
    # 測試配置
    print("=== CharaForge T2I Lab Configuration ===")

    validation = validate_cache_setup()
    print(f"Cache Status: {validation['status']}")
    print(f"Cache Root: {cache_paths.root}")
    print(f"GPU Available: {validation.get('gpu_available', False)}")

    if validation.get("gpu_available"):
        print(f"GPU: {validation.get('gpu_name', 'Unknown')}")

    app_config = config_manager.get_app_config()
    print(f"App: {app_config.get('app', {}).get('name', 'Unknown')}")
