# core/config.py - Unified configuration management
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class CachePaths:
    """Standardized cache paths from AI_CACHE_ROOT"""

    def __init__(self, cache_root: str):
        self.root = Path(cache_root)
        self.models = self.root / "models"
        self.datasets = self.root / "datasets"
        self.cache = self.root / "cache"
        self.runs = self.root / "runs"
        self.outputs = self.root / "outputs"

        # Model subdirectories
        self.models_sd = self.models / "sd"
        self.models_sdxl = self.models / "sdxl"
        self.models_controlnet = self.models / "controlnet"
        self.models_lora = self.models / "lora"
        self.models_ipadapter = self.models / "ipadapter"

        # Create directories if not exist
        self._ensure_dirs()

    def _ensure_dirs(self):
        """Create all cache directories"""
        dirs = [
            self.models,
            self.datasets,
            self.cache,
            self.runs,
            self.outputs,
            self.models_sd,
            self.models_sdxl,
            self.models_controlnet,
            self.models_lora,
            self.models_ipadapter,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Main application settings"""

    # Cache and paths
    ai_cache_root: str = Field(default="../ai_warehouse/cache", env="AI_CACHE_ROOT")

    # API settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_cors_origins: str = Field(
        default="http://localhost:3000", env="API_CORS_ORIGINS"
    )

    # Redis/Celery
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    celery_broker_url: str = Field(
        default="redis://localhost:6379/0", env="CELERY_BROKER_URL"
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/0", env="CELERY_RESULT_BACKEND"
    )

    # GPU/CUDA
    cuda_visible_devices: str = Field(default="0", env="CUDA_VISIBLE_DEVICES")

    # HuggingFace cache env vars
    def setup_hf_cache(self):
        """Setup HuggingFace cache environment variables"""
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

    def get_cache_paths(self) -> CachePaths:
        """Get standardized cache paths"""
        return CachePaths(self.ai_cache_root)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class ConfigManager:
    """YAML configuration manager"""

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self._cache = {}

    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load YAML config file with caching"""
        if filename in self._cache:
            return self._cache[filename]

        file_path = self.config_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self._cache[filename] = config
        return config

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get model configuration"""
        models_config = self.load_yaml("models.yaml")
        if model_name not in models_config:
            raise ValueError(f"Model config not found: {model_name}")
        return models_config[model_name]

    def get_train_config(self, config_name: str) -> Dict[str, Any]:
        """Get training configuration"""
        config_path = f"train/{config_name}"
        return self.load_yaml(config_path)

    def get_preset_config(self, preset_name: str) -> Dict[str, Any]:
        """Get style preset configuration"""
        config_path = f"presets/{preset_name}"
        return self.load_yaml(config_path)


# Global instances
settings = Settings()
config_manager = ConfigManager()

# Initialize HF cache on import
settings.setup_hf_cache()

# Export cache paths for easy access
cache_paths = settings.get_cache_paths()


def get_settings() -> Settings:
    """Get global settings instance"""
    return settings


def get_config_manager() -> ConfigManager:
    """Get global config manager instance"""
    return config_manager


def get_cache_paths() -> CachePaths:
    """Get global cache paths instance"""
    return cache_paths


# Convenience functions
def get_model_path(model_type: str, model_name: str) -> Path:
    """Get standardized model path"""
    paths = get_cache_paths()

    model_dirs = {
        "sd": paths.models_sd,
        "sdxl": paths.models_sdxl,
        "controlnet": paths.models_controlnet,
        "lora": paths.models_lora,
        "ipadapter": paths.models_ipadapter,
    }

    if model_type not in model_dirs:
        raise ValueError(f"Unknown model type: {model_type}")

    return model_dirs[model_type] / model_name


def get_run_output_dir(run_id: str) -> Path:
    """Get output directory for training run"""
    paths = get_cache_paths()
    run_dir = paths.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_dataset_path(dataset_name: str) -> Path:
    """Get dataset directory path"""
    paths = get_cache_paths()
    dataset_dir = paths.datasets / dataset_name
    return dataset_dir
