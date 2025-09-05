# core/config.py - Unified Configuration Management
"""
統一設定管理系統 - 支援 .env + configs/*.yaml
確保所有模組使用一致的設定與路徑管理
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from functools import lru_cache
import pydantic
from pydantic import Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

# ===== Path Configuration Objects =====


@dataclass(frozen=True)
class CachePaths:
    """Shared cache directory paths"""

    root: Path
    models: Path
    datasets: Path
    outputs: Path
    cache: Path
    runs: Path

    # AI Framework cache paths
    hf_home: Path
    torch_home: Path
    hf_hub: Path
    transformers: Path

    @classmethod
    def from_root(cls, cache_root: str) -> "CachePaths":
        """Create cache paths from root directory"""
        root = Path(cache_root).resolve()

        return cls(
            root=root,
            models=root / "models",
            datasets=root / "datasets",
            outputs=root / "outputs",
            cache=root / "cache",
            runs=root / "runs",
            hf_home=root / "cache" / "hf",
            torch_home=root / "cache" / "torch",
            hf_hub=root / "cache" / "hf" / "hub",
            transformers=root / "cache" / "hf" / "transformers",
        )

    def create_all(self) -> None:
        """Create all cache directories"""
        for path in [
            self.models,
            self.datasets,
            self.outputs,
            self.cache,
            self.runs,
            self.hf_home,
            self.torch_home,
            self.hf_hub,
            self.transformers,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class AppPaths:
    """Application specific directory paths"""

    root: Path
    configs: Path

    # Model specific paths
    models_sd15: Path
    models_sdxl: Path
    models_controlnet: Path
    lora_weights: Path
    embeddings: Path

    # Data paths
    datasets_raw: Path
    datasets_processed: Path
    training_runs: Path
    exports: Path

    @classmethod
    def from_cache_paths(
        cls, cache_paths: CachePaths, project_root: Optional[Path] = None
    ) -> "AppPaths":
        """Create app paths from cache paths"""
        if project_root is None:
            project_root = Path(__file__).parent.parent

        return cls(
            root=project_root,
            configs=project_root / "configs",
            # Model paths
            models_sd15=cache_paths.models / "sd15",
            models_sdxl=cache_paths.models / "sdxl",
            models_controlnet=cache_paths.models / "controlnet",
            lora_weights=cache_paths.models / "lora",
            embeddings=cache_paths.models / "embeddings",
            # Data paths
            datasets_raw=cache_paths.datasets / "raw",
            datasets_processed=cache_paths.datasets / "processed",
            training_runs=cache_paths.runs / "training",
            exports=cache_paths.outputs / "exports",
        )


# ===== Pydantic Settings Classes =====


class ModelConfig(BaseSettings):
    """Model configuration settings"""

    # Default models
    default_sd15_model: str = Field(default="runwayml/stable-diffusion-v1-5")
    default_sdxl_model: str = Field(default="stabilityai/stable-diffusion-xl-base-1.0")
    default_vae: Optional[str] = Field(default=None)

    # Generation defaults
    default_steps: int = Field(default=20, ge=1, le=150)
    default_guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    default_width: int = Field(default=512, ge=256, le=2048)
    default_height: int = Field(default=512, ge=256, le=2048)

    # Performance settings
    enable_xformers: bool = Field(default=True)
    enable_cpu_offload: bool = Field(default=False)
    low_vram_mode: bool = Field(default=False)
    use_fp16: bool = Field(default=True)
    use_bf16: bool = Field(default=False)

    class Config:
        env_prefix = "MODEL_"


class TrainingConfig(BaseSettings):
    """Training configuration settings"""

    # LoRA defaults
    lora_rank: int = Field(default=16, ge=4, le=128)
    lora_alpha: int = Field(default=32, ge=8, le=256)
    lora_dropout: float = Field(default=0.1, ge=0.0, le=0.5)

    # Training hyperparameters
    learning_rate: float = Field(default=1e-4, ge=1e-6, le=1e-2)
    train_batch_size: int = Field(default=1, ge=1, le=8)
    num_train_epochs: int = Field(default=10, ge=1, le=1000)
    max_train_steps: Optional[int] = Field(default=None)

    # Optimization
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    gradient_checkpointing: bool = Field(default=True)
    mixed_precision: str = Field(default="fp16", pattern="^(no|fp16|bf16)$")

    # Validation
    validation_steps: int = Field(default=100, ge=10)
    save_steps: int = Field(default=500, ge=50)

    class Config:
        env_prefix = "TRAIN_"


class APIConfig(BaseSettings):
    """API configuration settings"""

    # Server settings
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1000, le=65535)
    workers: int = Field(default=1, ge=1, le=16)

    # CORS and security
    cors_origins: str = Field(default="http://localhost:3000,http://127.0.0.1:3000")
    api_key: Optional[str] = Field(default=None)
    rate_limit: int = Field(default=100, ge=1)  # requests per minute

    # File uploads
    max_file_size_mb: int = Field(default=50, ge=1, le=500)
    allowed_extensions: str = Field(default=".jpg,.jpeg,.png,.webp")

    # Generation limits
    max_batch_size: int = Field(default=4, ge=1, le=16)
    max_steps: int = Field(default=150, ge=10, le=500)

    class Config:
        env_prefix = "API_"


class CeleryConfig(BaseSettings):
    """Celery worker configuration"""

    broker_url: str = Field(default="redis://localhost:6379/0")
    result_backend: str = Field(default="redis://localhost:6379/0")

    # Worker settings
    worker_concurrency: int = Field(default=1, ge=1, le=8)
    worker_prefetch_multiplier: int = Field(default=1, ge=1)
    task_soft_time_limit: int = Field(default=3600)  # 1 hour
    task_time_limit: int = Field(default=7200)  # 2 hours

    # Queue settings
    task_routes: Dict[str, str] = Field(
        default_factory=lambda: {
            "workers.tasks.training.*": "training",
            "workers.tasks.generation.*": "generation",
            "workers.tasks.batch.*": "batch",
        }
    )

    class Config:
        env_prefix = "CELERY_"


class AppSettings(BaseSettings):
    """Main application settings"""

    # Environment
    environment: str = Field(
        default="development", pattern="^(development|staging|production)$"
    )
    debug: bool = Field(default=True)
    log_level: str = Field(
        default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )

    # Cache configuration
    ai_cache_root: str = Field(default="../ai_warehouse/cache")

    # Component settings
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    celery: CeleryConfig = Field(default_factory=CeleryConfig)

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"


# ===== YAML Configuration Loading =====


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file"""
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Failed to load config {config_path}: {e}")
        return {}


def load_model_presets(configs_dir: Path) -> Dict[str, Any]:
    """Load model preset configurations"""
    presets = {}
    presets_dir = configs_dir / "presets"

    if not presets_dir.exists():
        return presets

    for preset_file in presets_dir.glob("*.yaml"):
        preset_name = preset_file.stem
        presets[preset_name] = load_yaml_config(preset_file)

    return presets


def load_training_configs(configs_dir: Path) -> Dict[str, Any]:
    """Load training configuration templates"""
    training_configs = {}
    train_dir = configs_dir / "train"

    if not train_dir.exists():
        return training_configs

    for config_file in train_dir.glob("*.yaml"):
        config_name = config_file.stem
        training_configs[config_name] = load_yaml_config(config_file)

    return training_configs


# ===== Global Settings Management =====

_settings_instance: Optional[AppSettings] = None
_cache_paths_instance: Optional[CachePaths] = None
_app_paths_instance: Optional[AppPaths] = None


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Get global application settings (cached)"""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = AppSettings()

        # Setup logging level
        logging.getLogger().setLevel(_settings_instance.log_level)

        logger.info(f"Settings loaded - Environment: {_settings_instance.environment}")

    return _settings_instance


@lru_cache(maxsize=1)
def get_cache_paths() -> CachePaths:
    """Get cache directory paths (cached)"""
    global _cache_paths_instance
    if _cache_paths_instance is None:
        settings = get_settings()
        _cache_paths_instance = CachePaths.from_root(settings.ai_cache_root)
        _cache_paths_instance.create_all()

        logger.info(f"Cache paths initialized: {_cache_paths_instance.root}")

    return _cache_paths_instance


@lru_cache(maxsize=1)
def get_app_paths() -> AppPaths:
    """Get application directory paths (cached)"""
    global _app_paths_instance
    if _app_paths_instance is None:
        cache_paths = get_cache_paths()
        _app_paths_instance = AppPaths.from_cache_paths(cache_paths)

        # Create app directories
        for path in [
            _app_paths_instance.configs,
            _app_paths_instance.models_sd15,
            _app_paths_instance.models_sdxl,
            _app_paths_instance.models_controlnet,
            _app_paths_instance.lora_weights,
            _app_paths_instance.embeddings,
            _app_paths_instance.datasets_raw,
            _app_paths_instance.datasets_processed,
            _app_paths_instance.training_runs,
            _app_paths_instance.exports,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        logger.info(f"App paths initialized: {_app_paths_instance.root}")

    return _app_paths_instance


def get_model_config(model_name: Optional[str] = None) -> Dict[str, Any]:
    """Get model configuration with optional preset override"""
    settings = get_settings()
    app_paths = get_app_paths()

    # Load base model config
    config = settings.model.dict()

    # Load presets
    presets = load_model_presets(app_paths.configs)

    # Apply specific preset if requested
    if model_name and model_name in presets:
        config.update(presets[model_name])

    return config


def get_training_config(config_name: Optional[str] = None) -> Dict[str, Any]:
    """Get training configuration with optional template override"""
    settings = get_settings()
    app_paths = get_app_paths()

    # Load base training config
    config = settings.training.dict()

    # Load training templates
    templates = load_training_configs(app_paths.configs)

    # Apply specific template if requested
    if config_name and config_name in templates:
        config.update(templates[config_name])

    return config


def setup_environment_variables() -> None:
    """Setup AI framework environment variables"""
    cache_paths = get_cache_paths()

    env_vars = {
        "HF_HOME": str(cache_paths.hf_home),
        "TRANSFORMERS_CACHE": str(cache_paths.transformers),
        "HF_HUB_CACHE": str(cache_paths.hf_hub),
        "TORCH_HOME": str(cache_paths.torch_home),
        "PYTORCH_KERNEL_CACHE_PATH": str(cache_paths.torch_home / "kernels"),
    }

    for key, value in env_vars.items():
        os.environ[key] = value
        Path(value).mkdir(parents=True, exist_ok=True)

    logger.info("AI framework environment variables configured")


def validate_cache_setup() -> Dict[str, Any]:
    """Validate cache directory setup and permissions"""
    cache_paths = get_cache_paths()
    validation_results = {
        "status": "healthy",
        "errors": [],
        "warnings": [],
        "paths_checked": [],
    }

    # Check all required paths
    required_paths = [
        cache_paths.root,
        cache_paths.models,
        cache_paths.datasets,
        cache_paths.outputs,
        cache_paths.cache,
        cache_paths.runs,
    ]

    for path in required_paths:
        result = {
            "path": str(path),
            "exists": path.exists(),
            "writable": False,
            "readable": False,
        }

        if path.exists():
            try:
                # Test write permission
                test_file = path / ".write_test"
                test_file.touch()
                test_file.unlink()
                result["writable"] = True
            except Exception:
                validation_results["errors"].append(f"Cannot write to {path}")

            try:
                # Test read permission
                list(path.iterdir())
                result["readable"] = True
            except Exception:
                validation_results["errors"].append(f"Cannot read from {path}")
        else:
            validation_results["errors"].append(f"Path does not exist: {path}")

        validation_results["paths_checked"].append(result)

    # Check disk space
    try:
        import shutil

        total, used, free = shutil.disk_usage(cache_paths.root)
        free_gb = free / (1024**3)

        if free_gb < 5.0:  # Less than 5GB free
            validation_results["warnings"].append(
                f"Low disk space: {free_gb:.1f}GB free"
            )
        elif free_gb < 1.0:  # Less than 1GB free
            validation_results["errors"].append(
                f"Very low disk space: {free_gb:.1f}GB free"
            )

    except Exception as e:
        validation_results["warnings"].append(f"Could not check disk space: {e}")

    # Determine overall status
    if validation_results["errors"]:
        validation_results["status"] = "error"
    elif validation_results["warnings"]:
        validation_results["status"] = "warning"

    return validation_results


# ===== Bootstrap Function =====


def bootstrap_config(verbose: bool = False) -> Dict[str, Any]:
    """Bootstrap the entire configuration system"""
    if verbose:
        print("🔧 Bootstrapping SagaForge T2I Lab configuration...")

    # Load settings
    settings = get_settings()
    cache_paths = get_cache_paths()
    app_paths = get_app_paths()

    # Setup environment variables
    setup_environment_variables()

    # Validate setup
    validation = validate_cache_setup()

    summary = {
        "settings_loaded": True,
        "cache_root": str(cache_paths.root),
        "environment": settings.environment,
        "debug_mode": settings.debug,
        "validation": validation,
        "ai_framework_vars_set": True,
    }

    if verbose:
        print(f"✅ Configuration bootstrap completed")
        print(f"   Cache root: {cache_paths.root}")
        print(f"   Environment: {settings.environment}")
        print(f"   Validation: {validation['status']}")

        if validation["errors"]:
            print(f"❌ Errors: {len(validation['errors'])}")
            for error in validation["errors"][:3]:  # Show first 3 errors
                print(f"   - {error}")

        if validation["warnings"]:
            print(f"⚠️  Warnings: {len(validation['warnings'])}")

    return summary


# ===== Utility Functions =====


def reset_settings_cache() -> None:
    """Reset cached settings (useful for testing)"""
    global _settings_instance, _cache_paths_instance, _app_paths_instance
    _settings_instance = None
    _cache_paths_instance = None
    _app_paths_instance = None

    # Clear LRU caches
    get_settings.cache_clear()
    get_cache_paths.cache_clear()
    get_app_paths.cache_clear()


def get_config_summary() -> Dict[str, Any]:
    """Get a summary of current configuration"""
    settings = get_settings()
    cache_paths = get_cache_paths()
    app_paths = get_app_paths()

    return {
        "environment": settings.environment,
        "debug": settings.debug,
        "cache_root": str(cache_paths.root),
        "project_root": str(app_paths.root),
        "api_host": f"{settings.api.host}:{settings.api.port}",
        "celery_broker": settings.celery.broker_url,
        "model_defaults": {
            "sd15": settings.model.default_sd15_model,
            "sdxl": settings.model.default_sdxl_model,
        },
        "performance": {
            "low_vram": settings.model.low_vram_mode,
            "fp16": settings.model.use_fp16,
            "xformers": settings.model.enable_xformers,
        },
    }
