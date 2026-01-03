# core/config.py - Unified Configuration Management (AI_WAREHOUSE 3.0 compliant)
"""
統一設定管理系統

This project follows the workstation storage spec in `~/Desktop/data_model_structure.md`:
- Code lives under `/mnt/c/ai_projects`
- Models live under `/mnt/c/ai_models`
- AI caches live under `/mnt/c/ai_cache` (never under `~/.cache`)
- Datasets and outputs live under `/mnt/data`
"""

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

# ===== Path Configuration Objects =====


@dataclass(frozen=True)
class CachePaths:
    """Unified storage paths (cross-mount, AI_WAREHOUSE 3.0)."""

    # Cache root (2TB, /mnt/c)
    root: Path  # e.g. /mnt/c/ai_cache
    cache: Path  # alias of root for compatibility

    # Models (2TB, /mnt/c)
    models: Path  # e.g. /mnt/c/ai_models

    # Datasets and runs (4TB, /mnt/data)
    datasets: Path  # e.g. /mnt/data/datasets/<project_slug>
    runs: Path  # e.g. /mnt/data/training/runs/<project_slug>
    outputs: Path  # e.g. /mnt/data/training/runs/<project_slug>/outputs

    # AI Framework cache paths
    hf_home: Path
    torch_home: Path
    hf_hub: Path
    transformers: Path

    @classmethod
    def from_settings(cls, settings: "AppSettings") -> "CachePaths":
        """Create paths from settings (AI_WAREHOUSE 3.0)."""
        project_slug = settings.project_slug

        cache_root = Path(settings.ai_cache_root).resolve()
        models_root = Path(settings.ai_models_root).resolve()

        datasets_root = (Path(settings.ai_datasets_root).resolve() / project_slug).resolve()
        runs_root = (
            Path(settings.ai_training_root).resolve() / "runs" / project_slug
        ).resolve()
        outputs_root = (runs_root / "outputs").resolve()

        # Respect explicit env vars when present, but default to the prescribed paths.
        xdg_cache_home = Path(os.getenv("XDG_CACHE_HOME", str(cache_root))).resolve()
        hf_home = Path(os.getenv("HF_HOME", str(cache_root / "huggingface"))).resolve()
        transformers_cache = Path(
            os.getenv("TRANSFORMERS_CACHE", str(hf_home))
        ).resolve()
        torch_home = Path(os.getenv("TORCH_HOME", str(cache_root / "torch"))).resolve()
        hf_hub = Path(os.getenv("HF_HUB_CACHE", str(hf_home / "hub"))).resolve()

        # Keep internal root consistent with XDG_CACHE_HOME.
        cache_root = xdg_cache_home

        return cls(
            root=cache_root,
            cache=cache_root,
            models=models_root,
            datasets=datasets_root,
            runs=runs_root,
            outputs=outputs_root,
            hf_home=hf_home,
            transformers=transformers_cache,
            torch_home=torch_home,
            hf_hub=hf_hub,
        )

    def create_all(self) -> None:
        """Create all required directories."""
        for path in {
            self.root,
            self.cache,
            self.models,
            self.datasets,
            self.runs,
            self.outputs,
            self.hf_home,
            self.torch_home,
            self.hf_hub,
            self.transformers,
        }:
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class AppPaths:
    """Application specific directory paths - Fixed with outputs"""

    root: Path
    configs: Path
    outputs: Path  # Add missing outputs property

    # Model specific paths
    models_sd15: Path
    models_sdxl: Path
    models_controlnet: Path
    lora_weights: Path
    lora_weights_sdxl: Path
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
            outputs=cache_paths.outputs,  # Add outputs path
            # Model paths
            models_sd15=cache_paths.models / "stable-diffusion" / "sd15",
            models_sdxl=cache_paths.models / "stable-diffusion" / "sdxl",
            models_controlnet=cache_paths.models / "controlnet",
            lora_weights=cache_paths.models / "lora",
            lora_weights_sdxl=cache_paths.models / "lora_sdxl",
            embeddings=cache_paths.models / "embeddings",
            # Data paths
            datasets_raw=cache_paths.datasets / "raw",
            datasets_processed=cache_paths.datasets / "processed",
            training_runs=cache_paths.runs,
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
    """API configuration settings - Fixed with missing properties"""

    # Server settings
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1000, le=65535)
    workers: int = Field(default=1, ge=1, le=16)

    # CORS and security
    cors_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173"
    )
    api_key: Optional[str] = Field(default=None, validation_alias=AliasChoices("KEY", "API_KEY"))
    api_keys: str = Field(default="", validation_alias=AliasChoices("KEYS", "API_KEYS"))
    api_admin_keys: str = Field(default="", validation_alias=AliasChoices("ADMIN_KEYS", "API_ADMIN_KEYS"))
    key_header: str = Field(default="X-API-Key")
    ws_allow_query_auth: bool = Field(
        default=True,
        validation_alias=AliasChoices("WS_ALLOW_QUERY_AUTH", "API_WS_ALLOW_QUERY_AUTH"),
    )
    ws_ticket_ttl_seconds: int = Field(
        default=30,
        ge=0,
        validation_alias=AliasChoices("WS_TICKET_TTL_SECONDS", "API_WS_TICKET_TTL_SECONDS"),
    )
    ws_ticket_replay_protection: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "WS_TICKET_REPLAY_PROTECTION", "API_WS_TICKET_REPLAY_PROTECTION"
        ),
    )
    jwt_access_ttl_seconds: int = Field(
        default=900,
        ge=0,
        validation_alias=AliasChoices("JWT_ACCESS_TTL_SECONDS", "API_JWT_ACCESS_TTL_SECONDS"),
    )
    jwt_refresh_ttl_seconds: int = Field(
        default=2592000,
        ge=0,
        validation_alias=AliasChoices("JWT_REFRESH_TTL_SECONDS", "API_JWT_REFRESH_TTL_SECONDS"),
    )
    jwt_refresh_replay_window_seconds: int = Field(
        default=86400,
        ge=0,
        validation_alias=AliasChoices(
            "JWT_REFRESH_REPLAY_WINDOW_SECONDS", "API_JWT_REFRESH_REPLAY_WINDOW_SECONDS"
        ),
    )
    jwt_refresh_cookie_name: str = Field(
        default="cfr_refresh",
        validation_alias=AliasChoices("JWT_REFRESH_COOKIE_NAME", "API_JWT_REFRESH_COOKIE_NAME"),
    )
    jwt_csrf_cookie_name: str = Field(
        default="cfr_csrf",
        validation_alias=AliasChoices("JWT_CSRF_COOKIE_NAME", "API_JWT_CSRF_COOKIE_NAME"),
    )
    jwt_cookie_path: str = Field(
        default="/api/v1/auth",
        validation_alias=AliasChoices("JWT_COOKIE_PATH", "API_JWT_COOKIE_PATH"),
    )
    jwt_cookie_domain: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("JWT_COOKIE_DOMAIN", "API_JWT_COOKIE_DOMAIN"),
    )
    jwt_cookie_samesite: str = Field(
        default="lax",
        pattern="^(lax|strict|none)$",
        validation_alias=AliasChoices("JWT_COOKIE_SAMESITE", "API_JWT_COOKIE_SAMESITE"),
    )
    jwt_cookie_secure: bool = Field(
        default=False,
        validation_alias=AliasChoices("JWT_COOKIE_SECURE", "API_JWT_COOKIE_SECURE"),
    )
    rate_limit: int = Field(default=100, ge=0)  # requests per minute (0 disables)
    auth_token_rate_limit: int = Field(
        default=10,
        ge=0,
        validation_alias=AliasChoices("AUTH_TOKEN_RATE_LIMIT", "API_AUTH_TOKEN_RATE_LIMIT"),
    )
    auth_refresh_rate_limit: int = Field(
        default=30,
        ge=0,
        validation_alias=AliasChoices("AUTH_REFRESH_RATE_LIMIT", "API_AUTH_REFRESH_RATE_LIMIT"),
    )
    scan_rate_limit: int = Field(
        default=5, ge=0, validation_alias=AliasChoices("SCAN_RATE_LIMIT", "API_SCAN_RATE_LIMIT")
    )

    # Async T2I job settings
    t2i_worker_enabled: bool = Field(
        default=True, validation_alias=AliasChoices("T2I_WORKER_ENABLED", "API_T2I_WORKER_ENABLED")
    )
    t2i_dispatch_mode: str = Field(
        default="redis",
        pattern="^(redis|celery)$",
        validation_alias=AliasChoices("T2I_DISPATCH_MODE", "API_T2I_DISPATCH_MODE"),
    )
    t2i_job_ttl_seconds: int = Field(
        default=86400, ge=0, validation_alias=AliasChoices("T2I_JOB_TTL_SECONDS", "API_T2I_JOB_TTL_SECONDS")
    )
    t2i_job_stale_seconds: int = Field(
        default=600, ge=0, validation_alias=AliasChoices("T2I_JOB_STALE_SECONDS", "API_T2I_JOB_STALE_SECONDS")
    )
    t2i_job_max_attempts: int = Field(
        default=2, ge=1, validation_alias=AliasChoices("T2I_JOB_MAX_ATTEMPTS", "API_T2I_JOB_MAX_ATTEMPTS")
    )
    t2i_output_ttl_seconds: int = Field(
        default=0,
        ge=0,
        validation_alias=AliasChoices("T2I_OUTPUT_TTL_SECONDS", "API_T2I_OUTPUT_TTL_SECONDS"),
    )
    t2i_image_token_ttl_seconds: int = Field(
        default=3600,
        ge=0,
        validation_alias=AliasChoices(
            "T2I_IMAGE_TOKEN_TTL_SECONDS", "API_T2I_IMAGE_TOKEN_TTL_SECONDS"
        ),
    )
    t2i_max_global_concurrent: int = Field(
        default=1,
        ge=0,
        validation_alias=AliasChoices(
            "T2I_MAX_GLOBAL_CONCURRENT", "API_T2I_MAX_GLOBAL_CONCURRENT"
        ),
    )
    t2i_max_global_queue: int = Field(
        default=0,
        ge=0,
        validation_alias=AliasChoices("T2I_MAX_GLOBAL_QUEUE", "API_T2I_MAX_GLOBAL_QUEUE"),
    )
    t2i_max_concurrent: int = Field(
        default=1, ge=0, validation_alias=AliasChoices("T2I_MAX_CONCURRENT", "API_T2I_MAX_CONCURRENT")
    )
    t2i_max_queue: int = Field(
        default=8, ge=0, validation_alias=AliasChoices("T2I_MAX_QUEUE", "API_T2I_MAX_QUEUE")
    )
    t2i_cost_rate_limit: int = Field(
        default=0, ge=0, validation_alias=AliasChoices("T2I_COST_RATE_LIMIT", "API_T2I_COST_RATE_LIMIT")
    )

    # Bucket rate limits (requests/minute, 0 disables)
    upload_rate_limit: int = Field(
        default=30, ge=0, validation_alias=AliasChoices("UPLOAD_RATE_LIMIT", "API_UPLOAD_RATE_LIMIT")
    )
    datasets_rate_limit: int = Field(
        default=60, ge=0, validation_alias=AliasChoices("DATASETS_RATE_LIMIT", "API_DATASETS_RATE_LIMIT")
    )

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
        }
    )

    class Config:
        env_prefix = "CELERY_"


class AppSettings(BaseSettings):
    """Main application settings - Fixed with missing properties"""

    # Environment - Add missing properties
    environment: str = Field(
        default="development", pattern="^(development|staging|production)$"
    )
    debug: bool = Field(default=True)
    log_level: str = Field(
        default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )

    # Project identity (used to create per-project folders under /mnt/data)
    project_slug: str = Field(default="charaforge-t2i-lab")

    # AI_WAREHOUSE 3.0 roots
    ai_models_root: str = Field(default="/mnt/c/ai_models")
    ai_cache_root: str = Field(default="/mnt/c/ai_cache")
    ai_datasets_root: str = Field(default="/mnt/data/datasets")
    ai_training_root: str = Field(default="/mnt/data/training")

    # Redis URL - Add missing redis_url property
    redis_url: str = Field(default="redis://localhost:6379/0")

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
        _cache_paths_instance = CachePaths.from_settings(settings)
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
            _app_paths_instance.outputs,  # Add outputs to creation list
            _app_paths_instance.models_sd15,
            _app_paths_instance.models_sdxl,
            _app_paths_instance.models_controlnet,
            _app_paths_instance.lora_weights,
            _app_paths_instance.lora_weights_sdxl,
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


def get_run_output_dir(run_id: str, run_type: str = "training") -> Path:
    """Get output directory for a specific run - Add missing function"""
    app_paths = get_app_paths()

    if run_type == "training":
        return app_paths.training_runs / run_id
    elif run_type == "generation":
        return app_paths.outputs / "generation" / run_id
    elif run_type == "batch":
        return app_paths.outputs / "batch" / run_id
    elif run_type == "export":
        return app_paths.exports / run_id
    else:
        return app_paths.outputs / run_type / run_id


def setup_environment_variables() -> None:
    """Setup AI framework environment variables"""
    cache_paths = get_cache_paths()

    env_vars = {
        # Force caches out of $HOME per data_model_structure.md
        "XDG_CACHE_HOME": str(cache_paths.cache),
        "HF_HOME": str(cache_paths.hf_home),
        "TRANSFORMERS_CACHE": str(cache_paths.transformers),
        "HF_HUB_CACHE": str(cache_paths.hf_hub),
        "HUGGINGFACE_HUB_CACHE": str(cache_paths.hf_hub),
        "TORCH_HOME": str(cache_paths.torch_home),
        "PYTORCH_KERNEL_CACHE_PATH": str(cache_paths.torch_home / "kernels"),
        "PIP_CACHE_DIR": str(cache_paths.cache / "pip"),
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

    # Check disk space on configured roots (avoids probing unrelated mounts).
    try:
        import shutil

        check_paths = {
            cache_paths.root,
            cache_paths.models,
            cache_paths.datasets,
            cache_paths.runs,
            cache_paths.outputs,
        }

        for target in sorted(check_paths):
            if not target.exists():
                continue
            total, used, free = shutil.disk_usage(target)
            free_gb = free / (1024**3)
            if free_gb < 1.0:
                validation_results["errors"].append(
                    f"Very low disk space on {target}: {free_gb:.1f}GB free"
                )
            elif free_gb < 5.0:
                validation_results["warnings"].append(
                    f"Low disk space on {target}: {free_gb:.1f}GB free"
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
        print("🔧 Bootstrapping CharaForge T2I Lab configuration...")

    # Load settings
    settings = get_settings()
    cache_paths = get_cache_paths()
    get_app_paths()

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
        print("✅ Configuration bootstrap completed")
        print(f"   Cache root: {cache_paths.root}")
        print(f"   Environment: {settings.environment}")
        print(f"   Validation: {validation['status']}")

        if validation["errors"]:
            print(f"❌ Errors: {len(validation['errors'])}")
            for error in validation["errors"][:3]:  # Show first 3 errors
                print(f"   - {error}")

        if validation["warnings"]:
            print(f"⚠️ Warnings: {len(validation['warnings'])}")

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
        "project_slug": settings.project_slug,
        "cache_root": str(cache_paths.cache),
        "models_root": str(cache_paths.models),
        "datasets_root": str(cache_paths.datasets),
        "runs_root": str(cache_paths.runs),
        "outputs_root": str(cache_paths.outputs),
        "project_root": str(app_paths.root),
        "api_host": f"{settings.api.host}:{settings.api.port}",
        "celery_broker": settings.celery.broker_url,
        "redis_url": settings.redis_url,
        "framework_caches": {
            "HF_HOME": os.getenv("HF_HOME"),
            "TRANSFORMERS_CACHE": os.getenv("TRANSFORMERS_CACHE"),
            "TORCH_HOME": os.getenv("TORCH_HOME"),
            "XDG_CACHE_HOME": os.getenv("XDG_CACHE_HOME"),
        },
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
