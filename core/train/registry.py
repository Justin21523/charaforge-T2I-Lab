# core/train/registry.py - Unified model and config registry
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import json
import yaml
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)

from core.config import get_cache_paths, get_model_config


logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    """Model registry entry"""

    name: str
    path: str
    model_type: str  # "sd15", "sdxl", "lora", "embedding"
    size_mb: Optional[float] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    created_at: Optional[str] = None
    last_used: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingPreset:
    """Training configuration preset"""

    name: str
    target_type: str  # "lora", "dreambooth", "controlnet"
    base_model: str
    config: Dict[str, Any]
    description: Optional[str] = None
    tags: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelRegistry:
    """Central registry for models, LoRAs, and training presets"""

    def __init__(self):
        self.cache_paths = get_cache_paths()
        self.registry_path = self.cache_paths.models / "registry.json"
        self.presets_path = self.cache_paths.models / "presets.json"

        # Initialize registry data
        self.models: Dict[str, ModelEntry] = {}
        self.presets: Dict[str, TrainingPreset] = {}

        # Load existing registries
        self._load_registries()

        # Auto-discover models on initialization
        self.auto_discover_models()

    def _load_registries(self):
        """Load existing registry files"""
        try:
            if self.registry_path.exists():
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.models = {
                        name: ModelEntry(**entry)
                        for name, entry in data.get("models", {}).items()
                    }
                logger.info(f"Loaded {len(self.models)} models from registry")

            if self.presets_path.exists():
                with open(self.presets_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.presets = {
                        name: TrainingPreset(**preset)
                        for name, preset in data.get("presets", {}).items()
                    }
                logger.info(f"Loaded {len(self.presets)} training presets")

        except Exception as e:
            logger.warning(f"Error loading registries: {e}")

    def _save_registries(self):
        """Save registries to disk"""
        try:
            # Save models registry
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "models": {
                            name: model.to_dict() for name, model in self.models.items()
                        },
                        "last_updated": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            # Save presets registry
            with open(self.presets_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "presets": {
                            name: preset.to_dict()
                            for name, preset in self.presets.items()
                        },
                        "last_updated": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

        except Exception as e:
            logger.error(f"Error saving registries: {e}")

    def auto_discover_models(self):
        """Auto-discover models in cache directories"""
        logger.info("Auto-discovering models...")

        # Discover base models
        self._discover_base_models()

        # Discover LoRA models
        self._discover_lora_models()

        # Discover embeddings
        self._discover_embeddings()

        # Save updated registry
        self._save_registries()

    def _discover_base_models(self):
        """Discover StableDiffusion base models"""
        hf_models_dir = self.cache_paths.hf_home / "hub"

        if not hf_models_dir.exists():
            return

        for model_dir in hf_models_dir.iterdir():
            if not model_dir.is_dir():
                continue

            # Check if it's a diffusion model
            unet_config = model_dir / "unet" / "config.json"
            model_index = model_dir / "model_index.json"

            if unet_config.exists() or model_index.exists():
                model_name = model_dir.name.replace("models--", "").replace("--", "/")

                if model_name not in self.models:
                    # Determine model type
                    model_type = "sdxl" if "xl" in model_name.lower() else "sd15"

                    # Calculate size
                    size_mb = self._calculate_dir_size(model_dir)

                    entry = ModelEntry(
                        name=model_name,
                        path=str(model_dir),
                        model_type=model_type,
                        size_mb=size_mb,
                        description=f"Auto-discovered {model_type.upper()} model",
                        tags=["base_model", model_type],
                        created_at=datetime.now().isoformat(),
                    )

                    self.models[model_name] = entry
                    logger.debug(f"Discovered model: {model_name}")

    def _discover_lora_models(self):
        """Discover LoRA models"""
        lora_dirs = [
            self.cache_paths.models / "lora",
            self.cache_paths.runs,  # Training outputs
        ]

        for lora_dir in lora_dirs:
            if not lora_dir.exists():
                continue

            for lora_path in lora_dir.rglob("*.safetensors"):
                if "lora" in lora_path.name.lower() or lora_path.parent.name == "lora":
                    model_name = f"lora/{lora_path.stem}"

                    if model_name not in self.models:
                        size_mb = lora_path.stat().st_size / (1024 * 1024)

                        entry = ModelEntry(
                            name=model_name,
                            path=str(lora_path),
                            model_type="lora",
                            size_mb=size_mb,
                            description="Auto-discovered LoRA model",
                            tags=["lora"],
                            created_at=datetime.now().isoformat(),
                        )

                        self.models[model_name] = entry
                        logger.debug(f"Discovered LoRA: {model_name}")

    def _discover_embeddings(self):
        """Discover textual inversion embeddings"""
        embeddings_dir = self.cache_paths.models / "embeddings"

        if not embeddings_dir.exists():
            return

        for embed_path in embeddings_dir.rglob("*.pt"):
            model_name = f"embedding/{embed_path.stem}"

            if model_name not in self.models:
                size_mb = embed_path.stat().st_size / (1024 * 1024)

                entry = ModelEntry(
                    name=model_name,
                    path=str(embed_path),
                    model_type="embedding",
                    size_mb=size_mb,
                    description="Auto-discovered embedding",
                    tags=["embedding"],
                    created_at=datetime.now().isoformat(),
                )

                self.models[model_name] = entry

    def _calculate_dir_size(self, directory: Path) -> float:
        """Calculate directory size in MB"""
        try:
            total_size = sum(
                f.stat().st_size for f in directory.rglob("*") if f.is_file()
            )
            return total_size / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0

    # Model management methods
    def register_model(self, entry: ModelEntry) -> bool:
        """Register a new model"""
        try:
            self.models[entry.name] = entry
            self._save_registries()
            logger.info(f"Registered model: {entry.name}")
            return True
        except Exception as e:
            logger.error(f"Error registering model {entry.name}: {e}")
            return False

    def get_model(self, name: str) -> Optional[ModelEntry]:
        """Get model by name"""
        return self.models.get(name)

    def list_models(
        self, model_type: Optional[str] = None, tags: Optional[List[str]] = None
    ) -> List[ModelEntry]:
        """List models with optional filtering"""
        models = list(self.models.values())

        if model_type:
            models = [m for m in models if m.model_type == model_type]

        if tags:
            models = [
                m for m in models if m.tags and any(tag in m.tags for tag in tags)
            ]

        return models

    def update_model_usage(self, name: str):
        """Update last used timestamp for a model"""
        if name in self.models:
            self.models[name].last_used = datetime.now().isoformat()
            self._save_registries()

    def remove_model(self, name: str) -> bool:
        """Remove model from registry"""
        if name in self.models:
            del self.models[name]
            self._save_registries()
            logger.info(f"Removed model: {name}")
            return True
        return False

    # Training preset methods
    def register_preset(self, preset: TrainingPreset) -> bool:
        """Register a training preset"""
        try:
            self.presets[preset.name] = preset
            self._save_registries()
            logger.info(f"Registered preset: {preset.name}")
            return True
        except Exception as e:
            logger.error(f"Error registering preset {preset.name}: {e}")
            return False

    def get_preset(self, name: str) -> Optional[TrainingPreset]:
        """Get training preset by name"""
        return self.presets.get(name)

    def list_presets(self, target_type: Optional[str] = None) -> List[TrainingPreset]:
        """List training presets with optional filtering"""
        presets = list(self.presets.values())

        if target_type:
            presets = [p for p in presets if p.target_type == target_type]

        return presets

    def load_default_presets(self):
        """Load default training presets"""
        default_presets = [
            TrainingPreset(
                name="lora_default_sd15",
                target_type="lora",
                base_model="runwayml/stable-diffusion-v1-5",
                config={
                    "num_epochs": 10,
                    "learning_rate": 1e-4,
                    "rank": 16,
                    "alpha": 32,
                    "train_batch_size": 1,
                    "gradient_accumulation_steps": 4,
                    "max_train_steps": 1000,
                    "save_steps": 500,
                    "checkpointing_steps": 500,
                    "validation_steps": 100,
                },
                description="Default LoRA training preset for SD1.5",
                tags=["lora", "sd15", "default"],
            ),
            TrainingPreset(
                name="lora_default_sdxl",
                target_type="lora",
                base_model="stabilityai/stable-diffusion-xl-base-1.0",
                config={
                    "num_epochs": 8,
                    "learning_rate": 8e-5,
                    "rank": 32,
                    "alpha": 64,
                    "train_batch_size": 1,
                    "gradient_accumulation_steps": 8,
                    "max_train_steps": 1200,
                    "save_steps": 300,
                    "checkpointing_steps": 300,
                    "validation_steps": 150,
                },
                description="Default LoRA training preset for SDXL",
                tags=["lora", "sdxl", "default"],
            ),
            TrainingPreset(
                name="dreambooth_default",
                target_type="dreambooth",
                base_model="runwayml/stable-diffusion-v1-5",
                config={
                    "num_epochs": 15,
                    "learning_rate": 5e-6,
                    "train_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "max_train_steps": 800,
                    "prior_preservation": True,
                    "prior_loss_weight": 1.0,
                    "class_data_dir": None,  # Will be set during training
                },
                description="Default DreamBooth training preset",
                tags=["dreambooth", "sd15", "default"],
            ),
        ]

        for preset in default_presets:
            if preset.name not in self.presets:
                self.presets[preset.name] = preset

        self._save_registries()
        logger.info(f"Loaded {len(default_presets)} default presets")

    # Utility methods
    def get_model_path(self, name: str) -> Optional[Path]:
        """Get full path to model"""
        model = self.get_model(name)
        return Path(model.path) if model else None

    def validate_model_exists(self, name: str) -> bool:
        """Check if model exists and is accessible"""
        model_path = self.get_model_path(name)
        return model_path is not None and model_path.exists()

    def get_compatible_models(self, target_type: str) -> List[ModelEntry]:
        """Get models compatible with specific training type"""
        if target_type in ["lora", "dreambooth"]:
            return self.list_models(model_type="sd15") + self.list_models(
                model_type="sdxl"
            )
        elif target_type == "controlnet":
            return self.list_models(model_type="sd15")  # ControlNet mainly for SD1.5
        else:
            return []

    def search_models(self, query: str) -> List[ModelEntry]:
        """Search models by name, description, or tags"""
        query = query.lower()
        results = []

        for model in self.models.values():
            if (
                query in model.name.lower()
                or (model.description and query in model.description.lower())
                or (model.tags and any(query in tag.lower() for tag in model.tags))
            ):
                results.append(model)

        return results


# Global registry instance
_registry_instance = None


def get_model_registry() -> ModelRegistry:
    """Get global model registry instance"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
    return _registry_instance
