# core/train/registry.py - Unified model and config registry
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import get_cache_paths
from core.file_lock import file_lock

logger = logging.getLogger(__name__)

REGISTRY_WRITE_LOCK_TIMEOUT_SECONDS = 5.0
SCAN_LOCK_TIMEOUT_SECONDS = 0.0


@dataclass
class ModelEntry:
    """Model registry entry"""

    name: str
    path: str
    model_type: str  # "sd15", "sdxl", "lora", "controlnet", "embedding"
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
        self._lock = threading.RLock()

        # Initialize registry data
        self.models: Dict[str, ModelEntry] = {}
        self.presets: Dict[str, TrainingPreset] = {}

        # Load existing registries
        self._load_registries()

        # NOTE: We avoid scanning the filesystem at import time. Use
        # `scan_filesystem()` (or the `/api/v1/models/scan` endpoint) when you
        # want to refresh the registry from `/mnt/c/ai_models`.

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

    def _locks_dir(self) -> Path:
        return self.cache_paths.cache / "locks"

    def _scan_lock_path(self) -> Path:
        return self._locks_dir() / "models_scan.lock"

    def _write_lock_path(self) -> Path:
        return self._locks_dir() / "model_registry_write.lock"

    def _atomic_write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        tmp.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(path)

    def _save_registries(self):
        """Save registries to disk"""
        try:
            with self._lock:
                models_payload = {
                    name: model.to_dict() for name, model in self.models.items()
                }
                presets_payload = {
                    name: preset.to_dict() for name, preset in self.presets.items()
                }

                with file_lock(
                    self._write_lock_path(),
                    timeout_s=REGISTRY_WRITE_LOCK_TIMEOUT_SECONDS,
                ):
                    self._atomic_write_json(
                        self.registry_path,
                        {
                            "models": models_payload,
                            "last_updated": datetime.now().isoformat(),
                        },
                    )
                    self._atomic_write_json(
                        self.presets_path,
                        {
                            "presets": presets_payload,
                            "last_updated": datetime.now().isoformat(),
                        },
                    )
        except Exception as e:
            logger.error(f"Error saving registries: {e}")

    def scan_filesystem(self, replace: bool = False) -> Dict[str, Any]:
        """Scan `/mnt/c/ai_models` and update `registry.json`.

        Args:
            replace: When True, rebuild the registry from disk (keeps presets).
        """
        try:
            with self._lock:
                with file_lock(
                    self._scan_lock_path(), timeout_s=SCAN_LOCK_TIMEOUT_SECONDS
                ):
                    before = len(self.models)

                    existing_usage = {
                        name: (entry.created_at, entry.last_used)
                        for name, entry in self.models.items()
                    }

                    if replace:
                        self.models = {}

                    added = 0
                    added += self._discover_base_models()
                    added += self._discover_controlnet_models()
                    added += self._discover_lora_models()
                    added += self._discover_embeddings()

                    # Preserve usage timestamps for unchanged model ids.
                    for name, entry in self.models.items():
                        created_at, last_used = existing_usage.get(name, (None, None))
                        if created_at and not entry.created_at:
                            entry.created_at = created_at
                        if last_used and not entry.last_used:
                            entry.last_used = last_used

                    self._save_registries()

                    return {
                        "status": "ok",
                        "replace": bool(replace),
                        "models_before": before,
                        "models_after": len(self.models),
                        "models_added": added,
                        "registry_path": str(self.registry_path),
                        "updated_at": datetime.now().isoformat(),
                    }
        except TimeoutError:
            return {
                "status": "busy",
                "error": "SCAN_IN_PROGRESS",
                "message": "Model scan already in progress",
                "registry_path": str(self.registry_path),
                "updated_at": datetime.now().isoformat(),
            }

    # Backwards compatible alias.
    def auto_discover_models(self) -> None:
        self.scan_filesystem(replace=False)

    def _upsert_model(self, entry: ModelEntry) -> bool:
        with self._lock:
            existing = self.models.get(entry.name)
            if existing:
                # Preserve human-meaningful timestamps from the registry.
                entry.created_at = existing.created_at or entry.created_at
                entry.last_used = existing.last_used or entry.last_used
                entry.metadata = existing.metadata or entry.metadata
            self.models[entry.name] = entry
            return existing is None

    def _discover_base_models(self) -> int:
        """Discover local Stable Diffusion base models under `/mnt/c/ai_models`."""
        discovered = 0

        roots = [
            ("sd15", self.cache_paths.models / "stable-diffusion" / "sd15"),
            ("sdxl", self.cache_paths.models / "stable-diffusion" / "sdxl"),
        ]

        for model_type, root in roots:
            if not root.exists():
                continue

            for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                if not (
                    (model_dir / "model_index.json").exists()
                    or (model_dir / "unet" / "config.json").exists()
                ):
                    continue

                model_name = f"{model_type}/{model_dir.name}"
                size_mb = self._calculate_dir_size(model_dir)

                entry = ModelEntry(
                    name=model_name,
                    path=str(model_dir),
                    model_type=model_type,
                    size_mb=size_mb,
                    description=f"Local {model_type.upper()} model",
                    tags=["base_model", model_type, "local"],
                    created_at=datetime.now().isoformat(),
                )

                if self._upsert_model(entry):
                    discovered += 1

        return discovered

    def _discover_lora_models(self) -> int:
        """Discover LoRA models under `/mnt/c/ai_models/lora*`."""
        discovered = 0

        roots = [
            (self.cache_paths.models / "lora", "lora/", []),
            (self.cache_paths.models / "lora_sdxl", "lora_sdxl/", ["sdxl"]),
        ]

        for root, prefix, extra_tags in roots:
            if not root.exists():
                continue

            for lora_path in sorted(root.rglob("*.safetensors")):
                model_name = f"{prefix}{lora_path.stem}"
                size_mb = lora_path.stat().st_size / (1024 * 1024)

                entry = ModelEntry(
                    name=model_name,
                    path=str(lora_path),
                    model_type="lora",
                    size_mb=size_mb,
                    description="Local LoRA model",
                    tags=["lora", "local", *extra_tags],
                    created_at=datetime.now().isoformat(),
                )

                if self._upsert_model(entry):
                    discovered += 1

        return discovered

    def _discover_controlnet_models(self) -> int:
        """Discover ControlNet models under `/mnt/c/ai_models/controlnet`."""
        discovered = 0
        root = self.cache_paths.models / "controlnet"
        if not root.exists():
            return 0

        # Diffusers-style folders
        for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if not (
                (model_dir / "config.json").exists()
                or (model_dir / "model_index.json").exists()
            ):
                continue
            model_name = f"controlnet/{model_dir.name}"
            size_mb = self._calculate_dir_size(model_dir)
            entry = ModelEntry(
                name=model_name,
                path=str(model_dir),
                model_type="controlnet",
                size_mb=size_mb,
                description="Local ControlNet model",
                tags=["controlnet", "local"],
                created_at=datetime.now().isoformat(),
            )
            if self._upsert_model(entry):
                discovered += 1

        # Single files (best-effort)
        for model_path in sorted(root.rglob("*.safetensors")):
            model_name = f"controlnet/{model_path.stem}"
            size_mb = model_path.stat().st_size / (1024 * 1024)
            entry = ModelEntry(
                name=model_name,
                path=str(model_path),
                model_type="controlnet",
                size_mb=size_mb,
                description="Local ControlNet weights",
                tags=["controlnet", "local"],
                created_at=datetime.now().isoformat(),
            )
            if self._upsert_model(entry):
                discovered += 1

        return discovered

    def _discover_embeddings(self) -> int:
        """Discover textual inversion embeddings under `/mnt/c/ai_models/embeddings`."""
        embeddings_dir = self.cache_paths.models / "embeddings"

        if not embeddings_dir.exists():
            return 0

        discovered = 0
        for embed_path in embeddings_dir.rglob("*.pt"):
            model_name = f"embedding/{embed_path.stem}"
            size_mb = embed_path.stat().st_size / (1024 * 1024)
            entry = ModelEntry(
                name=model_name,
                path=str(embed_path),
                model_type="embedding",
                size_mb=size_mb,
                description="Local embedding",
                tags=["embedding", "local"],
                created_at=datetime.now().isoformat(),
            )
            if self._upsert_model(entry):
                discovered += 1

        return discovered

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
            with self._lock:
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
        with self._lock:
            if name not in self.models:
                return
            self.models[name].last_used = datetime.now().isoformat()
        self._save_registries()

    def remove_model(self, name: str) -> bool:
        """Remove model from registry"""
        with self._lock:
            if name not in self.models:
                return False
            del self.models[name]
            self._save_registries()
            logger.info(f"Removed model: {name}")
            return True

    # Training preset methods
    def register_preset(self, preset: TrainingPreset) -> bool:
        """Register a training preset"""
        try:
            with self._lock:
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
            with self._lock:
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
