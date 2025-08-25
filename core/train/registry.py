# core/train/registry.py - Model and training registry
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from core.config import get_cache_paths, get_run_output_dir


class ModelRegistry:
    """Registry for managing trained models and runs"""

    def __init__(self):
        self.cache_paths = get_cache_paths()
        self.registry_file = self.cache_paths.runs / "registry.json"
        self._ensure_registry_file()

    def _ensure_registry_file(self):
        """Create registry file if it doesn't exist"""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_file.exists():
            with open(self.registry_file, "w") as f:
                json.dump({"runs": {}, "models": {}}, f, indent=2)

    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from file"""
        with open(self.registry_file, "r") as f:
            return json.load(f)

    def _save_registry(self, registry: Dict[str, Any]):
        """Save registry to file"""
        with open(self.registry_file, "w") as f:
            json.dump(registry, f, indent=2)

    def register_run(
        self, run_id: str, config: Dict[str, Any], status: str = "started"
    ) -> Dict[str, Any]:
        """Register a new training run"""

        registry = self._load_registry()

        run_entry = {
            "run_id": run_id,
            "status": status,
            "config": config,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "output_dir": str(get_run_output_dir(run_id)),
            "metrics": {},
            "artifacts": {},
        }

        registry["runs"][run_id] = run_entry
        self._save_registry(registry)

        return run_entry

    def update_run_status(
        self,
        run_id: str,
        status: str,
        metrics: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, str]] = None,
    ):
        """Update run status and metrics"""

        registry = self._load_registry()

        if run_id not in registry["runs"]:
            raise ValueError(f"Run not found: {run_id}")

        run_entry = registry["runs"][run_id]
        run_entry["status"] = status
        run_entry["updated_at"] = datetime.now().isoformat()

        if metrics:
            run_entry["metrics"].update(metrics)

        if artifacts:
            run_entry["artifacts"].update(artifacts)

        self._save_registry(registry)

    def register_model(
        self,
        model_id: str,
        run_id: str,
        model_type: str,
        model_path: str,
        metadata: Dict[str, Any],
    ):
        """Register a completed model"""

        registry = self._load_registry()

        model_entry = {
            "model_id": model_id,
            "run_id": run_id,
            "model_type": model_type,
            "model_path": model_path,
            "metadata": metadata,
            "created_at": datetime.now().isoformat(),
            "status": "available",
        }

        registry["models"][model_id] = model_entry
        self._save_registry(registry)

        return model_entry

    def list_runs(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List training runs with optional status filter"""
        registry = self._load_registry()
        runs = list(registry["runs"].values())

        if status:
            runs = [run for run in runs if run["status"] == status]

        # Sort by created_at descending
        runs.sort(key=lambda x: x["created_at"], reverse=True)
        return runs

    def list_models(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available models with optional type filter"""
        registry = self._load_registry()
        models = list(registry["models"].values())

        if model_type:
            models = [model for model in models if model["model_type"] == model_type]

        # Sort by created_at descending
        models.sort(key=lambda x: x["created_at"], reverse=True)
        return models

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get specific run details"""
        registry = self._load_registry()
        return registry["runs"].get(run_id)

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get specific model details"""
        registry = self._load_registry()
        return registry["models"].get(model_id)
