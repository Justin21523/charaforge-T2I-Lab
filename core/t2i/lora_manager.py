# core/t2i/lora_manager.py - LoRA loading and management
from typing import Dict, List, Tuple
import safetensors
from peft import LoraConfig, get_peft_model


class LoRAManager:
    """Manages LoRA loading, unloading and composition"""

    def __init__(self, pipeline_manager: PipelineManager):
        self.pipeline_manager = pipeline_manager
        self.loaded_loras: Dict[str, Dict] = (
            {}
        )  # lora_id -> {adapter, weight, metadata}
        self.cache_paths = get_cache_paths()

    def load_lora(self, lora_id: str, weight: float = 1.0) -> bool:
        """Load LoRA adapter"""
        if lora_id in self.loaded_loras:
            print(f"[LoRA] LoRA {lora_id} already loaded, updating weight to {weight}")
            self.loaded_loras[lora_id]["weight"] = weight
            return True

        lora_path = get_model_path("lora", lora_id)
        if not lora_path.exists():
            print(f"[LoRA] LoRA not found: {lora_path}")
            return False

        print(f"[LoRA] Loading LoRA: {lora_id} with weight {weight}")

        # TODO: Implement actual LoRA loading
        # adapter = load_lora_weights(lora_path)
        # Apply to current pipeline

        self.loaded_loras[lora_id] = {
            "path": lora_path,
            "weight": weight,
            "metadata": self._load_metadata(lora_path),
        }

        return True

    def unload_lora(self, lora_id: str) -> bool:
        """Unload LoRA adapter"""
        if lora_id not in self.loaded_loras:
            return False

        print(f"[LoRA] Unloading LoRA: {lora_id}")

        # TODO: Remove LoRA from pipeline
        del self.loaded_loras[lora_id]
        return True

    def list_loaded_loras(self) -> List[Dict]:
        """List currently loaded LoRAs"""
        return [
            {"lora_id": lora_id, "weight": info["weight"], "metadata": info["metadata"]}
            for lora_id, info in self.loaded_loras.items()
        ]

    def _load_metadata(self, lora_path: Path) -> Dict:
        """Load LoRA metadata from companion file"""
        metadata_path = lora_path.parent / f"{lora_path.stem}_metadata.json"
        if metadata_path.exists():
            import json

            with open(metadata_path, "r") as f:
                return json.load(f)
        return {}
