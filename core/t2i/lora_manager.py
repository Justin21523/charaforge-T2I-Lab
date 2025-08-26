# core/t2i/lora_manager.py - LoRA loading and management
from __future__ import annotations
import os, tempfile, shutil
from typing import Dict, List, Tuple
import safetensors
from pathlib import Path

from core.config import get_cache_paths, get_model_path
from core.t2i.pipeline import PipelineManager
from core.train.lora_trainer import LoRATrainer

import json
import torch
import safetensors.torch
from peft import LoraConfig, get_peft_model, PeftModel
from diffusers.loaders.utils import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor


class LoRAManager:
    """Manages LoRA loading, unloading and composition"""

    def __init__(self, pipeline_manager: PipelineManager):
        self.pipeline_manager = pipeline_manager
        self.loaded_loras: Dict[str, Dict] = (
            {}
        )  # lora_id -> {adapter, weight, metadata}
        self.cache_paths = get_cache_paths()
        self.original_attn_procs = {}  # Store original attention processors

    def _get_lora_path(self, lora_id: str) -> Path:
        """Get LoRA model path"""
        # Try different possible locations
        possible_paths = [
            get_model_path("lora", lora_id),  # Direct path
            get_model_path("lora", f"{lora_id}.safetensors"),  # With extension
            self.cache_paths.runs
            / lora_id
            / "checkpoints"
            / "final",  # Training output
        ]

        for path in possible_paths:
            if path.exists():
                return path
            # Check for safetensors file
            safetensors_path = path.with_suffix(".safetensors")
            if safetensors_path.exists():
                return safetensors_path

        raise FileNotFoundError(f"LoRA not found: {lora_id}")

    def _load_lora_weights(self, lora_path: Path) -> Dict[str, torch.Tensor]:
        """Load LoRA weights from file"""
        if lora_path.suffix == ".safetensors":
            return safetensors.torch.load_file(lora_path, device="cpu")
        elif lora_path.suffix == ".bin":
            return torch.load(lora_path, map_location="cpu")
        else:
            # Try to load as directory (HuggingFace format)
            adapter_config_path = lora_path / "adapter_config.json"
            if adapter_config_path.exists():
                # Load using PEFT
                return PeftModel.from_pretrained(None, lora_path)  # type: ignore
            else:
                raise ValueError(f"Unsupported LoRA format: {lora_path}")

    def _apply_lora_weights(
        self, pipeline, lora_weights: Dict[str, torch.Tensor], alpha: float = 1.0
    ):
        """Apply LoRA weights to pipeline UNet"""
        unet = pipeline.unet

        # Store original processors if not already stored
        if not self.original_attn_procs:
            self.original_attn_procs = unet.attn_processors

        # Create new attention processors with LoRA
        lora_attn_procs = {}

        for name, processor in unet.attn_processors.items():
            # Check if we have LoRA weights for this attention layer
            lora_key_q = f"unet.{name.replace('.processor', '')}.to_q.lora_A.weight"
            lora_key_k = f"unet.{name.replace('.processor', '')}.to_k.lora_A.weight"
            lora_key_v = f"unet.{name.replace('.processor', '')}.to_v.lora_A.weight"
            lora_key_out = (
                f"unet.{name.replace('.processor', '')}.to_out.0.lora_A.weight"
            )

            has_lora = any(
                key in lora_weights
                for key in [lora_key_q, lora_key_k, lora_key_v, lora_key_out]
            )

            if has_lora:
                # Create LoRA attention processor
                lora_attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=(
                        processor.hidden_size
                        if hasattr(processor, "hidden_size")
                        else 1024
                    ),
                    cross_attention_dim=(
                        processor.cross_attention_dim
                        if hasattr(processor, "cross_attention_dim")
                        else None
                    ),
                    rank=16,  # Default rank, should be read from config
                    network_alpha=alpha,
                )
            else:
                lora_attn_procs[name] = processor

        # Apply new processors
        unet.set_attn_processor(lora_attn_procs)

        # Load actual LoRA weights into the processors
        for name, weights in lora_weights.items():
            # Parse the name to find the correct attention processor
            # This is a simplified version - full implementation would need
            # proper name parsing and weight assignment
            pass

    def load_lora(self, lora_id: str, weight: float = 1.0) -> bool:
        """Load LoRA adapter"""
        if lora_id in self.loaded_loras:
            print(f"[LoRA] LoRA {lora_id} already loaded, updating weight to {weight}")
            self.loaded_loras[lora_id]["weight"] = weight
            self._reapply_loras()
            return True

        try:
            lora_path = self._get_lora_path(lora_id)
            print(f"[LoRA] Loading LoRA: {lora_id} from {lora_path}")

            # Load LoRA weights
            lora_weights = self._load_lora_weights(lora_path)

            # Load metadata
            metadata = self._load_metadata(lora_path)

            # Store LoRA info
            self.loaded_loras[lora_id] = {
                "path": lora_path,
                "weight": weight,
                "weights": lora_weights,
                "metadata": metadata,
            }

            # Apply to current pipeline if available
            if self.pipeline_manager.current_pipeline:
                pipeline = self.pipeline_manager.pipelines[
                    self.pipeline_manager.current_pipeline
                ]
                self._apply_lora_weights(pipeline, lora_weights, weight)

            print(f"[LoRA] Successfully loaded: {lora_id}")
            return True

        except Exception as e:
            print(f"[LoRA] Failed to load {lora_id}: {e}")
            return False

    def unload_lora(self, lora_id: str) -> bool:
        """Unload LoRA adapter"""
        if lora_id not in self.loaded_loras:
            return False

        print(f"[LoRA] Unloading LoRA: {lora_id}")

        # Remove from loaded list
        del self.loaded_loras[lora_id]

        # Reapply remaining LoRAs or restore original
        self._reapply_loras()

        return True

    def _reapply_loras(self):
        """Reapply all currently loaded LoRAs"""
        if not self.pipeline_manager.current_pipeline:
            return

        pipeline = self.pipeline_manager.pipelines[
            self.pipeline_manager.current_pipeline
        ]

        if not self.loaded_loras:
            # Restore original attention processors
            if self.original_attn_procs:
                pipeline.unet.set_attn_processor(self.original_attn_procs)
        else:
            # Combine all LoRA weights
            combined_weights = {}
            for lora_id, lora_info in self.loaded_loras.items():
                weight = lora_info["weight"]
                for key, tensor in lora_info["weights"].items():
                    if key in combined_weights:
                        combined_weights[key] += tensor * weight
                    else:
                        combined_weights[key] = tensor * weight

            # Apply combined weights
            self._apply_lora_weights(pipeline, combined_weights)

    def list_loaded_loras(self) -> List[Dict]:
        """List currently loaded LoRAs"""
        return [
            {
                "lora_id": lora_id,
                "weight": info["weight"],
                "path": str(info["path"]),
                "metadata": info["metadata"],
            }
            for lora_id, info in self.loaded_loras.items()
        ]

    def list_available_loras(self) -> List[Dict]:
        """List all available LoRA models"""
        loras = []

        # Check lora models directory
        lora_dir = get_model_path("lora", "")
        if lora_dir.exists():
            for item in lora_dir.iterdir():
                if item.is_file() and item.suffix in [".safetensors", ".bin"]:
                    metadata = self._load_metadata(item)
                    loras.append(
                        {"lora_id": item.stem, "path": str(item), "metadata": metadata}
                    )
                elif item.is_dir() and (item / "adapter_config.json").exists():
                    metadata = self._load_metadata(item)
                    loras.append(
                        {"lora_id": item.name, "path": str(item), "metadata": metadata}
                    )

        # Check training runs directory
        runs_dir = self.cache_paths.runs
        if runs_dir.exists():
            for run_dir in runs_dir.iterdir():
                if run_dir.is_dir():
                    final_checkpoint = run_dir / "checkpoints" / "final"
                    if final_checkpoint.exists():
                        metadata = self._load_metadata(final_checkpoint)
                        loras.append(
                            {
                                "lora_id": run_dir.name,
                                "path": str(final_checkpoint),
                                "metadata": metadata,
                                "source": "training",
                            }
                        )

        return loras

    def _load_metadata(self, lora_path: Path) -> Dict:
        """Load LoRA metadata from companion file"""
        # Try different metadata file locations
        metadata_files = [
            lora_path.parent / f"{lora_path.stem}_metadata.json",
            lora_path.parent / "metadata.json",
            lora_path / "metadata.json" if lora_path.is_dir() else None,
            lora_path / "adapter_config.json" if lora_path.is_dir() else None,
        ]

        for metadata_file in metadata_files:
            if metadata_file and metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        return json.load(f)
                except Exception as e:
                    print(f"[LoRA] Failed to load metadata from {metadata_file}: {e}")

        # Return default metadata
        return {
            "name": lora_path.stem,
            "created_at": None,
            "base_model": "unknown",
            "rank": 16,
            "alpha": 16,
        }
