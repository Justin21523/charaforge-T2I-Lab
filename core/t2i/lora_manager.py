# core/t2i/lora_manager.py - Advanced LoRA loading, switching, and merging
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import json
import logging
from datetime import datetime
import torch
import torch.nn as F
from dataclasses import dataclass, asdict
import safetensors.torch as st
from copy import deepcopy

# Diffusers and PEFT
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTextModelWithProjection

from core.config import get_cache_paths
from core.train.registry import get_model_registry, ModelEntry


logger = logging.getLogger(__name__)


@dataclass
class LoRAInfo:
    """LoRA model information"""

    name: str
    path: str
    rank: int
    alpha: float
    target_modules: List[str]
    scale: float = 1.0
    enabled: bool = True
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LoRAStack:
    """Multiple LoRA models stacked together"""

    loras: List[LoRAInfo]
    total_scale: float = 1.0
    merged: bool = False

    def get_active_loras(self) -> List[LoRAInfo]:
        """Get only enabled LoRAs"""
        return [lora for lora in self.loras if lora.enabled]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loras": [lora.to_dict() for lora in self.loras],
            "total_scale": self.total_scale,
            "merged": self.merged,
        }


class LoRAManager:
    """Advanced LoRA management for Stable Diffusion models"""

    def __init__(self, device: str = "auto"):
        self.device = self._get_device(device)
        self.cache_paths = get_cache_paths()
        self.model_registry = get_model_registry()

        # Currently loaded models
        self.pipeline = None
        self.base_model_name = None
        self.is_sdxl = False

        # LoRA management
        self.loaded_loras: Dict[str, LoRAInfo] = {}
        self.lora_stack = LoRAStack(loras=[])
        self.original_unet_state = None
        self.original_text_encoder_state = None

        # Cache for LoRA weights
        self.lora_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self.cache_limit = 10  # Maximum number of LoRAs to keep in memory

        logger.info(f"LoRA Manager initialized on device: {self.device}")

    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def load_base_model(self, model_name: str, **kwargs) -> bool:
        """Load base Stable Diffusion model"""
        try:
            # Check if model is registered
            model_entry = self.model_registry.get_model(model_name)
            if not model_entry:
                logger.error(f"Model not found in registry: {model_name}")
                return False

            # Determine model type
            self.is_sdxl = (
                "xl" in model_name.lower() or model_entry.model_type == "sdxl"
            )

            # Load pipeline
            pipeline_class = (
                StableDiffusionXLPipeline if self.is_sdxl else StableDiffusionPipeline
            )

            default_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "cache_dir": self.cache_paths.hf_home,
                "safety_checker": None,
                "requires_safety_checker": False,
            }
            default_kwargs.update(kwargs)

            if model_entry.path.startswith("/"):
                # Local model
                self.pipeline = pipeline_class.from_pretrained(
                    model_entry.path, **default_kwargs
                )
            else:
                # HuggingFace model
                self.pipeline = pipeline_class.from_pretrained(
                    model_name, **default_kwargs
                )

            self.pipeline.to(self.device)
            self.base_model_name = model_name

            # Store original states for restoration
            self._backup_original_states()

            # Update model usage
            self.model_registry.update_model_usage(model_name)

            logger.info(
                f"Base model loaded: {model_name} ({'SDXL' if self.is_sdxl else 'SD1.5'})"
            )
            return True

        except Exception as e:
            logger.error(f"Error loading base model {model_name}: {e}")
            return False

    def _backup_original_states(self):
        """Backup original model states for restoration"""
        if self.pipeline:
            self.original_unet_state = deepcopy(self.pipeline.unet.state_dict())
            self.original_text_encoder_state = deepcopy(
                self.pipeline.text_encoder.state_dict()
            )

            # For SDXL, also backup text_encoder_2
            if self.is_sdxl and hasattr(self.pipeline, "text_encoder_2"):
                self.original_text_encoder_2_state = deepcopy(
                    self.pipeline.text_encoder_2.state_dict()
                )

    def load_lora(
        self, lora_name: str, scale: float = 1.0, force_reload: bool = False
    ) -> bool:
        """Load a single LoRA model"""
        try:
            # Check if already loaded
            if lora_name in self.loaded_loras and not force_reload:
                logger.info(f"LoRA already loaded: {lora_name}")
                self.loaded_loras[lora_name].scale = scale
                self.loaded_loras[lora_name].enabled = True
                return True

            # Get LoRA info from registry
            lora_entry = self.model_registry.get_model(lora_name)
            if not lora_entry or lora_entry.model_type != "lora":
                logger.error(f"LoRA not found in registry: {lora_name}")
                return False

            lora_path = Path(lora_entry.path)
            if not lora_path.exists():
                logger.error(f"LoRA file not found: {lora_path}")
                return False

            # Load LoRA weights
            lora_weights = self._load_lora_weights(lora_path)
            if not lora_weights:
                return False

            # Extract LoRA configuration
            lora_config = self._extract_lora_config(lora_weights)

            # Create LoRA info
            lora_info = LoRAInfo(
                name=lora_name,
                path=str(lora_path),
                rank=lora_config.get("rank", 16),
                alpha=lora_config.get("alpha", 32),
                target_modules=lora_config.get("target_modules", []),
                scale=scale,
                enabled=True,
                metadata=lora_entry.metadata,
            )

            # Store in cache and loaded LoRAs
            self.lora_cache[lora_name] = lora_weights
            self.loaded_loras[lora_name] = lora_info

            # Manage cache size
            self._manage_cache()

            # Update model usage
            self.model_registry.update_model_usage(lora_name)

            logger.info(f"LoRA loaded: {lora_name} (scale: {scale})")
            return True

        except Exception as e:
            logger.error(f"Error loading LoRA {lora_name}: {e}")
            return False

    def _load_lora_weights(self, lora_path: Path) -> Optional[Dict[str, torch.Tensor]]:
        """Load LoRA weights from file"""
        try:
            if lora_path.suffix == ".safetensors":
                return st.load_file(str(lora_path), device="cpu")
            elif lora_path.suffix in [".pt", ".pth"]:
                return torch.load(str(lora_path), map_location="cpu")
            else:
                logger.error(f"Unsupported LoRA file format: {lora_path.suffix}")
                return None
        except Exception as e:
            logger.error(f"Error loading LoRA weights from {lora_path}: {e}")
            return None

    def _extract_lora_config(
        self, lora_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Extract LoRA configuration from weights"""
        config = {"rank": 16, "alpha": 32, "target_modules": []}

        # Try to infer rank from weight shapes
        lora_a_keys = [k for k in lora_weights.keys() if "lora_A" in k]
        if lora_a_keys:
            # Get rank from first LoRA layer
            first_key = lora_a_keys[0]
            config["rank"] = lora_weights[first_key].shape[0]

        # Try to infer alpha (often stored as metadata or can be guessed)
        # Default to rank * 2 if not found
        config["alpha"] = config["rank"] * 2

        # Extract target modules from weight keys
        target_modules = set()
        for key in lora_weights.keys():
            if "lora_" in key:
                # Extract module name (everything before .lora_A or .lora_B)
                parts = key.split(".lora_")
                if len(parts) > 1:
                    module_path = parts[0]
                    # Get the last part as the target module
                    module_name = module_path.split(".")[-1]
                    target_modules.add(module_name)

        config["target_modules"] = list(target_modules)

        return config

    def _manage_cache(self):
        """Manage LoRA cache size"""
        if len(self.lora_cache) > self.cache_limit:
            # Remove least recently used LoRAs
            # For now, just remove the first one (TODO: implement proper LRU)
            oldest_lora = next(iter(self.lora_cache))
            del self.lora_cache[oldest_lora]
            logger.debug(f"Removed {oldest_lora} from cache due to size limit")

    def unload_lora(self, lora_name: str) -> bool:
        """Unload a specific LoRA"""
        if lora_name in self.loaded_loras:
            del self.loaded_loras[lora_name]
            if lora_name in self.lora_cache:
                del self.lora_cache[lora_name]
            logger.info(f"LoRA unloaded: {lora_name}")
            return True
        return False

    def set_lora_scale(self, lora_name: str, scale: float) -> bool:
        """Set LoRA scale"""
        if lora_name in self.loaded_loras:
            self.loaded_loras[lora_name].scale = scale
            logger.info(f"LoRA scale updated: {lora_name} -> {scale}")
            return True
        return False

    def enable_lora(self, lora_name: str, enabled: bool = True) -> bool:
        """Enable or disable a LoRA"""
        if lora_name in self.loaded_loras:
            self.loaded_loras[lora_name].enabled = enabled
            logger.info(f"LoRA {'enabled' if enabled else 'disabled'}: {lora_name}")
            return True
        return False

    def create_lora_stack(self, lora_configs: List[Dict[str, Any]]) -> LoRAStack:
        """Create a stack of multiple LoRAs"""
        stack_loras = []

        for config in lora_configs:
            lora_name = config["name"]
            scale = config.get("scale", 1.0)
            enabled = config.get("enabled", True)

            # Load LoRA if not already loaded
            if lora_name not in self.loaded_loras:
                if not self.load_lora(lora_name, scale):
                    continue

            # Update configuration
            lora_info = self.loaded_loras[lora_name]
            lora_info.scale = scale
            lora_info.enabled = enabled

            stack_loras.append(lora_info)

        return LoRAStack(loras=stack_loras)

    def apply_lora_stack(self, lora_stack: Optional[LoRAStack] = None) -> bool:
        """Apply LoRA stack to the current pipeline"""
        if not self.pipeline:
            logger.error("No base model loaded")
            return False

        if lora_stack is None:
            lora_stack = self.lora_stack

        try:
            # Restore original states
            self._restore_original_states()

            # Apply each LoRA in the stack
            active_loras = lora_stack.get_active_loras()

            if not active_loras:
                logger.info("No active LoRAs to apply")
                return True

            logger.info(f"Applying {len(active_loras)} LoRAs to pipeline")

            # Merge all LoRA weights
            merged_weights = self._merge_lora_weights(active_loras)

            # Apply merged weights to pipeline
            self._apply_weights_to_pipeline(merged_weights)

            # Update current stack
            self.lora_stack = lora_stack
            self.lora_stack.merged = True

            logger.info("LoRA stack applied successfully")
            return True

        except Exception as e:
            logger.error(f"Error applying LoRA stack: {e}")
            return False

    def _restore_original_states(self):
        """Restore original model states"""
        if self.pipeline and self.original_unet_state:
            self.pipeline.unet.load_state_dict(self.original_unet_state)

        if self.pipeline and self.original_text_encoder_state:
            self.pipeline.text_encoder.load_state_dict(self.original_text_encoder_state)

        if (
            self.is_sdxl
            and hasattr(self, "original_text_encoder_2_state")
            and hasattr(self.pipeline, "text_encoder_2")
        ):
            self.pipeline.text_encoder_2.load_state_dict(  # type: ignore
                self.original_text_encoder_2_state
            )

    def _merge_lora_weights(self, loras: List[LoRAInfo]) -> Dict[str, torch.Tensor]:
        """Merge multiple LoRA weights"""
        merged_weights = {}

        for lora in loras:
            if lora.name not in self.lora_cache:
                logger.warning(f"LoRA weights not in cache: {lora.name}")
                continue

            lora_weights = self.lora_cache[lora.name]
            scale = lora.scale

            # Apply LoRA weights with scaling
            for key, weight in lora_weights.items():
                if key not in merged_weights:
                    merged_weights[key] = weight * scale
                else:
                    merged_weights[key] += weight * scale

        return merged_weights

    def _apply_weights_to_pipeline(self, weights: Dict[str, torch.Tensor]):
        """Apply LoRA weights to pipeline components"""
        unet_weights = {}
        text_encoder_weights = {}
        text_encoder_2_weights = {}

        # Separate weights by component
        for key, weight in weights.items():
            if key.startswith("lora_unet_"):
                # Remove prefix and convert to UNet parameter name
                unet_key = key.replace("lora_unet_", "").replace("lora_", "")
                unet_weights[unet_key] = weight
            elif key.startswith("lora_te_") or "text_encoder" in key:
                # Text encoder weights
                te_key = key.replace("lora_te_", "").replace("lora_", "")
                if "text_encoder_2" in key:
                    text_encoder_2_weights[te_key] = weight
                else:
                    text_encoder_weights[te_key] = weight

        # Apply to UNet
        if unet_weights:
            self._apply_lora_to_component(self.pipeline.unet, unet_weights)  # type: ignore

        # Apply to text encoders
        if text_encoder_weights:
            self._apply_lora_to_component(
                self.pipeline.text_encoder, text_encoder_weights  # type: ignore
            )

        if text_encoder_2_weights and hasattr(self.pipeline, "text_encoder_2"):
            self._apply_lora_to_component(
                self.pipeline.text_encoder_2, text_encoder_2_weights  # type: ignore
            )

    def _apply_lora_to_component(
        self, component: torch.nn.Module, lora_weights: Dict[str, torch.Tensor]
    ):
        """Apply LoRA weights to a specific component"""
        component_state = component.state_dict()

        for key, lora_weight in lora_weights.items():
            if key in component_state:
                # Add LoRA weight to original weight
                original_weight = component_state[key]
                if original_weight.shape == lora_weight.shape:
                    component_state[key] = original_weight + lora_weight.to(
                        original_weight.device, original_weight.dtype
                    )
                else:
                    logger.warning(
                        f"Shape mismatch for {key}: {original_weight.shape} vs {lora_weight.shape}"
                    )

        component.load_state_dict(component_state)

    def save_merged_model(
        self, output_path: Union[str, Path], include_text_encoder: bool = True
    ) -> bool:
        """Save current state as a merged model"""
        if not self.pipeline:
            logger.error("No pipeline loaded")
            return False

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Save the current pipeline state
            self.pipeline.save_pretrained(output_path)

            # Save metadata about merged LoRAs
            metadata = {
                "base_model": self.base_model_name,
                "merged_loras": self.lora_stack.to_dict(),
                "merge_timestamp": datetime.now().isoformat(),
                "is_sdxl": self.is_sdxl,
            }

            with open(output_path / "merge_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Merged model saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving merged model: {e}")
            return False

    def extract_lora_from_merged(
        self,
        merged_model_path: Union[str, Path],
        original_model_path: Union[str, Path],
        output_path: Union[str, Path],
        rank: int = 16,
        alpha: float = 32,
    ) -> bool:
        """Extract LoRA weights from a merged model by comparing with original"""
        try:
            merged_model_path = Path(merged_model_path)
            original_model_path = Path(original_model_path)
            output_path = Path(output_path)

            # Load both models
            pipeline_class = (
                StableDiffusionXLPipeline if self.is_sdxl else StableDiffusionPipeline
            )

            original_pipeline = pipeline_class.from_pretrained(
                str(original_model_path), torch_dtype=torch.float32, device_map="cpu"
            )

            merged_pipeline = pipeline_class.from_pretrained(
                str(merged_model_path), torch_dtype=torch.float32, device_map="cpu"
            )

            # Calculate differences
            lora_weights = {}

            # Process UNet
            orig_unet_state = original_pipeline.unet.state_dict()
            merged_unet_state = merged_pipeline.unet.state_dict()

            for key in orig_unet_state.keys():
                if key in merged_unet_state:
                    diff = merged_unet_state[key] - orig_unet_state[key]

                    # Skip if difference is negligible
                    if torch.abs(diff).max() < 1e-6:
                        continue

                    # Decompose into LoRA format using SVD
                    if len(diff.shape) >= 2:
                        lora_a, lora_b = self._decompose_weight_to_lora(diff, rank)
                        if lora_a is not None and lora_b is not None:
                            lora_weights[f"lora_unet_{key}.lora_A.weight"] = lora_a
                            lora_weights[f"lora_unet_{key}.lora_B.weight"] = lora_b

            # Save extracted LoRA
            if lora_weights:
                st.save_file(lora_weights, str(output_path))

                # Save metadata
                metadata = {
                    "original_model": str(original_model_path),
                    "merged_model": str(merged_model_path),
                    "rank": rank,
                    "alpha": alpha,
                    "extraction_timestamp": datetime.now().isoformat(),
                }

                metadata_path = output_path.with_suffix(".json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                logger.info(f"LoRA extracted and saved to: {output_path}")
                return True
            else:
                logger.warning("No significant differences found between models")
                return False

        except Exception as e:
            logger.error(f"Error extracting LoRA: {e}")
            return False

    def _decompose_weight_to_lora(
        self, weight: torch.Tensor, rank: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Decompose weight difference into LoRA A and B matrices using SVD"""
        try:
            if len(weight.shape) == 4:  # Conv layer
                weight_2d = weight.flatten(1)
            elif len(weight.shape) == 2:  # Linear layer
                weight_2d = weight
            else:
                return None, None

            # Perform SVD
            U, S, V = torch.svd(weight_2d)

            # Truncate to rank
            rank = min(rank, S.shape[0])
            U_r = U[:, :rank]
            S_r = S[:rank]
            V_r = V[:, :rank]

            # Create LoRA matrices
            lora_A = V_r.T  # Shape: (rank, in_features)
            lora_B = U_r * S_r.unsqueeze(0)  # Shape: (out_features, rank)

            # Reshape back for conv layers
            if len(weight.shape) == 4:
                lora_A = lora_A.view(
                    rank, weight.shape[1], weight.shape[2], weight.shape[3]
                )
                lora_B = lora_B.view(weight.shape[0], rank, 1, 1)

            return lora_A, lora_B

        except Exception as e:
            logger.error(f"Error in SVD decomposition: {e}")
            return None, None

    def list_loaded_loras(self) -> List[Dict[str, Any]]:
        """List all currently loaded LoRAs"""
        return [lora.to_dict() for lora in self.loaded_loras.values()]

    def get_lora_info(self, lora_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific LoRA"""
        if lora_name in self.loaded_loras:
            lora_info = self.loaded_loras[lora_name].to_dict()

            # Add additional runtime info
            lora_info["cached"] = lora_name in self.lora_cache
            lora_info["in_current_stack"] = any(
                lora.name == lora_name for lora in self.lora_stack.loras
            )

            return lora_info
        return None

    def clear_all_loras(self):
        """Clear all loaded LoRAs and restore original model"""
        self.loaded_loras.clear()
        self.lora_cache.clear()
        self.lora_stack = LoRAStack(loras=[])

        if self.pipeline:
            self._restore_original_states()

        logger.info("All LoRAs cleared and model restored to original state")

    def optimize_memory(self):
        """Optimize memory usage by clearing unused LoRAs from cache"""
        # Get list of LoRAs in current stack
        active_lora_names = {lora.name for lora in self.lora_stack.loras}

        # Remove inactive LoRAs from cache
        to_remove = []
        for lora_name in self.lora_cache.keys():
            if lora_name not in active_lora_names:
                to_remove.append(lora_name)

        for lora_name in to_remove:
            del self.lora_cache[lora_name]
            logger.debug(f"Removed {lora_name} from cache (not in active stack)")

        # Force garbage collection
        import gc

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Memory optimized. {len(to_remove)} LoRAs removed from cache.")

    def benchmark_lora_performance(
        self, test_prompt: str = "a beautiful landscape", num_runs: int = 3
    ) -> Dict[str, Any]:
        """Benchmark performance with and without LoRAs"""
        if not self.pipeline:
            logger.error("No pipeline loaded")
            return {}

        import time

        results = {
            "test_prompt": test_prompt,
            "num_runs": num_runs,
            "baseline_times": [],
            "lora_times": [],
            "current_stack": self.lora_stack.to_dict(),
        }

        # Benchmark baseline (no LoRAs)
        self._restore_original_states()

        for i in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                _ = self.pipeline(
                    test_prompt,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    height=512,
                    width=512,
                ).images[  # type: ignore
                    0
                ]
            end_time = time.time()
            results["baseline_times"].append(end_time - start_time)

        # Benchmark with current LoRA stack
        if self.lora_stack.loras:
            self.apply_lora_stack()

            for i in range(num_runs):
                start_time = time.time()
                with torch.no_grad():
                    _ = self.pipeline(
                        test_prompt,
                        num_inference_steps=20,
                        guidance_scale=7.5,
                        height=512,
                        width=512,
                    ).images[  # type: ignore
                        0
                    ]
                end_time = time.time()
                results["lora_times"].append(end_time - start_time)

        # Calculate statistics
        if results["baseline_times"]:
            results["baseline_avg"] = sum(results["baseline_times"]) / len(
                results["baseline_times"]
            )

        if results["lora_times"]:
            results["lora_avg"] = sum(results["lora_times"]) / len(
                results["lora_times"]
            )
            results["overhead_percent"] = (
                (results["lora_avg"] - results["baseline_avg"])
                / results["baseline_avg"]
            ) * 100

        logger.info(f"Performance benchmark completed:")
        logger.info(f"  Baseline: {results.get('baseline_avg', 0):.2f}s")
        logger.info(f"  With LoRAs: {results.get('lora_avg', 0):.2f}s")
        logger.info(f"  Overhead: {results.get('overhead_percent', 0):.1f}%")

        return results

    def export_lora_preset(
        self, preset_name: str, output_path: Union[str, Path]
    ) -> bool:
        """Export current LoRA configuration as a preset"""
        try:
            preset_config = {
                "name": preset_name,
                "base_model": self.base_model_name,
                "lora_stack": self.lora_stack.to_dict(),
                "created_at": datetime.now().isoformat(),
                "model_type": "sdxl" if self.is_sdxl else "sd15",
            }

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(preset_config, f, indent=2, ensure_ascii=False)

            logger.info(f"LoRA preset exported: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting preset: {e}")
            return False

    def load_lora_preset(self, preset_path: Union[str, Path]) -> bool:
        """Load LoRA configuration from a preset file"""
        try:
            preset_path = Path(preset_path)

            if not preset_path.exists():
                logger.error(f"Preset file not found: {preset_path}")
                return False

            with open(preset_path, "r", encoding="utf-8") as f:
                preset_config = json.load(f)

            # Validate preset
            if "lora_stack" not in preset_config:
                logger.error("Invalid preset format: missing lora_stack")
                return False

            # Load the base model if different
            preset_base_model = preset_config.get("base_model")
            if preset_base_model and preset_base_model != self.base_model_name:
                logger.info(f"Loading different base model: {preset_base_model}")
                if not self.load_base_model(preset_base_model):
                    return False

            # Create LoRA stack from preset
            stack_config = preset_config["lora_stack"]
            lora_configs = [lora for lora in stack_config.get("loras", [])]

            # Load LoRAs
            loaded_loras = []
            for lora_config in lora_configs:
                lora_name = lora_config["name"]
                scale = lora_config.get("scale", 1.0)
                enabled = lora_config.get("enabled", True)

                if self.load_lora(lora_name, scale):
                    self.enable_lora(lora_name, enabled)
                    loaded_loras.append(self.loaded_loras[lora_name])
                else:
                    logger.warning(f"Failed to load LoRA from preset: {lora_name}")

            # Create and apply new stack
            new_stack = LoRAStack(
                loras=loaded_loras, total_scale=stack_config.get("total_scale", 1.0)
            )

            self.apply_lora_stack(new_stack)

            logger.info(
                f"LoRA preset loaded successfully: {preset_config.get('name', 'unnamed')}"
            )
            logger.info(f"Loaded {len(loaded_loras)} LoRAs")

            return True

        except Exception as e:
            logger.error(f"Error loading preset: {e}")
            return False


# Global LoRA manager instance
_lora_manager_instance = None


def get_lora_manager() -> LoRAManager:
    """Get global LoRA manager instance"""
    global _lora_manager_instance
    if _lora_manager_instance is None:
        _lora_manager_instance = LoRAManager()
    return _lora_manager_instance
