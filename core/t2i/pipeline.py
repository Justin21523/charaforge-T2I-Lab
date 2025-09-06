# core/t2i/pipeline.py - Unified Text-to-Image Pipeline Manager
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from pathlib import Path
import logging
import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass, asdict
import json
import time
from datetime import datetime

# Diffusers

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)
from diffusers.schedulers.scheduling_euler_ancestral_discrete import (
    EulerAncestralDiscreteScheduler,
)
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler

# Core components
from core.config import get_cache_paths, get_model_config
from core.t2i.lora_manager import get_lora_manager, LoRAStack
from core.t2i.safety import SafetyChecker
from core.t2i.watermark import WatermarkProcessor
from core.train.registry import get_model_registry


logger = logging.getLogger(__name__)


@dataclass
class GenerationParams:
    """Text-to-Image generation parameters"""

    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 512
    height: int = 512
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    num_images_per_prompt: int = 1
    eta: float = 0.0
    generator: Optional[torch.Generator] = None
    seed: Optional[int] = None
    output_type: str = "pil"
    return_dict: bool = True
    cross_attention_kwargs: Optional[Dict[str, Any]] = None
    clip_skip: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Remove non-serializable fields
        data.pop("generator", None)
        return data


@dataclass
class GenerationResult:
    """Generation result with metadata"""

    images: List[Image.Image]
    params: GenerationParams
    metadata: Dict[str, Any]
    generation_time: float
    safety_check_passed: bool = True
    watermarked: bool = False


class T2IPipelineManager:
    """Unified Text-to-Image Pipeline Manager"""

    def __init__(
        self,
        device: str = "auto",
        enable_safety: bool = True,
        enable_watermark: bool = False,
    ):
        self.device = self._get_device(device)
        self.cache_paths = get_cache_paths()
        self.model_registry = get_model_registry()
        self.lora_manager = get_lora_manager()

        # Pipeline state
        self.current_pipeline = None
        self.current_model = None
        self.is_sdxl = False
        self.pipeline_loaded = False

        # Components
        self.safety_checker = SafetyChecker() if enable_safety else None
        self.watermark_processor = WatermarkProcessor() if enable_watermark else None

        # Available schedulers
        self.available_schedulers = {
            "DPMSolverMultistep": DPMSolverMultistepScheduler,
            "EulerAncestral": EulerAncestralDiscreteScheduler,
            "DDIM": DDIMScheduler,
            "PNDM": PNDMScheduler,
            "LMS": LMSDiscreteScheduler,
        }

        # Generation cache
        self.generation_cache = {}
        self.cache_enabled = True
        self.max_cache_size = 50

        # Performance settings
        self.low_vram_mode = False
        self.attention_slicing = True
        self.memory_efficient_attention = True

        logger.info(f"T2I Pipeline Manager initialized on {self.device}")

    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def load_model(self, model_name: str, **kwargs) -> bool:
        """Load a Stable Diffusion model"""
        try:
            # Check if model is already loaded
            if self.current_model == model_name and self.pipeline_loaded:
                logger.info(f"Model already loaded: {model_name}")
                return True

            # Get model info from registry
            model_entry = self.model_registry.get_model(model_name)
            if not model_entry:
                logger.error(f"Model not found in registry: {model_name}")
                return False

            # Determine model type
            self.is_sdxl = (
                "xl" in model_name.lower() or model_entry.model_type == "sdxl"
            )

            # Configure pipeline settings
            pipeline_kwargs = self._get_pipeline_kwargs(**kwargs)

            # Load pipeline
            pipeline_class = (
                StableDiffusionXLPipeline if self.is_sdxl else StableDiffusionPipeline
            )

            if model_entry.path.startswith("/"):
                # Local model
                self.current_pipeline = pipeline_class.from_pretrained(
                    model_entry.path, **pipeline_kwargs
                )
            else:
                # HuggingFace model
                self.current_pipeline = pipeline_class.from_pretrained(
                    model_name, **pipeline_kwargs
                )

            # Move to device
            self.current_pipeline.to(self.device)

            # Configure pipeline for performance
            self._configure_pipeline_performance()

            # Load model in LoRA manager
            self.lora_manager.load_base_model(model_name)

            # Update state
            self.current_model = model_name
            self.pipeline_loaded = True

            # Update model usage
            self.model_registry.update_model_usage(model_name)

            logger.info(
                f"Model loaded successfully: {model_name} ({'SDXL' if self.is_sdxl else 'SD1.5'})"
            )
            return True

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            self.pipeline_loaded = False
            return False

    def _get_pipeline_kwargs(self, **kwargs) -> Dict[str, Any]:
        """Get pipeline loading configuration"""
        default_kwargs = {
            "torch_dtype": (
                torch.float16 if self.device.type == "cuda" else torch.float32
            ),
            "cache_dir": self.cache_paths.hf_home,
            "safety_checker": None,  # We use our own safety checker
            "requires_safety_checker": False,
        }

        # Low VRAM optimizations
        if self.low_vram_mode:
            default_kwargs.update(
                {
                    "device_map": "auto",
                    "low_cpu_mem_usage": True,
                    "load_in_8bit": kwargs.get("load_in_8bit", False),
                    "load_in_4bit": kwargs.get("load_in_4bit", False),
                }
            )

        # Update with user kwargs
        default_kwargs.update(kwargs)
        return default_kwargs

    def _configure_pipeline_performance(self):
        """Configure pipeline for optimal performance"""
        if not self.current_pipeline:
            return

        # Enable attention slicing for memory efficiency
        if self.attention_slicing:
            try:
                self.current_pipeline.enable_attention_slicing()
                logger.debug("Attention slicing enabled")
            except Exception as e:
                logger.warning(f"Could not enable attention slicing: {e}")

        # Enable memory efficient attention
        if self.memory_efficient_attention:
            try:
                self.current_pipeline.enable_memory_efficient_attention()
                logger.debug("Memory efficient attention enabled")
            except Exception as e:
                logger.warning(f"Could not enable memory efficient attention: {e}")

        # Enable VAE slicing for lower VRAM usage
        if self.low_vram_mode:
            try:
                self.current_pipeline.enable_vae_slicing()
                logger.debug("VAE slicing enabled")
            except Exception as e:
                logger.warning(f"Could not enable VAE slicing: {e}")

    def set_scheduler(self, scheduler_name: str, **scheduler_kwargs) -> bool:
        """Change the noise scheduler"""
        if not self.pipeline_loaded:
            logger.error("No model loaded")
            return False

        if scheduler_name not in self.available_schedulers:
            logger.error(f"Unknown scheduler: {scheduler_name}")
            logger.info(
                f"Available schedulers: {list(self.available_schedulers.keys())}"
            )
            return False

        try:
            scheduler_class = self.available_schedulers[scheduler_name]

            # Get scheduler config from current scheduler
            scheduler_config = self.current_pipeline.scheduler.config  # type: ignore

            # Create new scheduler with same config but different class
            new_scheduler = scheduler_class.from_config(
                scheduler_config, **scheduler_kwargs
            )

            # Replace scheduler
            self.current_pipeline.scheduler = new_scheduler  # type: ignore

            logger.info(f"Scheduler changed to: {scheduler_name}")
            return True

        except Exception as e:
            logger.error(f"Error changing scheduler: {e}")
            return False

    def generate(
        self, params: Union[GenerationParams, Dict[str, Any]], **kwargs
    ) -> GenerationResult:
        """Generate images with the current pipeline"""
        if not self.pipeline_loaded:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # Convert dict to GenerationParams if needed
        if isinstance(params, dict):
            params = GenerationParams(**params)

        # Override params with kwargs
        for key, value in kwargs.items():
            if hasattr(params, key):
                setattr(params, key, value)

        # Setup generator with seed
        generator = None
        if params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(params.seed)
        elif params.generator is not None:
            generator = params.generator

        # Check cache
        cache_key = None
        if self.cache_enabled:
            cache_key = self._get_cache_key(params)
            if cache_key in self.generation_cache:
                logger.debug("Returning cached result")
                return self.generation_cache[cache_key]

        # Generate images
        start_time = time.time()

        try:
            # Prepare generation kwargs
            gen_kwargs = {
                "prompt": params.prompt,
                "negative_prompt": params.negative_prompt,
                "width": params.width,
                "height": params.height,
                "num_inference_steps": params.num_inference_steps,
                "guidance_scale": params.guidance_scale,
                "num_images_per_prompt": params.num_images_per_prompt,
                "eta": params.eta,
                "generator": generator,
                "output_type": params.output_type,
                "return_dict": params.return_dict,
                "cross_attention_kwargs": params.cross_attention_kwargs,
            }

            # Add CLIP skip for SDXL
            if self.is_sdxl and params.clip_skip:
                gen_kwargs["clip_skip"] = params.clip_skip

            # Remove None values
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

            # Generate
            with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
                result = self.current_pipeline(**gen_kwargs)  # type: ignore

            images = result.images if hasattr(result, "images") else [result]  # type: ignore

            generation_time = time.time() - start_time

            # Safety check
            safety_passed = True
            if self.safety_checker:
                images, safety_passed = self.safety_checker.check_images(
                    images, [params.prompt]  # type: ignore
                )

            # Watermarking
            watermarked = False
            if self.watermark_processor and safety_passed:
                images = self.watermark_processor.add_watermark(images)  # type: ignore
                watermarked = True

            # Create result
            generation_result = GenerationResult(
                images=images,  # type: ignore
                params=params,
                metadata={
                    "model": self.current_model,
                    "is_sdxl": self.is_sdxl,
                    "device": str(self.device),
                    "scheduler": self.current_pipeline.scheduler.__class__.__name__,  # type: ignore
                    "lora_stack": (
                        self.lora_manager.lora_stack.to_dict()
                        if self.lora_manager.lora_stack.loras
                        else None
                    ),
                    "generation_id": self._generate_id(),
                    "timestamp": datetime.now().isoformat(),
                },
                generation_time=generation_time,
                safety_check_passed=safety_passed,
                watermarked=watermarked,
            )

            # Cache result
            if self.cache_enabled and cache_key and safety_passed:
                self._add_to_cache(cache_key, generation_result)

            logger.info(f"Generated {len(images)} image(s) in {generation_time:.2f}s")
            return generation_result

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise

    def generate_batch(
        self,
        prompts: List[str],
        shared_params: Dict[str, Any] = None,  # type: ignore
        individual_params: List[Dict[str, Any]] = None,  # type: ignore
    ) -> List[GenerationResult]:
        """Generate multiple images with different prompts"""
        if not prompts:
            return []

        shared_params = shared_params or {}
        individual_params = individual_params or [{}] * len(prompts)

        results = []
        for i, prompt in enumerate(prompts):
            # Merge parameters
            params = shared_params.copy()
            params.update(individual_params[i] if i < len(individual_params) else {})
            params["prompt"] = prompt

            try:
                result = self.generate(params)
                results.append(result)
            except Exception as e:
                logger.error(f"Error generating image for prompt '{prompt}': {e}")
                # Add empty result to maintain list alignment
                results.append(None)

        return results

    def load_lora_stack(self, lora_configs: List[Dict[str, Any]]) -> bool:
        """Load and apply a stack of LoRA models"""
        try:
            # Create LoRA stack
            lora_stack = self.lora_manager.create_lora_stack(lora_configs)

            # Apply to current pipeline
            success = self.lora_manager.apply_lora_stack(lora_stack)

            if success:
                logger.info(
                    f"LoRA stack applied: {len(lora_stack.get_active_loras())} active LoRAs"
                )

            return success

        except Exception as e:
            logger.error(f"Error loading LoRA stack: {e}")
            return False

    def unload_loras(self):
        """Unload all LoRAs and restore base model"""
        self.lora_manager.clear_all_loras()
        logger.info("All LoRAs unloaded")

    def set_lora_scale(self, lora_name: str, scale: float) -> bool:
        """Adjust LoRA scale and reapply"""
        if self.lora_manager.set_lora_scale(lora_name, scale):
            # Reapply current stack with new scale
            return self.lora_manager.apply_lora_stack()
        return False

    def enable_low_vram_mode(self, enabled: bool = True):
        """Enable/disable low VRAM optimizations"""
        self.low_vram_mode = enabled

        if self.pipeline_loaded:
            self._configure_pipeline_performance()
            logger.info(f"Low VRAM mode {'enabled' if enabled else 'disabled'}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model"""
        if not self.pipeline_loaded:
            return {}

        return {
            "model_name": self.current_model,
            "model_type": "SDXL" if self.is_sdxl else "SD1.5",
            "device": str(self.device),
            "scheduler": self.current_pipeline.scheduler.__class__.__name__,  # type: ignore
            "low_vram_mode": self.low_vram_mode,
            "attention_slicing": self.attention_slicing,
            "memory_efficient_attention": self.memory_efficient_attention,
            "safety_enabled": self.safety_checker is not None,
            "watermark_enabled": self.watermark_processor is not None,
            "loaded_loras": self.lora_manager.list_loaded_loras(),
            "current_lora_stack": self.lora_manager.lora_stack.to_dict(),
        }

    def _get_cache_key(self, params: GenerationParams) -> str:
        """Generate cache key for generation parameters"""
        import hashlib

        # Create deterministic key from parameters
        key_data = {
            "model": self.current_model,
            "prompt": params.prompt,
            "negative_prompt": params.negative_prompt,
            "width": params.width,
            "height": params.height,
            "steps": params.num_inference_steps,
            "guidance": params.guidance_scale,
            "seed": params.seed,
            "scheduler": self.current_pipeline.scheduler.__class__.__name__,  # type: ignore
            "lora_stack": (
                self.lora_manager.lora_stack.to_dict()
                if self.lora_manager.lora_stack.loras
                else None
            ),
        }

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _add_to_cache(self, key: str, result: GenerationResult):
        """Add result to cache with size management"""
        if len(self.generation_cache) >= self.max_cache_size:
            # Remove oldest entry (FIFO)
            oldest_key = next(iter(self.generation_cache))
            del self.generation_cache[oldest_key]

        self.generation_cache[key] = result

    def clear_cache(self):
        """Clear generation cache"""
        self.generation_cache.clear()
        logger.info("Generation cache cleared")

    def _generate_id(self) -> str:
        """Generate unique ID for generation"""
        import uuid

        return str(uuid.uuid4())[:8]

    def save_generation_result(
        self, result: GenerationResult, output_dir: Union[str, Path]
    ) -> List[Path]:
        """Save generation result to directory"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        # Save images
        for i, image in enumerate(result.images):
            filename = f"generated_{result.metadata['generation_id']}_{i:02d}.png"
            image_path = output_dir / filename
            image.save(image_path)
            saved_paths.append(image_path)

        # Save metadata
        metadata_path = output_dir / f"metadata_{result.metadata['generation_id']}.json"
        metadata = {
            "params": result.params.to_dict(),
            "metadata": result.metadata,
            "generation_time": result.generation_time,
            "safety_check_passed": result.safety_check_passed,
            "watermarked": result.watermarked,
            "num_images": len(result.images),
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Generation result saved: {len(saved_paths)} images + metadata")
        return saved_paths

    def create_image_grid(
        self,
        results: List[GenerationResult],
        grid_size: Optional[Tuple[int, int]] = None,
    ) -> Image.Image:
        """Create a grid from multiple generation results"""
        # Collect all images
        all_images = []
        for result in results:
            if result and result.images:
                all_images.extend(result.images)

        if not all_images:
            raise ValueError("No images to create grid from")

        # Determine grid size
        if grid_size is None:
            num_images = len(all_images)
            cols = int(np.ceil(np.sqrt(num_images)))
            rows = int(np.ceil(num_images / cols))
            grid_size = (rows, cols)

        rows, cols = grid_size

        # Get image size (assume all images are same size)
        img_width, img_height = all_images[0].size

        # Create grid
        grid_width = cols * img_width
        grid_height = rows * img_height
        grid_image = Image.new("RGB", (grid_width, grid_height), color="white")  # type: ignore

        # Place images
        for i, image in enumerate(all_images[: rows * cols]):
            row = i // cols
            col = i % cols
            x = col * img_width
            y = row * img_height
            grid_image.paste(image, (x, y))

        return grid_image

    def benchmark_performance(
        self, test_params: GenerationParams = None, num_runs: int = 3  # type: ignore
    ) -> Dict[str, Any]:
        """Benchmark generation performance"""
        if not self.pipeline_loaded:
            raise RuntimeError("No model loaded")

        if test_params is None:
            test_params = GenerationParams(
                prompt="a beautiful landscape, highly detailed",
                width=512,
                height=512,
                num_inference_steps=20,
                seed=42,
            )

        logger.info(f"Running performance benchmark ({num_runs} runs)")

        times = []
        for i in range(num_runs):
            logger.debug(f"Benchmark run {i+1}/{num_runs}")

            start_time = time.time()
            result = self.generate(test_params)
            end_time = time.time()

            times.append(end_time - start_time)

        # Calculate statistics
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)

        # Get memory usage if on CUDA
        memory_stats = {}
        if torch.cuda.is_available():
            memory_stats = {
                "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "cached_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
            }

        benchmark_result = {
            "test_params": test_params.to_dict(),
            "num_runs": num_runs,
            "times": times,
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "std_time": std_time,
            "throughput_imgs_per_sec": test_params.num_images_per_prompt / avg_time,
            "model_info": self.get_model_info(),
            "memory_stats": memory_stats,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Benchmark completed:")
        logger.info(f"  Average time: {avg_time:.2f}s ± {std_time:.2f}s")
        logger.info(
            f"  Throughput: {benchmark_result['throughput_imgs_per_sec']:.2f} img/s"
        )

        return benchmark_result

    def optimize_memory(self):
        """Optimize memory usage"""
        # Clear generation cache
        self.clear_cache()

        # Optimize LoRA manager
        self.lora_manager.optimize_memory()

        # Force garbage collection
        import gc

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("Memory optimization completed")

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        base_models = self.model_registry.list_models(
            model_type="sd15"
        ) + self.model_registry.list_models(model_type="sdxl")

        return [
            {
                "name": model.name,
                "type": model.model_type,
                "size_mb": model.size_mb,
                "description": model.description,
                "tags": model.tags,
            }
            for model in base_models
        ]

    def get_available_loras(self) -> List[Dict[str, Any]]:
        """Get list of available LoRA models"""
        loras = self.model_registry.list_models(model_type="lora")

        return [
            {
                "name": lora.name,
                "size_mb": lora.size_mb,
                "description": lora.description,
                "tags": lora.tags,
                "loaded": lora.name in self.lora_manager.loaded_loras,
            }
            for lora in loras
        ]

    def export_pipeline_config(self, output_path: Union[str, Path]) -> bool:
        """Export current pipeline configuration"""
        if not self.pipeline_loaded:
            logger.error("No pipeline loaded")
            return False

        try:
            config = {
                "model": self.current_model,
                "model_type": "SDXL" if self.is_sdxl else "SD1.5",
                "scheduler": self.current_pipeline.scheduler.__class__.__name__,  # type: ignore
                "lora_stack": self.lora_manager.lora_stack.to_dict(),
                "settings": {
                    "low_vram_mode": self.low_vram_mode,
                    "attention_slicing": self.attention_slicing,
                    "memory_efficient_attention": self.memory_efficient_attention,
                    "safety_enabled": self.safety_checker is not None,
                    "watermark_enabled": self.watermark_processor is not None,
                },
                "exported_at": datetime.now().isoformat(),
            }

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            logger.info(f"Pipeline configuration exported: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            return False

    def load_pipeline_config(self, config_path: Union[str, Path]) -> bool:
        """Load pipeline configuration from file"""
        try:
            config_path = Path(config_path)

            if not config_path.exists():
                logger.error(f"Configuration file not found: {config_path}")
                return False

            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            # Load model
            model_name = config.get("model")
            if model_name and model_name != self.current_model:
                if not self.load_model(model_name):
                    return False

            # Set scheduler
            scheduler_name = config.get("scheduler")
            if scheduler_name:
                self.set_scheduler(scheduler_name)

            # Load LoRA stack
            lora_stack_config = config.get("lora_stack", {})
            if lora_stack_config.get("loras"):
                lora_configs = lora_stack_config["loras"]
                self.load_lora_stack(lora_configs)

            # Apply settings
            settings = config.get("settings", {})
            self.enable_low_vram_mode(settings.get("low_vram_mode", False))
            self.attention_slicing = settings.get("attention_slicing", True)
            self.memory_efficient_attention = settings.get(
                "memory_efficient_attention", True
            )

            logger.info(f"Pipeline configuration loaded: {config_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False


# Global pipeline manager instance
_pipeline_manager_instance = None


def get_pipeline_manager() -> T2IPipelineManager:
    """Get global pipeline manager instance"""
    global _pipeline_manager_instance
    if _pipeline_manager_instance is None:
        _pipeline_manager_instance = T2IPipelineManager()
    return _pipeline_manager_instance
