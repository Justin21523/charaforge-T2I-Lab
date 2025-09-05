# core/t2i/pipeline.py - T2I Pipeline Manager
"""
SD1.5/SDXL pipeline management with memory optimization
Supports LoRA loading, safety filtering, and caching
"""

import gc
import logging
import time
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from PIL import Image
import numpy as np

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.schedulers.scheduling_euler_ancestral_discrete import (
    EulerAncestralDiscreteScheduler,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    StableDiffusionImg2ImgPipeline,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import (
    StableDiffusionXLImg2ImgPipeline,
)

from core.config import get_settings, get_app_paths, get_model_config
from core.shared_cache import get_shared_cache
from core.exceptions import (
    T2IError,
    PipelineError,
    ModelLoadError,
    CUDAOutOfMemoryError,
    handle_cuda_oom,
    memory_cleanup_decorator,
)

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Generation result container"""

    images: List[Image.Image]
    metadata: Dict[str, Any]
    generation_time: float
    seed_used: int

    def save_images(self, output_dir: Path, prefix: str = "generated") -> List[str]:
        """Save images and return file paths"""
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []

        for i, image in enumerate(self.images):
            filename = f"{prefix}_{self.seed_used}_{i:03d}.png"
            filepath = output_dir / filename

            # Add metadata to image
            metadata_str = f"Parameters: {self.metadata.get('prompt', '')[:100]}..."
            image.save(filepath, pnginfo=self._create_png_info())
            saved_paths.append(str(filepath))

        return saved_paths

    def _create_png_info(self) -> Any:
        """Create PNG metadata"""
        try:
            from PIL.PngImagePlugin import PngInfo

            metadata = PngInfo()
            for key, value in self.metadata.items():
                if isinstance(value, (str, int, float)):
                    metadata.add_text(key, str(value))
            return metadata
        except ImportError:
            return None


class T2IPipeline:
    """Text-to-Image Pipeline Manager"""

    def __init__(self, model_type: str = "sd15", device: Optional[str] = None):
        self.model_type = model_type
        self.device = device or self._get_optimal_device()
        self.settings = get_settings()
        self.cache = get_shared_cache()

        # Pipeline components
        self.pipeline = None
        self.loaded_loras: Dict[str, float] = {}  # lora_id -> weight
        self.safety_checker = None

        # Performance settings
        self.enable_memory_efficient_attention = True
        self.enable_vae_slicing = True
        self.enable_cpu_offload = self.settings.model.enable_cpu_offload

        logger.info(f"T2I Pipeline initialized: {model_type} on {self.device}")

    def _get_optimal_device(self) -> str:
        """Determine optimal device configuration"""
        try:
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )

                if gpu_memory_gb >= 12:
                    return "cuda"
                elif gpu_memory_gb >= 8:
                    return "cuda"  # Will use memory optimizations
                else:
                    return "cuda"  # Will use aggressive optimizations
            else:
                return "cpu"
        except Exception:
            return "cpu"

    @handle_cuda_oom
    @memory_cleanup_decorator(cleanup_before=True, cleanup_after=True)
    def load_pipeline(self, model_name: Optional[str] = None) -> bool:
        """Load the diffusion pipeline"""
        try:
            logger.info(f"Loading {self.model_type} pipeline...")

            # Determine model to load
            if model_name is None:
                if self.model_type == "sd15":
                    model_name = self.settings.model.default_sd15_model
                elif self.model_type == "sdxl":
                    model_name = self.settings.model.default_sdxl_model
                else:
                    raise ValueError(f"Unknown model type: {self.model_type}")

            # Configure pipeline loading
            pipeline_kwargs = {
                "torch_dtype": (
                    torch.float16 if self.device == "cuda" else torch.float32
                ),
                "device_map": "auto" if self.settings.model.low_vram_mode else None,
                "use_safetensors": True,
            }

            # Add 4-bit/8-bit loading if needed
            if self.settings.model.low_vram_mode and self.device == "cuda":
                try:
                    from transformers import BitsAndBytesConfig

                    pipeline_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
                    )
                except ImportError:
                    logger.warning(
                        "BitsAndBytesConfig not available, skipping quantization"
                    )

            # Load pipeline
            if self.model_type == "sd15":
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    model_name, **pipeline_kwargs
                )
            elif self.model_type == "sdxl":
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_name, **pipeline_kwargs
                )

            # Apply optimizations
            self._apply_optimizations()

            # Move to device if not using device_map
            if pipeline_kwargs["device_map"] is None:
                self.pipeline = self.pipeline.to(self.device)

            # Load safety checker
            self._load_safety_checker()

            logger.info(f"Pipeline loaded successfully: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            if "out of memory" in str(e).lower():
                raise CUDAOutOfMemoryError(model_name)
            else:
                raise ModelLoadError(model_name, str(e))

    def _apply_optimizations(self):
        """Apply memory and performance optimizations"""
        if self.pipeline is None:
            return

        try:
            # Enable memory efficient attention
            if self.enable_memory_efficient_attention:
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("✅ xFormers memory efficient attention enabled")
                except Exception as e:
                    logger.warning(f"⚠️ xFormers not available: {e}")
                    try:
                        self.pipeline.enable_attention_slicing()
                        logger.info("✅ Attention slicing enabled as fallback")
                    except Exception:
                        pass

            # Enable VAE slicing for memory efficiency
            if self.enable_vae_slicing:
                try:
                    self.pipeline.enable_vae_slicing()
                    logger.info("✅ VAE slicing enabled")
                except Exception as e:
                    logger.warning(f"VAE slicing failed: {e}")

            # Enable CPU offload if requested
            if self.enable_cpu_offload:
                try:
                    self.pipeline.enable_model_cpu_offload()
                    logger.info("✅ CPU offload enabled")
                except Exception as e:
                    logger.warning(f"CPU offload failed: {e}")

            # Enable tiled VAE for very low VRAM
            if self.settings.model.low_vram_mode:
                try:
                    self.pipeline.enable_vae_tiling()
                    logger.info("✅ VAE tiling enabled")
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Optimization setup failed: {e}")

    def _load_safety_checker(self):
        """Load NSFW safety checker"""
        try:
            # TODO: Implement safety checker loading
            # For now, use built-in safety checker if available
            if (
                hasattr(self.pipeline, "safety_checker")
                and self.pipeline.safety_checker
            ):
                self.safety_checker = self.pipeline.safety_checker
                logger.info("✅ Built-in safety checker loaded")
            else:
                logger.warning("⚠️ No safety checker available")
        except Exception as e:
            logger.warning(f"Safety checker loading failed: {e}")

    def load_lora(self, lora_id: str, weight: float = 1.0) -> bool:
        """Load LoRA weights"""
        try:
            if self.pipeline is None:
                raise PipelineError("Pipeline not loaded", self.model_type)

            # Get LoRA path from cache
            model_info = self.cache.get_model_info(lora_id)
            if not model_info:
                logger.error(f"LoRA not found in cache: {lora_id}")
                return False

            lora_path = model_info.path

            # Load LoRA using diffusers
            try:
                self.pipeline.load_lora_weights(str(lora_path), adapter_name=lora_id)
                self.pipeline.set_adapters([lora_id], adapter_weights=[weight])

                self.loaded_loras[lora_id] = weight
                logger.info(f"✅ LoRA loaded: {lora_id} (weight: {weight})")
                return True

            except Exception as e:
                logger.error(f"Failed to load LoRA {lora_id}: {e}")
                return False

        except Exception as e:
            logger.error(f"LoRA loading error: {e}")
            return False

    def unload_lora(self, lora_id: str) -> bool:
        """Unload specific LoRA"""
        try:
            if lora_id in self.loaded_loras:
                # Remove from adapters
                remaining_loras = [k for k in self.loaded_loras.keys() if k != lora_id]
                remaining_weights = [self.loaded_loras[k] for k in remaining_loras]

                if remaining_loras:
                    self.pipeline.set_adapters(
                        remaining_loras, adapter_weights=remaining_weights
                    )
                else:
                    self.pipeline.disable_lora()

                # Remove LoRA weights
                try:
                    self.pipeline.delete_adapters([lora_id])
                except Exception:
                    pass

                del self.loaded_loras[lora_id]
                logger.info(f"✅ LoRA unloaded: {lora_id}")
                return True
            else:
                logger.warning(f"LoRA not loaded: {lora_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to unload LoRA {lora_id}: {e}")
            return False

    def unload_all_loras(self):
        """Unload all LoRA weights"""
        try:
            if self.pipeline and self.loaded_loras:
                self.pipeline.disable_lora()
                self.loaded_loras.clear()
                logger.info("✅ All LoRAs unloaded")
        except Exception as e:
            logger.error(f"Failed to unload all LoRAs: {e}")

    @handle_cuda_oom
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        num_images: int = 1,
        seed: Optional[int] = None,
        scheduler: str = "euler_a",
        **kwargs,
    ) -> GenerationResult:
        """Generate images from text prompt"""

        if self.pipeline is None:
            raise PipelineError("Pipeline not loaded")

        start_time = time.time()

        try:
            # Set up generation parameters
            if seed is None:
                seed = torch.randint(0, 2**32 - 1, (1,)).item()

            generator = torch.Generator(device=self.device).manual_seed(seed)

            # Adjust dimensions for model type
            if self.model_type == "sd15":
                width = min(width, 768)  # SD1.5 works best at 512-768
                height = min(height, 768)
            elif self.model_type == "sdxl":
                width = min(width, 1024)  # SDXL can handle up to 1024
                height = min(height, 1024)

            # Ensure dimensions are multiples of 8
            width = (width // 8) * 8
            height = (height // 8) * 8

            # Prepare generation arguments
            generation_args = {
                "prompt": prompt,
                "negative_prompt": negative_prompt or "",
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images_per_prompt": num_images,
                "generator": generator,
                **kwargs,
            }

            # Set scheduler if different
            self._set_scheduler(scheduler)

            logger.info(f"Generating {num_images} image(s): {prompt[:50]}...")

            # Generate images
            with torch.inference_mode():
                result = self.pipeline(**generation_args)

            # Extract images (handle both old and new diffusers API)
            if hasattr(result, "images"):
                images = result.images
            else:
                images = result[0]

            # Safety check (if available)
            if hasattr(result, "nsfw_content_detected") and any(
                result.nsfw_content_detected
            ):
                logger.warning("⚠️ NSFW content detected, filtering images")
                # Filter out NSFW images
                safe_images = []
                for i, is_nsfw in enumerate(result.nsfw_content_detected):
                    if not is_nsfw:
                        safe_images.append(images[i])
                images = safe_images

                if not images:
                    from core.exceptions import NSFWContentError

                    raise NSFWContentError()

            generation_time = time.time() - start_time

            # Create result metadata
            metadata = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "scheduler": scheduler,
                "model_type": self.model_type,
                "loaded_loras": list(self.loaded_loras.keys()),
                "generation_time": generation_time,
                "seed": seed,
            }

            logger.info(f"✅ Generation completed in {generation_time:.2f}s")

            return GenerationResult(
                images=images,
                metadata=metadata,
                generation_time=generation_time,
                seed_used=seed,
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            if "out of memory" in str(e).lower():
                raise CUDAOutOfMemoryError("")
            else:
                raise T2IError(f"Generation failed: {str(e)}")

    def _set_scheduler(self, scheduler_name: str):
        """Set pipeline scheduler"""
        try:

            if scheduler_name == "euler_a":
                self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                    self.pipeline.scheduler.config
                )
            elif scheduler_name == "dpm":
                self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    self.pipeline.scheduler.config
                )
            elif scheduler_name == "ddim":
                self.pipeline.scheduler = DDIMScheduler.from_config(
                    self.pipeline.scheduler.config
                )
            elif scheduler_name == "lms":
                self.pipeline.scheduler = LMSDiscreteScheduler.from_config(
                    self.pipeline.scheduler.config
                )

        except Exception as e:
            logger.warning(f"Failed to set scheduler {scheduler_name}: {e}")

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        memory_info = {
            "system_memory_gb": 0.0,
            "gpu_memory_allocated_gb": 0.0,
            "gpu_memory_reserved_gb": 0.0,
            "gpu_memory_total_gb": 0.0,
        }

        try:
            import psutil

            memory_info["system_memory_gb"] = psutil.virtual_memory().used / (1024**3)

            if torch.cuda.is_available():
                memory_info["gpu_memory_allocated_gb"] = (
                    torch.cuda.memory_allocated() / (1024**3)
                )
                memory_info["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (
                    1024**3
                )
                memory_info["gpu_memory_total_gb"] = torch.cuda.get_device_properties(
                    0
                ).total_memory / (1024**3)

        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")

        return memory_info

    def cleanup(self):
        """Cleanup pipeline and free memory"""
        try:
            if self.pipeline:
                # Unload all LoRAs
                self.unload_all_loras()

                # Move pipeline to CPU to free GPU memory
                if hasattr(self.pipeline, "to"):
                    self.pipeline.to("cpu")

                # Clear pipeline
                self.pipeline = None

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Force garbage collection
            gc.collect()

            logger.info("✅ Pipeline cleanup completed")

        except Exception as e:
            logger.error(f"Pipeline cleanup failed: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status information"""
        return {
            "model_type": self.model_type,
            "device": self.device,
            "pipeline_loaded": self.pipeline is not None,
            "loaded_loras": self.loaded_loras.copy(),
            "memory_usage": self.get_memory_usage(),
            "optimizations": {
                "memory_efficient_attention": self.enable_memory_efficient_attention,
                "vae_slicing": self.enable_vae_slicing,
                "cpu_offload": self.enable_cpu_offload,
            },
        }


# ===== Pipeline Manager (Global Instance) =====


class T2IPipelineManager:
    """Global T2I Pipeline Manager"""

    def __init__(self):
        self.pipelines: Dict[str, T2IPipeline] = {}
        self.active_pipeline: Optional[str] = None

    def get_pipeline(self, model_type: str = "sd15") -> T2IPipeline:
        """Get or create pipeline for model type"""
        if model_type not in self.pipelines:
            self.pipelines[model_type] = T2IPipeline(model_type)

        return self.pipelines[model_type]

    def load_pipeline(
        self, model_type: str = "sd15", model_name: Optional[str] = None
    ) -> bool:
        """Load specific pipeline"""
        pipeline = self.get_pipeline(model_type)

        if pipeline.load_pipeline(model_name):
            self.active_pipeline = model_type
            return True
        return False

    def get_active_pipeline(self) -> Optional[T2IPipeline]:
        """Get currently active pipeline"""
        if self.active_pipeline and self.active_pipeline in self.pipelines:
            return self.pipelines[self.active_pipeline]
        return None

    def cleanup_all(self):
        """Cleanup all pipelines"""
        for pipeline in self.pipelines.values():
            pipeline.cleanup()
        self.pipelines.clear()
        self.active_pipeline = None


# Global pipeline manager instance
_pipeline_manager = T2IPipelineManager()


def get_pipeline_manager() -> T2IPipelineManager:
    """Get global pipeline manager"""
    return _pipeline_manager
