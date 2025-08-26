# core/t2i/pipeline.py - SD/SDXL pipeline management
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import torch
import gc
import json, time
import numpy as np
from PIL import Image

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

from core.config import get_cache_paths, get_model_path


class PipelineManager:
    """Manages SD/SDXL pipelines with caching and LoRA support"""

    def __init__(self):
        self.cache_paths = get_cache_paths()
        self.pipelines: Dict[str, Any] = {}
        self.current_pipeline: Optional[str] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Memory optimization flags
        self.enable_xformers = False
        self.enable_slicing = True
        self.enable_cpu_offload = False  # Enable if VRAM < 8GB

    def _setup_memory_optimization(self, pipeline):
        """Apply memory optimization settings"""
        if self.enable_xformers and hasattr(
            pipeline.unet, "enable_xformers_memory_efficient_attention"
        ):
            try:
                pipeline.unet.enable_xformers_memory_efficient_attention()
                if hasattr(pipeline, "vae"):
                    pipeline.vae.enable_xformers_memory_efficient_attention()
                print("[Pipeline] XFormers optimization enabled")
            except Exception as e:
                print(f"[Pipeline] XFormers failed: {e}")

        if self.enable_slicing:
            pipeline.enable_attention_slicing("auto")
            if hasattr(pipeline, "enable_vae_slicing"):
                pipeline.enable_vae_slicing()
            print("[Pipeline] Attention slicing enabled")

        if self.enable_cpu_offload:
            pipeline.enable_model_cpu_offload()
            print("[Pipeline] CPU offload enabled")

        return pipeline

    def load_pipeline(self, model_id: str, pipeline_type: str = "sdxl") -> Any:
        """Load SD pipeline with caching"""
        cache_key = f"{pipeline_type}_{model_id.replace('/', '_')}"

        if cache_key in self.pipelines:
            print(f"[Pipeline] Using cached pipeline: {cache_key}")
            pipeline = self.pipelines[cache_key]
            self.current_pipeline = cache_key
            return pipeline

        print(f"[Pipeline] Loading {pipeline_type} pipeline: {model_id}")

        # Clear GPU memory before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        try:
            # Load pipeline based on type
            if pipeline_type.lower() == "sdxl":
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_id,
                    torch_dtype=self.dtype,
                    use_safetensors=True,
                    variant="fp16" if self.dtype == torch.float16 else None,
                    device_map="auto" if self.enable_cpu_offload else None,
                )
                # Setup scheduler
                pipeline.scheduler = EulerDiscreteScheduler.from_config(
                    pipeline.scheduler.config, timestep_spacing="trailing"
                )
            else:  # SD 1.5
                pipeline = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=self.dtype,
                    use_safetensors=True,
                    device_map="auto" if self.enable_cpu_offload else None,
                )
                # Setup scheduler
                pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipeline.scheduler.config, use_karras_sigmas=True
                )

            # Move to device if not using CPU offload
            if not self.enable_cpu_offload:
                pipeline = pipeline.to(self.device)

            # Apply optimizations
            pipeline = self._setup_memory_optimization(pipeline)

            # Cache pipeline
            self.pipelines[cache_key] = pipeline
            self.current_pipeline = cache_key

            print(f"[Pipeline] Successfully loaded: {cache_key}")
            return pipeline

        except Exception as e:
            print(f"[Pipeline] Failed to load {model_id}: {e}")
            raise

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 768,
        height: int = 768,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        num_images: int = 1,
        strength: Optional[float] = None,
        init_image: Optional[Image.Image] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate image with current pipeline"""

        if not self.current_pipeline:
            # Auto-load SDXL by default
            self.load_pipeline("stabilityai/stable-diffusion-xl-base-1.0", "sdxl")

        pipeline = self.pipelines[self.current_pipeline]  # type: ignore

        # Setup generator for reproducible results
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()  # type: ignore
            generator = torch.Generator(device=self.device).manual_seed(seed)  # type: ignore

        start_time = time.time()

        try:
            # Prepare generation parameters
            gen_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
                "num_images_per_prompt": num_images,
            }

            # Add dimensions for txt2img
            if init_image is None:
                gen_kwargs.update({"width": width, "height": height})

            # img2img mode
            if init_image is not None and strength is not None:
                gen_kwargs.update({"image": init_image, "strength": strength})
                # Use img2img pipeline if available
                if hasattr(pipeline, "img2img"):
                    result = pipeline.img2img(**gen_kwargs)
                else:
                    # Convert to img2img pipeline temporarily
                    if "sdxl" in self.current_pipeline:  # type: ignore
                        img2img_pipeline = StableDiffusionXLImg2ImgPipeline(
                            **pipeline.components
                        )
                    else:
                        img2img_pipeline = StableDiffusionImg2ImgPipeline(
                            **pipeline.components
                        )
                    result = img2img_pipeline(**gen_kwargs)
            else:
                # Standard txt2img generation
                result = pipeline(**gen_kwargs)

            elapsed_time = time.time() - start_time

            return {
                "images": result.images,  # type: ignore
                "seed": seed,
                "elapsed_time": elapsed_time,
                "metadata": {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "seed": seed,
                    "pipeline": self.current_pipeline,
                    "scheduler": pipeline.scheduler.__class__.__name__,
                },
            }

        except Exception as e:
            print(f"[Pipeline] Generation failed: {e}")
            # Try to recover memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

    def switch_scheduler(self, scheduler_name: str):
        """Switch scheduler for current pipeline"""
        if not self.current_pipeline:
            return False

        pipeline = self.pipelines[self.current_pipeline]

        scheduler_classes = {
            "ddim": DDIMScheduler,
            "dpm": DPMSolverMultistepScheduler,
            "euler": EulerDiscreteScheduler,
            "euler_a": EulerAncestralDiscreteScheduler,
        }

        scheduler_class = scheduler_classes.get(scheduler_name.lower())
        if scheduler_class:
            pipeline.scheduler = scheduler_class.from_config(pipeline.scheduler.config)
            print(f"[Pipeline] Switched to {scheduler_name} scheduler")
            return True

        return False

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get current pipeline information"""
        if not self.current_pipeline:
            return {"error": "No pipeline loaded"}

        pipeline = self.pipelines[self.current_pipeline]

        return {
            "current_pipeline": self.current_pipeline,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "scheduler": pipeline.scheduler.__class__.__name__,
            "loaded_pipelines": list(self.pipelines.keys()),
            "memory_optimizations": {
                "xformers": self.enable_xformers,
                "attention_slicing": self.enable_slicing,
                "cpu_offload": self.enable_cpu_offload,
            },
        }

    def unload_pipeline(self, cache_key: Optional[str] = None):
        """Unload pipeline to free memory"""
        if cache_key:
            if cache_key in self.pipelines:
                del self.pipelines[cache_key]
                if self.current_pipeline == cache_key:
                    self.current_pipeline = None
                print(f"[Pipeline] Unloaded: {cache_key}")
        else:
            # Unload all pipelines
            self.pipelines.clear()
            self.current_pipeline = None
            print("[Pipeline] Unloaded all pipelines")

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
