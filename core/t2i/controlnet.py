# core/t2i/controlnet.py - ControlNet integration
from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np
from PIL import Image, ImageOps
from peft import LoraConfig, get_peft_model
from pathlib import Path
import cv2
import torch
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
from diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl import (
    StableDiffusionXLControlNetPipeline,
)
from diffusers.pipelines.controlnet.pipeline_controlnet import (
    StableDiffusionControlNetPipeline,
)
from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel

# Import ControlNet preprocessors
try:
    from controlnet_aux import (
        OpenposeDetector,
        MidasDetector,
        CannyDetector,
        LineartDetector,
        LineartAnimeDetector,
        MLSDdetector,
        HEDdetector,
        NormalBaeDetector,
    )

    CONTROLNET_AUX_AVAILABLE = True
except ImportError:
    print("[ControlNet] controlnet_aux not available, using basic preprocessing")
    CONTROLNET_AUX_AVAILABLE = False

from t2i_controlnet import ControlNetModel

from core.config import get_cache_paths, get_model_path
from core.t2i.pipeline import PipelineManager


class ControlNetManager:
    """Manages ControlNet preprocessing and integration"""

    def __init__(self, pipeline_manager: PipelineManager):
        self.pipeline_manager = pipeline_manager
        self.processors = {}  # controlnet_type -> processor
        self.controlnet_models = {}  # controlnet_type -> model
        self.cache_paths = get_cache_paths()

        # ControlNet model configurations
        self.controlnet_configs = {
            "pose": {
                "sd15": "lllyasviel/sd-controlnet-openpose",
                "sdxl": "thibaud/controlnet-openpose-sdxl-1.0",
            },
            "depth": {
                "sd15": "lllyasviel/sd-controlnet-depth",
                "sdxl": "diffusers/controlnet-depth-sdxl-1.0",
            },
            "canny": {
                "sd15": "lllyasviel/sd-controlnet-canny",
                "sdxl": "diffusers/controlnet-canny-sdxl-1.0",
            },
            "lineart": {
                "sd15": "lllyasviel/sd-controlnet-mlsd",
                "sdxl": "TheMistoAI/MistoLine",
            },
            "normal": {
                "sd15": "lllyasviel/sd-controlnet-normalbae",
                "sdxl": "diffusers/controlnet-normal-sdxl-1.0",
            },
            "hed": {
                "sd15": "lllyasviel/sd-controlnet-hed",
                "sdxl": None,  # Not available for SDXL
            },
        }

    def _get_pipeline_type(self) -> str:
        """Determine current pipeline type (sd15 or sdxl)"""
        if not self.pipeline_manager.current_pipeline:
            return "sdxl"  # Default
        return (
            "sdxl"
            if "sdxl" in self.pipeline_manager.current_pipeline.lower()
            else "sd15"
        )

    def load_controlnet_processor(self, controlnet_type: str) -> bool:
        """Load ControlNet preprocessor"""
        if controlnet_type in self.processors:
            return True

        print(f"[ControlNet] Loading {controlnet_type} processor")

        if not CONTROLNET_AUX_AVAILABLE:
            # Use basic OpenCV-based preprocessing
            self.processors[controlnet_type] = self._get_basic_processor(
                controlnet_type
            )
            return True

        try:
            if controlnet_type == "pose":
                self.processors[controlnet_type] = OpenposeDetector.from_pretrained(
                    "lllyasviel/Annotators"
                )
            elif controlnet_type == "depth":
                self.processors[controlnet_type] = MidasDetector.from_pretrained(
                    "lllyasviel/Annotators"
                )
            elif controlnet_type == "canny":
                self.processors[controlnet_type] = CannyDetector()
            elif controlnet_type == "lineart":
                self.processors[controlnet_type] = LineartDetector.from_pretrained(
                    "lllyasviel/Annotators"
                )
            elif controlnet_type == "lineart_anime":
                self.processors[controlnet_type] = LineartAnimeDetector.from_pretrained(
                    "lllyasviel/Annotators"
                )
            elif controlnet_type == "mlsd":
                self.processors[controlnet_type] = MLSDdetector.from_pretrained(
                    "lllyasviel/Annotators"
                )
            elif controlnet_type == "hed":
                self.processors[controlnet_type] = HEDdetector.from_pretrained(
                    "lllyasviel/Annotators"
                )
            elif controlnet_type == "normal":
                self.processors[controlnet_type] = NormalBaeDetector.from_pretrained(
                    "lllyasviel/Annotators"
                )
            else:
                print(f"[ControlNet] Unknown type: {controlnet_type}")
                return False

            print(f"[ControlNet] Loaded {controlnet_type} processor")
            return True

        except Exception as e:
            print(f"[ControlNet] Failed to load {controlnet_type} processor: {e}")
            # Fallback to basic processor
            self.processors[controlnet_type] = self._get_basic_processor(
                controlnet_type
            )
            return True

    def _get_basic_processor(self, controlnet_type: str):
        """Get basic OpenCV-based processor as fallback"""
        if controlnet_type == "canny":
            return lambda img: self._basic_canny(img)
        elif controlnet_type == "depth":
            return lambda img: self._basic_depth(img)
        elif controlnet_type == "pose":
            return lambda img: self._basic_pose(img)
        else:
            return lambda img: img  # Identity function

    def _basic_canny(self, image: Image.Image) -> Image.Image:
        """Basic Canny edge detection"""
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(edges)

    def _basic_depth(self, image: Image.Image) -> Image.Image:
        """Basic depth map (grayscale conversion)"""
        return ImageOps.grayscale(image).convert("RGB")

    def _basic_pose(self, image: Image.Image) -> Image.Image:
        """Basic pose (edge detection as placeholder)"""
        return self._basic_canny(image)

    def load_controlnet_model(self, controlnet_type: str) -> bool:
        """Load ControlNet model"""
        if controlnet_type in self.controlnet_models:
            return True

        pipeline_type = self._get_pipeline_type()

        if controlnet_type not in self.controlnet_configs:
            print(f"[ControlNet] Unknown type: {controlnet_type}")
            return False

        model_id = self.controlnet_configs[controlnet_type].get(pipeline_type)
        if not model_id:
            print(f"[ControlNet] {controlnet_type} not available for {pipeline_type}")
            return False

        try:
            print(f"[ControlNet] Loading {controlnet_type} model: {model_id}")

            controlnet = ControlNetModel.from_pretrained(
                model_id,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
                use_safetensors=True,
            )

            if torch.cuda.is_available():
                controlnet = controlnet.to("cuda")

            self.controlnet_models[controlnet_type] = controlnet
            print(f"[ControlNet] Loaded {controlnet_type} model")
            return True

        except Exception as e:
            print(f"[ControlNet] Failed to load {controlnet_type} model: {e}")
            return False

    def preprocess_image(
        self, image: Image.Image, controlnet_type: str, **kwargs
    ) -> Image.Image:
        """Preprocess image for ControlNet"""
        if not self.load_controlnet_processor(controlnet_type):
            raise ValueError(f"Failed to load ControlNet processor: {controlnet_type}")

        processor = self.processors[controlnet_type]

        try:
            # Apply processor with parameters
            if controlnet_type == "canny":
                low_threshold = kwargs.get("low_threshold", 100)
                high_threshold = kwargs.get("high_threshold", 200)
                processed_image = processor(image, low_threshold, high_threshold)
            elif controlnet_type == "pose":
                # OpenPose parameters
                processed_image = processor(
                    image,
                    hand_and_face=kwargs.get("hand_and_face", False),
                    output_type="pil",
                )
            elif controlnet_type == "depth":
                # Depth parameters
                processed_image = processor(
                    image,
                    detect_resolution=kwargs.get("detect_resolution", 512),
                    image_resolution=kwargs.get("image_resolution", 512),
                )
            else:
                # Generic processing
                processed_image = processor(image)

            return processed_image

        except Exception as e:
            print(f"[ControlNet] Preprocessing failed for {controlnet_type}: {e}")
            return image  # Return original image as fallback

    def create_controlnet_pipeline(
        self, controlnet_types: Union[str, List[str]], base_pipeline=None
    ) -> Any:
        """Create ControlNet pipeline"""
        if isinstance(controlnet_types, str):
            controlnet_types = [controlnet_types]

        # Load all required ControlNet models
        controlnets = []
        for ctype in controlnet_types:
            if not self.load_controlnet_model(ctype):
                raise ValueError(f"Failed to load ControlNet model: {ctype}")
            controlnets.append(self.controlnet_models[ctype])

        # Use single ControlNet or MultiControlNet
        if len(controlnets) == 1:
            controlnet = controlnets[0]
        else:
            controlnet = MultiControlNetModel(controlnets)

        # Get base pipeline info
        pipeline_type = self._get_pipeline_type()

        # Get base model from current pipeline or use default
        if base_pipeline:
            base_model_id = base_pipeline
        elif self.pipeline_manager.current_pipeline:
            # Extract model ID from current pipeline (simplified)
            base_model_id = (
                "stabilityai/stable-diffusion-xl-base-1.0"
                if pipeline_type == "sdxl"
                else "runwayml/stable-diffusion-v1-5"
            )
        else:
            base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"

        try:
            print(f"[ControlNet] Creating pipeline with {controlnet_types}")

            # Create ControlNet pipeline
            if pipeline_type == "sdxl":
                pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                    base_model_id,
                    controlnet=controlnet,
                    torch_dtype=(
                        torch.float16 if torch.cuda.is_available() else torch.float32
                    ),
                    use_safetensors=True,
                    variant="fp16" if torch.cuda.is_available() else None,
                )
            else:
                pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                    base_model_id,
                    controlnet=controlnet,
                    torch_dtype=(
                        torch.float16 if torch.cuda.is_available() else torch.float32
                    ),
                    use_safetensors=True,
                )

            # Move to device
            if torch.cuda.is_available():
                pipeline = pipeline.to("cuda")

            # Apply optimizations
            pipeline = self.pipeline_manager._setup_memory_optimization(pipeline)

            return pipeline

        except Exception as e:
            print(f"[ControlNet] Failed to create pipeline: {e}")
            raise

    def generate_with_controlnet(
        self,
        prompt: str,
        control_images: Union[Image.Image, List[Image.Image]],
        controlnet_types: Union[str, List[str]],
        negative_prompt: str = "",
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        width: int = 768,
        height: int = 768,
        seed: Optional[int] = None,
        preprocess: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate image with ControlNet conditioning"""

        # Normalize inputs to lists
        if isinstance(controlnet_types, str):
            controlnet_types = [controlnet_types]

        if isinstance(control_images, Image.Image):
            control_images = [control_images]

        if isinstance(controlnet_conditioning_scale, (int, float)):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(
                controlnet_types  # type: ignore
            )

        if isinstance(control_guidance_start, (int, float)):
            control_guidance_start = [control_guidance_start] * len(controlnet_types)  # type: ignore

        if isinstance(control_guidance_end, (int, float)):
            control_guidance_end = [control_guidance_end] * len(controlnet_types)  # type: ignore

        # Preprocess control images
        processed_images = []
        for i, (image, ctype) in enumerate(zip(control_images, controlnet_types)):
            if preprocess:
                processed_image = self.preprocess_image(image, ctype, **kwargs)
            else:
                processed_image = image

            # Resize to target dimensions
            processed_image = processed_image.resize(
                (width, height), Image.Resampling.LANCZOS
            )
            processed_images.append(processed_image)

        # Create ControlNet pipeline
        pipeline = self.create_controlnet_pipeline(controlnet_types)

        # Setup generator
        generator = None
        if seed is not None:
            generator = torch.Generator(device=pipeline.device).manual_seed(seed)
        else:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()  # type: ignore
            generator = torch.Generator(device=pipeline.device).manual_seed(seed)  # type: ignore

        try:
            import time

            start_time = time.time()

            # Generation parameters
            gen_kwargs = {
                "prompt": prompt,
                "image": (
                    processed_images[0]
                    if len(processed_images) == 1
                    else processed_images
                ),
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "controlnet_conditioning_scale": (
                    controlnet_conditioning_scale[0]  # type: ignore
                    if len(controlnet_conditioning_scale) == 1  # type: ignore
                    else controlnet_conditioning_scale
                ),
                "control_guidance_start": (
                    control_guidance_start[0]  # type: ignore
                    if len(control_guidance_start) == 1  # type: ignore
                    else control_guidance_start
                ),
                "control_guidance_end": (
                    control_guidance_end[0]  # type: ignore
                    if len(control_guidance_end) == 1  # type: ignore
                    else control_guidance_end
                ),
                "generator": generator,
                "width": width,
                "height": height,
            }

            result = pipeline(**gen_kwargs)
            elapsed_time = time.time() - start_time

            return {
                "images": result.images,
                "control_images": processed_images,
                "seed": seed,
                "elapsed_time": elapsed_time,
                "metadata": {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "controlnet_types": controlnet_types,
                    "controlnet_conditioning_scale": controlnet_conditioning_scale,
                    "control_guidance_start": control_guidance_start,
                    "control_guidance_end": control_guidance_end,
                    "width": width,
                    "height": height,
                    "steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "seed": seed,
                },
            }

        except Exception as e:
            print(f"[ControlNet] Generation failed: {e}")
            raise
        finally:
            # Clean up pipeline to save memory
            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_available_controlnets(self) -> Dict[str, Dict]:
        """Get list of available ControlNet types and models"""
        pipeline_type = self._get_pipeline_type()
        available = {}

        for ctype, models in self.controlnet_configs.items():
            if pipeline_type in models and models[pipeline_type]:
                available[ctype] = {
                    "model_id": models[pipeline_type],
                    "loaded": ctype in self.controlnet_models,
                    "processor_loaded": ctype in self.processors,
                    "description": self._get_controlnet_description(ctype),
                }

        return available

    def _get_controlnet_description(self, controlnet_type: str) -> str:
        """Get description for ControlNet type"""
        descriptions = {
            "pose": "Human pose control using OpenPose keypoints",
            "depth": "Depth-based control for 3D structure",
            "canny": "Edge-based control using Canny edge detection",
            "lineart": "Line art control for clean line drawings",
            "lineart_anime": "Anime-style line art control",
            "mlsd": "Straight line detection for architectural images",
            "hed": "Holistically-nested edge detection",
            "normal": "Surface normal control for lighting and geometry",
        }
        return descriptions.get(controlnet_type, "Unknown ControlNet type")

    def unload_controlnet(self, controlnet_type: str):
        """Unload specific ControlNet model and processor"""
        if controlnet_type in self.controlnet_models:
            del self.controlnet_models[controlnet_type]
            print(f"[ControlNet] Unloaded model: {controlnet_type}")

        if controlnet_type in self.processors:
            del self.processors[controlnet_type]
            print(f"[ControlNet] Unloaded processor: {controlnet_type}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload_all_controlnets(self):
        """Unload all ControlNet models and processors"""
        self.controlnet_models.clear()
        self.processors.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[ControlNet] Unloaded all ControlNet models and processors")
