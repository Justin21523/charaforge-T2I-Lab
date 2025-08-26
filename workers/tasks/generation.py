# workers/tasks/generation.py - Image generation tasks
import uuid
import json
from datetime import datetime
import logging
from pathlib import Path
import traceback
from typing import Dict, Any, Optional, List, Union

from workers.celery_app import celery_app
from core.t2i.pipeline import PipelineManager
from core.t2i.lora_manager import LoRAManager
from core.t2i.controlnet import ControlNetManager
from core.t2i.safety import SafetyChecker
from core.t2i.watermark import WatermarkManager
from core.config import get_cache_paths


logger = logging.getLogger(__name__)


@celery_app.task(name="generate_single_image")
def generate_single_image_task(
    prompt: str,
    negative_prompt: str = "",
    width: int = 768,
    height: int = 768,
    steps: int = 25,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    lora_ids: List[str] = None,  # type: ignore
    lora_weights: List[float] = None,  # type: ignore
    controlnet_type: Optional[str] = None,
    controlnet_image_path: Optional[str] = None,
    safety_check: bool = True,
    add_watermark: bool = True,
    pipeline_type: str = "sdxl",
    **kwargs,
) -> Dict[str, Any]:
    """Generate a single image with all features"""

    try:
        logger.info(f"Starting image generation: {prompt[:50]}...")

        # Initialize components
        pipeline_manager = PipelineManager()
        lora_manager = LoRAManager(pipeline_manager)
        controlnet_manager = ControlNetManager(pipeline_manager)
        safety_checker = SafetyChecker() if safety_check else None
        watermark_manager = WatermarkManager() if add_watermark else None

        # Load base pipeline
        base_model = kwargs.get(
            "base_model",
            (
                "stabilityai/stable-diffusion-xl-base-1.0"
                if pipeline_type == "sdxl"
                else "runwayml/stable-diffusion-v1-5"
            ),
        )
        pipeline_manager.load_pipeline(base_model, pipeline_type)

        # Load LoRAs if specified
        if lora_ids:
            weights = lora_weights or [1.0] * len(lora_ids)
            for lora_id, weight in zip(lora_ids, weights):
                success = lora_manager.load_lora(lora_id, weight)
                if not success:
                    logger.warning(f"Failed to load LoRA: {lora_id}")

        # Handle ControlNet generation
        if controlnet_type and controlnet_image_path:
            try:
                from PIL import Image

                control_image = Image.open(controlnet_image_path)

                result = controlnet_manager.generate_with_controlnet(
                    prompt=prompt,
                    control_images=control_image,
                    controlnet_types=controlnet_type,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    **kwargs,
                )
            except Exception as e:
                logger.error(f"ControlNet generation failed: {e}")
                # Fallback to regular generation
                result = pipeline_manager.generate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    **kwargs,
                )
        else:
            # Standard generation
            result = pipeline_manager.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                **kwargs,
            )

        if not result["images"]:
            return {"status": "failed", "error": "No images generated"}

        generated_image = result["images"][0]

        # Safety check
        if safety_checker:
            safety_result = safety_checker.is_content_allowed(generated_image, prompt)
            if not safety_result["allowed"]:
                return {
                    "status": "blocked",
                    "reason": safety_result["reason"],
                    "content_hash": safety_result["content_hash"],
                }

        # Add watermark
        if watermark_manager:
            generated_image = watermark_manager.add_text_watermark(
                generated_image,
                template="attribution",
                metadata={"model": base_model.split("/")[-1], "seed": result["seed"]},
            )

        # Save image and metadata
        cache_paths = get_cache_paths()
        output_dir = (
            cache_paths.outputs / "generation" / datetime.now().strftime("%Y-%m-%d")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        image_id = str(uuid.uuid4())[:8]
        image_path = output_dir / f"{image_id}.png"
        metadata_path = output_dir / f"{image_id}.json"

        # Save image
        generated_image.save(image_path, "PNG")

        # Save metadata
        full_metadata = {
            **result["metadata"],
            "lora_ids": lora_ids or [],
            "lora_weights": lora_weights or [],
            "controlnet_type": controlnet_type,
            "safety_checked": safety_check,
            "watermarked": add_watermark,
            "generated_at": datetime.now().isoformat(),
            "image_id": image_id,
        }

        with open(metadata_path, "w") as f:
            json.dump(full_metadata, f, indent=2)

        logger.info(f"Image generated successfully: {image_path}")

        return {
            "status": "success",
            "image_path": str(image_path),
            "metadata_path": str(metadata_path),
            "seed": result["seed"],
            "elapsed_time": result["elapsed_time"],
            "metadata": full_metadata,
            "image_id": image_id,
        }

    except Exception as e:
        error_msg = f"Image generation failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())

        return {
            "status": "failed",
            "error": error_msg,
            "traceback": traceback.format_exc(),
        }


@celery_app.task(name="generate_image_variations")
def generate_image_variations_task(
    base_prompt: str,
    variations: List[str],
    common_params: Dict[str, Any],
    variation_seeds: List[int] = None,  # type: ignore
) -> Dict[str, Any]:
    """Generate multiple variations of an image"""

    try:
        logger.info(f"Generating {len(variations)} image variations")

        results = []
        base_seed = common_params.get("seed", 42)

        for i, variation in enumerate(variations):
            # Use provided seeds or generate sequential ones
            if variation_seeds and i < len(variation_seeds):
                seed = variation_seeds[i]
            else:
                seed = base_seed + i

            # Combine base prompt with variation
            full_prompt = f"{base_prompt}, {variation}" if variation else base_prompt

            # Generate image
            result = generate_single_image_task.delay(
                prompt=full_prompt, seed=seed, **common_params
            ).get()

            results.append({"variation": variation, "seed": seed, "result": result})

        return {
            "status": "completed",
            "total_variations": len(variations),
            "results": results,
        }

    except Exception as e:
        return {"status": "failed", "error": f"Variation generation failed: {str(e)}"}
