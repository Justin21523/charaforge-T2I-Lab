# workers/tasks/generation.py - Image generation tasks
from workers.celery_app import celery_app
from core.t2i.pipeline import PipelineManager
from core.t2i.lora_manager import LoRAManager
from core.t2i.safety import SafetyChecker
from core.config import get_cache_paths
from typing import Dict, Any, Optional
import uuid
from datetime import datetime


@celery_app.task(name="generate_single_image")
def generate_single_image_task(
    prompt: str,
    negative_prompt: str = "",
    width: int = 768,
    height: int = 768,
    steps: int = 25,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    lora_ids: list = None,
    lora_weights: list = None,
    safety_check: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """Generate a single image"""

    try:
        # Initialize pipeline
        pipeline_manager = PipelineManager()
        lora_manager = LoRAManager(pipeline_manager)
        safety_checker = SafetyChecker() if safety_check else None

        # Load base pipeline
        pipeline_manager.load_pipeline(
            "stabilityai/stable-diffusion-xl-base-1.0", "sdxl"
        )

        # Load LoRAs if specified
        if lora_ids:
            weights = lora_weights or [1.0] * len(lora_ids)
            for lora_id, weight in zip(lora_ids, weights):
                lora_manager.load_lora(lora_id, weight)

        # Generate image
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

        # Safety check
        if safety_checker and result["images"]:
            is_nsfw, scores = safety_checker.check_nsfw(result["images"][0])
            if is_nsfw:
                return {
                    "status": "blocked",
                    "reason": "Content filtered by safety check",
                    "safety_scores": scores,
                }

        # Save image and metadata
        cache_paths = get_cache_paths()
        output_dir = (
            cache_paths.outputs / "generation" / datetime.now().strftime("%Y-%m-%d")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        image_id = str(uuid.uuid4())[:8]
        image_path = output_dir / f"{image_id}.png"
        metadata_path = output_dir / f"{image_id}.json"

        # TODO: Save actual image and metadata
        # result["images"][0].save(image_path)
        # with open(metadata_path, 'w') as f:
        #     json.dump(result["metadata"], f, indent=2)

        return {
            "status": "success",
            "image_path": str(image_path),
            "metadata_path": str(metadata_path),
            "seed": result["seed"],
            "metadata": result["metadata"],
        }

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
