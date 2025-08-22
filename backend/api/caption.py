# ===== backend/api/caption.py =====
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from backend.schemas.caption import CaptionRequest, CaptionResponse
from backend.core.pipeline_loader import pipeline_loader
import io
import time
from PIL import Image
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


async def validate_image(image: UploadFile = File(...)) -> Image.Image:
    """Validate and load uploaded image"""
    try:
        # Check file type
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Check file size (max 10MB)
        contents = await image.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large (max 10MB)")

        # Load and convert image
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Check image dimensions
        if pil_image.size[0] * pil_image.size[1] > 4096 * 4096:
            raise HTTPException(status_code=400, detail="Image resolution too high")

        return pil_image

    except Exception as e:
        logger.error(f"Image validation failed: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")


@router.post("/caption", response_model=CaptionResponse)
async def generate_caption(
    image: Image.Image = Depends(validate_image),
    max_length: int = 50,
    num_beams: int = 3,
    temperature: float = 1.0,
):
    """Generate image caption using BLIP-2"""
    start_time = time.time()

    try:
        # Validate parameters
        request = CaptionRequest(
            max_length=max_length, num_beams=num_beams, temperature=temperature
        )

        # Generate caption
        caption = pipeline_loader.generate_caption(
            image,
            max_length=request.max_length,
            num_beams=request.num_beams,
            temperature=request.temperature,
        )

        elapsed_ms = int((time.time() - start_time) * 1000)

        logger.info(f"Caption generated in {elapsed_ms}ms: {caption[:50]}...")

        return CaptionResponse(
            caption=caption,
            confidence=0.9,  # Placeholder - BLIP doesn't provide confidence
            model_used="blip-image-captioning-base",
            elapsed_ms=elapsed_ms,
        )

    except Exception as e:
        logger.error(f"Caption generation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Caption generation failed: {str(e)}"
        )


@router.get("/caption/models")
async def list_caption_models():
    """List available caption models"""
    return {
        "models": [
            {
                "id": "blip-image-captioning-base",
                "name": "BLIP Image Captioning Base",
                "description": "Salesforce BLIP model for image captioning",
                "loaded": pipeline_loader._caption_model is not None,
            }
        ]
    }
