# backend/schemas/vqa.py
from pydantic import BaseModel, Field
from typing import Optional, List
import base64


class VQARequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image or file path")
    question: str = Field(..., description="Question about the image")
    max_length: int = Field(default=100, ge=10, le=500)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)
    language: str = Field(default="auto", description="Response language: auto/zh/en")


class VQAResponse(BaseModel):
    question: str
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_used: str
    language_detected: str
    elapsed_ms: int


# backend/core/vqa_pipeline.py
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import AutoProcessor, AutoModelForCausalLM
import os
from PIL import Image
import base64
import io
import time
from typing import Union, Optional


class VQAPipeline:
    def __init__(
        self, model_name: str = "llava-hf/llava-1.5-7b-hf", device: str = "auto"
    ):
        self.model_name = model_name
        self.device = device
        self.processor = None
        self.model = None
        self.loaded = False

    def load_model(self):
        """Load VQA model with memory optimization"""
        if self.loaded:
            return

        print(f"Loading VQA model: {self.model_name}")
        try:
            # Choose model based on name
            if "qwen" in self.model_name.lower():
                self._load_qwen_vl()
            else:
                self._load_llava()

            self.loaded = True
            print(f"✅ VQA model loaded on {self.model.device}")

        except Exception as e:
            print(f"❌ Failed to load VQA model: {e}")
            raise

    def _load_llava(self):
        """Load LLaVA model with optimizations"""
        self.processor = LlavaNextProcessor.from_pretrained(
            self.model_name, cache_dir=os.environ.get("TRANSFORMERS_CACHE")
        )

        # Load with optimizations
        load_kwargs = {
            "cache_dir": os.environ.get("TRANSFORMERS_CACHE"),
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
        }

        # Add device mapping for auto-balancing
        if self.device == "auto":
            load_kwargs["device_map"] = "auto"

        # Add quantization if VRAM is limited
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory < 12:  # Less than 12GB VRAM
                load_kwargs["load_in_4bit"] = True
                print("🔧 Using 4-bit quantization for low VRAM")

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_name, **load_kwargs
        )

        # Enable memory optimizations
        if hasattr(self.model, "enable_model_cpu_offload"):
            self.model.enable_model_cpu_offload()

    def _load_qwen_vl(self):
        """Load Qwen-VL model (placeholder for future)"""
        # TODO: Implement Qwen-VL loading
        raise NotImplementedError("Qwen-VL support coming soon")

    def predict(
        self,
        image: Image.Image,
        question: str,
        max_length: int = 100,
        temperature: float = 0.7,
    ) -> dict:
        """Generate answer for visual question"""
        if not self.loaded:
            self.load_model()

        start_time = time.time()

        try:
            # Prepare inputs
            prompt = f"USER: <image>\n{question}\nASSISTANT:"

            inputs = self.processor(prompt, image, return_tensors="pt").to(
                self.model.device
            )

            # Generate with temperature sampling
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )

            # Decode response
            full_response = self.processor.decode(outputs[0], skip_special_tokens=True)

            # Extract only the assistant's response
            if "ASSISTANT:" in full_response:
                answer = full_response.split("ASSISTANT:")[-1].strip()
            else:
                answer = full_response.strip()

            elapsed_ms = int((time.time() - start_time) * 1000)

            return {
                "answer": answer,
                "confidence": 0.85,  # Placeholder confidence
                "model_used": self.model_name,
                "elapsed_ms": elapsed_ms,
            }

        except Exception as e:
            print(f"❌ VQA prediction failed: {e}")
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "confidence": 0.0,
                "model_used": self.model_name,
                "elapsed_ms": int((time.time() - start_time) * 1000),
            }


# Global pipeline instance
_vqa_pipeline = None


def get_vqa_pipeline() -> VQAPipeline:
    """Get or create VQA pipeline singleton"""
    global _vqa_pipeline
    if _vqa_pipeline is None:
        model_name = os.getenv("VQA_MODEL", "llava-hf/llava-1.5-7b-hf")
        device = os.getenv("DEVICE", "auto")
        _vqa_pipeline = VQAPipeline(model_name, device)
    return _vqa_pipeline


# backend/api/vqa.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from backend.schemas.vqa import VQARequest, VQAResponse
from backend.core.vqa_pipeline import get_vqa_pipeline
from PIL import Image
import base64
import io
import re

router = APIRouter()


def load_image_from_input(image_input: str) -> Image.Image:
    """Load image from base64 string or file path"""
    try:
        # Try base64 first
        if image_input.startswith("data:image"):
            # Remove data URL prefix
            image_data = image_input.split(",")[1]
        else:
            image_data = image_input

        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image

    except:
        # Try as file path
        try:
            image = Image.open(image_input).convert("RGB")
            return image
        except:
            raise ValueError(
                "Invalid image input: must be base64 string or valid file path"
            )


def detect_language(text: str) -> str:
    """Simple language detection"""
    # Check for Chinese characters
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    if len(chinese_chars) > len(text) * 0.3:
        return "zh"
    return "en"


@router.post("/vqa", response_model=VQAResponse)
async def visual_question_answering(request: VQARequest):
    """
    Visual Question Answering using LLaVA

    Accepts an image and question, returns an intelligent answer about the image content.
    Supports both Chinese and English questions.
    """
    try:
        # Load image
        image = load_image_from_input(request.image)

        # Get VQA pipeline
        pipeline = get_vqa_pipeline()

        # Generate answer
        result = pipeline.predict(
            image=image,
            question=request.question,
            max_length=request.max_length,
            temperature=request.temperature,
        )

        # Detect language
        language_detected = detect_language(request.question)

        return VQAResponse(
            question=request.question,
            answer=result["answer"],
            confidence=result["confidence"],
            model_used=result["model_used"],
            language_detected=language_detected,
            elapsed_ms=result["elapsed_ms"],
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VQA processing failed: {str(e)}")


@router.post("/vqa/upload", response_model=VQAResponse)
async def vqa_with_upload(
    image: UploadFile = File(...),
    question: str = Form(...),
    max_length: int = Form(default=100),
    temperature: float = Form(default=0.7),
):
    """
    VQA endpoint that accepts file upload directly
    """
    try:
        # Read uploaded image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Get VQA pipeline
        pipeline = get_vqa_pipeline()

        # Generate answer
        result = pipeline.predict(
            image=pil_image,
            question=question,
            max_length=max_length,
            temperature=temperature,
        )

        # Detect language
        language_detected = detect_language(question)

        return VQAResponse(
            question=question,
            answer=result["answer"],
            confidence=result["confidence"],
            model_used=result["model_used"],
            language_detected=language_detected,
            elapsed_ms=result["elapsed_ms"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"VQA upload processing failed: {str(e)}"
        )


@router.get("/vqa/models")
async def list_vqa_models():
    """List available VQA models"""
    return {
        "available_models": [
            "llava-hf/llava-1.5-7b-hf",
            "llava-hf/llava-1.5-13b-hf",
            # "Qwen/Qwen-VL-Chat" # Coming soon
        ],
        "current_model": get_vqa_pipeline().model_name,
        "loaded": get_vqa_pipeline().loaded,
    }
