# ===== backend/core/pipeline_loader.py =====
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PipelineLoader:
    def __init__(self):
        self._caption_processor = None
        self._caption_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_caption_pipeline(self):
        """Load BLIP-2 caption pipeline with memory optimization"""
        if self._caption_processor is None or self._caption_model is None:
            try:
                logger.info("Loading BLIP-2 caption model...")
                model_id = "Salesforce/blip-image-captioning-base"

                # Load with memory optimization
                self._caption_processor = BlipProcessor.from_pretrained(model_id)
                self._caption_model = BlipForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=(
                        torch.float16 if self.device == "cuda" else torch.float32
                    ),
                    device_map="auto" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True,
                )

                if torch.cuda.is_available():
                    # Enable memory optimizations
                    self._caption_model.half()

                logger.info(f"BLIP-2 loaded on {self.device}")

            except Exception as e:
                logger.error(f"Failed to load BLIP-2: {e}")
                raise

        return self._caption_processor, self._caption_model

    def generate_caption(self, image, max_length=50, num_beams=3, temperature=1.0):
        """Generate caption for image"""
        processor, model = self.get_caption_pipeline()

        try:
            # Process image
            inputs = processor(image, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate caption
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    temperature=temperature,
                    do_sample=temperature > 1.0,
                    early_stopping=True,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )

            # Decode result
            caption = processor.decode(outputs[0], skip_special_tokens=True)
            return caption

        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            raise


_chat_pipeline = None


def get_chat_pipeline():
    """Get or load chat pipeline for RAG"""
    global _chat_pipeline

    if _chat_pipeline is None:
        print("Loading chat pipeline for RAG...")
        try:
            from transformers import pipeline

            # Try Qwen first, fall back to smaller models
            model_options = [
                "Qwen/Qwen2-1.5B-Instruct",
                "microsoft/DialoGPT-medium",
                "facebook/blenderbot-400M-distill",
            ]

            for model_name in model_options:
                try:
                    _chat_pipeline = pipeline(
                        "text-generation",
                        model=model_name,
                        device_map="auto",
                        torch_dtype="auto",
                        trust_remote_code=True,
                    )
                    print(f"✅ Loaded chat model: {model_name}")
                    break
                except Exception as e:
                    print(f"❌ Failed to load {model_name}: {e}")
                    continue

            if _chat_pipeline is None:
                raise RuntimeError("Failed to load any chat model")

        except Exception as e:
            print(f"Error loading chat pipeline: {e}")
            raise

    return _chat_pipeline


def get_embedding_model():
    """Get embedding model for testing"""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="auto")
        return model
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        raise


# Global pipeline loader instance
pipeline_loader = PipelineLoader()
