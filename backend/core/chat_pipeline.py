# backend/core/chat_pipeline.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
import time
from typing import List, Dict, Optional
from backend.schemas.chat import ChatMessage, MessageRole
from backend.core.conversation_manager import ConversationManager
from backend.core.safety_filter import SafetyFilter


class ChatPipeline:
    def __init__(self, model_name: str = "Qwen/Qwen-7B-Chat", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.loaded = False

        # Initialize managers
        cache_root = os.environ.get("AI_CACHE_ROOT", "/tmp/ai_cache")
        self.conversation_manager = ConversationManager(cache_root)
        self.safety_filter = SafetyFilter()

    def load_model(self):
        """Load chat model with optimizations"""
        if self.loaded:
            return

        print(f"Loading chat model: {self.model_name}")
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=os.environ.get("TRANSFORMERS_CACHE"),
                trust_remote_code=True,
            )

            # Configure model loading
            load_kwargs = {
                "cache_dir": os.environ.get("TRANSFORMERS_CACHE"),
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
            }

            # Device mapping and quantization
            if self.device == "auto":
                load_kwargs["device_map"] = "auto"

            # VRAM optimization
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                if gpu_memory < 16:  # Less than 16GB for 7B model
                    load_kwargs["load_in_4bit"] = True
                    print("🔧 Using 4-bit quantization for chat model")

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **load_kwargs
            )

            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto" if self.device == "auto" else self.device,
            )

            self.loaded = True
            print(f"✅ Chat model loaded")

        except Exception as e:
            print(f"❌ Failed to load chat model: {e}")
            raise

    def format_conversation(
        self, messages: List[ChatMessage], persona_prompt: Optional[str] = None
    ) -> str:
        """Format conversation for model input"""
        if not messages:
            return ""

        # Build conversation string
        conversation_parts = []

        # Add persona/system prompt if provided
        if persona_prompt:
            conversation_parts.append(f"System: {persona_prompt}")

        # Add conversation history
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                conversation_parts.append(f"System: {msg.content}")
            elif msg.role == MessageRole.USER:
                conversation_parts.append(f"User: {msg.content}")
            elif msg.role == MessageRole.ASSISTANT:
                conversation_parts.append(f"Assistant: {msg.content}")

        # Add prompt for next response
        conversation_parts.append("Assistant:")

        return "\n".join(conversation_parts)

    def generate_response(
        self,
        messages: List[ChatMessage],
        session_id: Optional[str] = None,
        persona_prompt: Optional[str] = None,
        max_length: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        safety_level: str = "moderate",
    ) -> Dict:
        """Generate chat response"""
        if not self.loaded:
            self.load_model()

        start_time = time.time()

        try:
            # Get the last user message for safety filtering
            user_message = None
            for msg in reversed(messages):
                if msg.role == MessageRole.USER:
                    user_message = msg.content
                    break

            # Safety filter input
            if user_message:
                is_safe, filtered_input, warnings = self.safety_filter.filter_input(
                    user_message, safety_level
                )
                if not is_safe:
                    return {
                        "response": "I cannot respond to that request due to safety guidelines.",
                        "safety_filtered": True,
                        "elapsed_ms": int((time.time() - start_time) * 1000),
                    }

            # Format conversation
            prompt = self.format_conversation(messages, persona_prompt)

            # Generate response
            with torch.no_grad():
                outputs = self.pipeline(
                    prompt,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_full_text=False,
                )

            # Extract response
            response_text = outputs[0]["generated_text"].strip()

            # Remove "Assistant:" prefix if present
            if response_text.startswith("Assistant:"):
                response_text = response_text[10:].strip()

            # Safety filter output
            is_safe_output, filtered_response = self.safety_filter.filter_output(
                response_text, safety_level
            )

            elapsed_ms = int((time.time() - start_time) * 1000)

            return {
                "response": filtered_response,
                "safety_filtered": not is_safe_output,
                "elapsed_ms": elapsed_ms,
                "usage": {
                    "prompt_tokens": len(self.tokenizer.encode(prompt)),
                    "completion_tokens": len(self.tokenizer.encode(response_text)),
                },
            }

        except Exception as e:
            print(f"❌ Chat generation failed: {e}")
            return {
                "response": f"Sorry, I encountered an error: {str(e)}",
                "safety_filtered": False,
                "elapsed_ms": int((time.time() - start_time) * 1000),
            }


# Global pipeline instance
_chat_pipeline = None


def get_chat_pipeline() -> ChatPipeline:
    """Get or create chat pipeline singleton"""
    global _chat_pipeline
    if _chat_pipeline is None:
        model_name = os.getenv(
            "CHAT_MODEL", "microsoft/DialoGPT-medium"
        )  # Fallback to smaller model
        device = os.getenv("DEVICE", "auto")
        _chat_pipeline = ChatPipeline(model_name, device)
    return _chat_pipeline
