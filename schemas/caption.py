# ===== backend/schemas/caption.py =====
from pydantic import BaseModel, Field
from typing import Optional
import time


class CaptionRequest(BaseModel):
    max_length: int = Field(
        default=50, ge=10, le=200, description="Maximum caption length"
    )
    num_beams: int = Field(
        default=3, ge=1, le=10, description="Number of beams for generation"
    )
    temperature: float = Field(
        default=1.0, ge=0.1, le=2.0, description="Generation temperature"
    )


class CaptionResponse(BaseModel):
    caption: str = Field(description="Generated image caption")
    confidence: float = Field(description="Model confidence score")
    model_used: str = Field(description="Model identifier")
    elapsed_ms: int = Field(description="Generation time in milliseconds")
    timestamp: str = Field(default_factory=lambda: str(int(time.time())))
