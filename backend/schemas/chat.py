# backend/schemas/chat.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str
    timestamp: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Conversation history")
    max_length: int = Field(default=200, ge=10, le=1000)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)
    persona_id: Optional[str] = Field(default=None, description="Persona to use")
    session_id: Optional[str] = Field(default=None, description="Session for memory")
    safety_level: str = Field(default="moderate", description="Safety filtering level")


class ChatResponse(BaseModel):
    message: ChatMessage
    session_id: str
    model_used: str
    safety_filtered: bool = False
    persona_used: Optional[str] = None
    elapsed_ms: int
    usage: Dict[str, int] = Field(default_factory=dict)


class SessionInfo(BaseModel):
    session_id: str
    message_count: int
    created_at: float
    last_activity: float
    persona_id: Optional[str] = None
