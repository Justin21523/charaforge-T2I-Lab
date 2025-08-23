# backend/schemas/personas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class PersonaConfig(BaseModel):
    id: str = Field(..., description="Unique persona identifier")
    name: str = Field(..., description="Display name")
    description: str = Field(..., description="Persona description")
    personality_traits: List[str] = Field(default_factory=list)
    speaking_style: str = Field(default="friendly and helpful")
    knowledge_areas: List[str] = Field(default_factory=list)
    safety_rules: List[str] = Field(default_factory=list)
    system_prompt: str = Field(..., description="System prompt template")
    memory_slots: int = Field(default=10, ge=1, le=50)
    temperature_override: Optional[float] = None


class CreatePersonaRequest(BaseModel):
    name: str
    description: str
    personality_traits: List[str] = Field(default_factory=list)
    speaking_style: str = "friendly and helpful"
    knowledge_areas: List[str] = Field(default_factory=list)
    safety_rules: List[str] = Field(default_factory=list)
    custom_prompt: Optional[str] = None
