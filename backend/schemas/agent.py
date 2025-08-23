# backend/schemas/agent.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class ToolCall(BaseModel):
    tool_name: str = Field(..., description="Name of the tool to call")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Tool parameters"
    )


class AgentRequest(BaseModel):
    query: str = Field(..., description="User query for the agent")
    tools: Optional[List[str]] = Field(
        default=None, description="Available tools (None = all)"
    )
    max_iterations: int = Field(default=3, description="Maximum tool call iterations")
    temperature: float = Field(
        default=0.1, description="LLM temperature for tool selection"
    )


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: int


class AgentResponse(BaseModel):
    query: str
    final_answer: str
    tool_calls: List[ToolResult]
    reasoning_steps: List[str]
    total_time_ms: int
