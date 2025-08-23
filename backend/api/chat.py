# backend/api/chat.py
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException
from backend.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ChatMessage,
    MessageRole,
    SessionInfo,
)
from backend.core.chat_pipeline import get_chat_pipeline
from backend.core.conversation_manager import ConversationManager
import time
import os

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    """
    Generate chat response with conversation memory

    Supports multi-turn conversations, persona-based responses,
    and safety filtering.
    """
    try:
        # Get chat pipeline
        pipeline = get_chat_pipeline()

        # Create or get session
        session_id = request.session_id
        if not session_id:
            session_id = pipeline.conversation_manager.create_session(
                request.persona_id
            )

        # Get persona prompt if specified
        persona_prompt = None
        if request.persona_id:
            persona_prompt = _get_persona_prompt(request.persona_id)

        # Add user message to conversation history
        if request.messages and request.messages[-1].role == MessageRole.USER:
            user_message = request.messages[-1]
            pipeline.conversation_manager.add_message(session_id, user_message)

        # Get context for generation
        context_messages = pipeline.conversation_manager.get_context_for_generation(
            session_id, max_context=10
        )

        # Combine with new messages
        all_messages = context_messages + request.messages

        # Generate response
        result = pipeline.generate_response(
            messages=all_messages,
            session_id=session_id,
            persona_prompt=persona_prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            safety_level=request.safety_level,
        )

        # Create assistant message
        assistant_message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=result["response"],
            timestamp=time.time(),
        )

        # Add to conversation history
        pipeline.conversation_manager.add_message(session_id, assistant_message)

        return ChatResponse(
            message=assistant_message,
            session_id=session_id,
            model_used=pipeline.model_name,
            safety_filtered=result.get("safety_filtered", False),
            persona_used=request.persona_id,
            elapsed_ms=result["elapsed_ms"],
            usage=result.get("usage", {}),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@router.get("/chat/sessions", response_model=List[SessionInfo])
async def list_chat_sessions():
    """List all active chat sessions"""
    try:
        pipeline = get_chat_pipeline()
        sessions = pipeline.conversation_manager.list_sessions()
        return sessions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/sessions/{session_id}/history")
async def get_session_history(session_id: str, limit: int = 20):
    """Get conversation history for a session"""
    try:
        pipeline = get_chat_pipeline()
        history = pipeline.conversation_manager.get_history(session_id, limit)
        return {"session_id": session_id, "messages": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat session"""
    try:
        pipeline = get_chat_pipeline()
        success = pipeline.conversation_manager.delete_session(session_id)
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/models")
async def list_chat_models():
    """List available chat models"""
    return {
        "available_models": [
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-large",
            "Qwen/Qwen-7B-Chat",
            "meta-llama/Llama-2-7b-chat-hf",
        ],
        "current_model": get_chat_pipeline().model_name,
        "loaded": get_chat_pipeline().loaded,
    }


def _get_persona_prompt(persona_id: str) -> Optional[str]:
    """Get persona system prompt"""
    try:
        personas_file = os.path.join("configs", "personas.json")
        if os.path.exists(personas_file):
            import json

            with open(personas_file, "r", encoding="utf-8") as f:
                personas = json.load(f)

            for persona in personas.get("personas", []):
                if persona.get("id") == persona_id:
                    return persona.get("system_prompt", "")

        return None
    except Exception:
        return None
