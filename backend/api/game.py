# backend/api/game.py
from fastapi import APIRouter, HTTPException
from backend.schemas.game import (
    NewGameRequest,
    GameStepRequest,
    GameResponse,
    GameSaveRequest,
    GameLoadRequest,
    GameDifficulty,
)
from backend.core.game_engine import GameEngine
from backend.core.persona_manager import PersonaManager

router = APIRouter()

# Global game engine instance
game_engine = GameEngine()
persona_manager = PersonaManager()


@router.post("/game/new", response_model=GameResponse)
async def new_game(request: NewGameRequest):
    """Start a new text adventure game session"""
    try:
        session = game_engine.create_session(
            persona=request.persona,
            setting=request.setting,
            difficulty=request.difficulty,
            player_name=request.player_name,
        )

        return GameResponse(
            session_id=session.id,
            scene=session.current_scene,
            choices=session.available_choices,
            player_state=session.player_state,
            status=session.status,
            narrator_message=f"歡迎來到冒險世界！你正在與{persona_manager.get_persona(session.persona)['display_name']}開始一段奇妙的旅程。",
            turn_number=session.turn_number,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/game/step", response_model=GameResponse)
async def game_step(request: GameStepRequest):
    """Take an action in the game"""
    try:
        session = game_engine.process_action(
            session_id=request.session_id,
            action=request.action,
            message=request.message,
        )

        # Get character dialogue from recent history
        character_dialogue = None
        if session.game_history:
            character_dialogue = session.game_history[-1].get("response", "")

        return GameResponse(
            session_id=session.id,
            scene=session.current_scene,
            choices=session.available_choices,
            player_state=session.player_state,
            status=session.status,
            character_dialogue=character_dialogue,
            turn_number=session.turn_number,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/game/session/{session_id}", response_model=GameResponse)
async def get_game_session(session_id: str):
    """Get current game session state"""
    try:
        session = game_engine.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return GameResponse(
            session_id=session.id,
            scene=session.current_scene,
            choices=session.available_choices,
            player_state=session.player_state,
            status=session.status,
            turn_number=session.turn_number,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/game/save")
async def save_game(request: GameSaveRequest):
    """Save game session"""
    try:
        success = game_engine.save_session(request.session_id, request.save_name)
        if not success:
            raise HTTPException(
                status_code=404, detail="Session not found or save failed"
            )

        return {"status": "success", "save_name": request.save_name}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/game/load", response_model=GameResponse)
async def load_game(request: GameLoadRequest):
    """Load game session"""
    try:
        session = game_engine.load_session(request.save_name)
        if not session:
            raise HTTPException(status_code=404, detail="Save file not found")

        return GameResponse(
            session_id=session.id,
            scene=session.current_scene,
            choices=session.available_choices,
            player_state=session.player_state,
            status=session.status,
            narrator_message=f"遊戲已載入：{request.save_name}",
            turn_number=session.turn_number,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/game/saves")
async def list_saves():
    """List available save files"""
    try:
        saves = game_engine.list_saves()
        return {"saves": saves}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/game/personas")
async def list_personas():
    """List available game personas"""
    try:
        personas = persona_manager.get_persona_descriptions()
        return {"personas": personas}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/game/cleanup")
async def cleanup_sessions():
    """Clean up old inactive sessions"""
    try:
        game_engine.cleanup_old_sessions()
        return {"status": "success", "message": "Old sessions cleaned up"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
