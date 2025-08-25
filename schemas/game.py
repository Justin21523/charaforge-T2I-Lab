# backend/schemas/game.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum


class GameDifficulty(str, Enum):
    EASY = "easy"
    NORMAL = "normal"
    HARD = "hard"


class GameStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class Choice(BaseModel):
    id: str = Field(..., description="Unique choice identifier")
    text: str = Field(..., description="Choice description for player")
    consequence: Optional[str] = Field(None, description="Hint about consequences")


class GameScene(BaseModel):
    scene_id: str = Field(..., description="Unique scene identifier")
    title: str = Field(..., description="Scene title")
    description: str = Field(..., description="Scene description")
    mood: str = Field(default="neutral", description="Scene emotional tone")
    location: str = Field(..., description="Current location")


class PlayerState(BaseModel):
    health: int = Field(default=100, ge=0, le=100)
    energy: int = Field(default=100, ge=0, le=100)
    mood: str = Field(default="curious")
    inventory: List[str] = Field(default_factory=list)
    stats: Dict[str, int] = Field(default_factory=dict)


class NewGameRequest(BaseModel):
    persona: str = Field(..., description="Character persona to use")
    setting: str = Field(default="modern", description="Game setting/world")
    difficulty: GameDifficulty = Field(default=GameDifficulty.NORMAL)
    player_name: Optional[str] = Field(None, description="Player character name")


class GameStepRequest(BaseModel):
    session_id: str = Field(..., description="Game session ID")
    action: str = Field(..., description="Player action or choice ID")
    message: Optional[str] = Field(None, description="Additional player message")


class GameResponse(BaseModel):
    session_id: str
    scene: GameScene
    choices: List[Choice]
    player_state: PlayerState
    status: GameStatus
    narrator_message: Optional[str] = None
    character_dialogue: Optional[str] = None
    turn_number: int = 0


class GameSaveRequest(BaseModel):
    session_id: str
    save_name: str


class GameLoadRequest(BaseModel):
    save_name: str
