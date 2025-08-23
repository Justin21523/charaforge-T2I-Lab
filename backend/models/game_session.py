# backend/models/game_session.py
import uuid
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from backend.schemas.game import (
    GameScene,
    Choice,
    PlayerState,
    GameStatus,
    GameDifficulty,
)


class GameSession:
    def __init__(
        self,
        persona: str,
        setting: str,
        difficulty: GameDifficulty,
        player_name: Optional[str] = None,
    ):
        self.id = str(uuid.uuid4())
        self.persona = persona
        self.setting = setting
        self.difficulty = difficulty
        self.player_name = player_name or "Adventurer"
        self.status = GameStatus.ACTIVE
        self.created_at = datetime.now()
        self.turn_number = 0

        # Game state
        self.current_scene = None
        self.available_choices = []
        self.player_state = PlayerState()
        self.game_history = []
        self.character_memory = []

        # Metadata
        self.last_action_time = time.time()
        self.total_playtime = 0

    def update_scene(self, scene: GameScene, choices: List[Choice]):
        """Update current scene and available choices"""
        self.current_scene = scene
        self.available_choices = choices
        self.turn_number += 1
        self.last_action_time = time.time()

    def add_to_history(
        self, action: str, scene_description: str, character_response: str
    ):
        """Add action and response to game history"""
        self.game_history.append(
            {
                "turn": self.turn_number,
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "scene": scene_description,
                "response": character_response,
            }
        )

        # Keep only last 20 turns to manage memory
        if len(self.game_history) > 20:
            self.game_history = self.game_history[-20:]

    def add_to_memory(self, memory_item: str, importance: int = 1):
        """Add item to character memory with importance weighting"""
        self.character_memory.append(
            {
                "content": memory_item,
                "importance": importance,
                "timestamp": datetime.now().isoformat(),
                "turn": self.turn_number,
            }
        )

        # Sort by importance and keep top items
        self.character_memory.sort(key=lambda x: x["importance"], reverse=True)
        if len(self.character_memory) > 50:
            self.character_memory = self.character_memory[:50]

    def get_context_summary(self) -> str:
        """Get summary of recent context for LLM"""
        context = f"Player: {self.player_name}\n"
        context += f"Setting: {self.setting}\n"
        context += f"Turn: {self.turn_number}\n\n"

        if self.current_scene:
            context += f"Current Scene: {self.current_scene.title}\n"
            context += f"Location: {self.current_scene.location}\n"
            context += f"Mood: {self.current_scene.mood}\n\n"

        context += f"Player State:\n"
        context += f"- Health: {self.player_state.health}/100\n"
        context += f"- Energy: {self.player_state.energy}/100\n"
        context += f"- Mood: {self.player_state.mood}\n"
        context += f"- Inventory: {', '.join(self.player_state.inventory) if self.player_state.inventory else 'Empty'}\n\n"

        if self.character_memory:
            context += "Important Memories:\n"
            for memory in self.character_memory[:5]:
                context += f"- {memory['content']}\n"
            context += "\n"

        if self.game_history:
            context += "Recent History:\n"
            for event in self.game_history[-3:]:
                context += f"Turn {event['turn']}: {event['action']} -> {event['response'][:100]}...\n"

        return context

    def to_dict(self) -> Dict:
        """Convert session to dictionary for serialization"""
        return {
            "id": self.id,
            "persona": self.persona,
            "setting": self.setting,
            "difficulty": self.difficulty.value,
            "player_name": self.player_name,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "turn_number": self.turn_number,
            "current_scene": self.current_scene.dict() if self.current_scene else None,
            "available_choices": [choice.dict() for choice in self.available_choices],
            "player_state": self.player_state.dict(),
            "game_history": self.game_history,
            "character_memory": self.character_memory,
            "total_playtime": self.total_playtime,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "GameSession":
        """Create session from dictionary"""
        session = cls(
            persona=data["persona"],
            setting=data["setting"],
            difficulty=GameDifficulty(data["difficulty"]),
            player_name=data["player_name"],
        )

        session.id = data["id"]
        session.status = GameStatus(data["status"])
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.turn_number = data["turn_number"]
        session.game_history = data["game_history"]
        session.character_memory = data["character_memory"]
        session.total_playtime = data.get("total_playtime", 0)

        if data["current_scene"]:
            session.current_scene = GameScene(**data["current_scene"])

        session.available_choices = [
            Choice(**choice) for choice in data["available_choices"]
        ]
        session.player_state = PlayerState(**data["player_state"])

        return session
