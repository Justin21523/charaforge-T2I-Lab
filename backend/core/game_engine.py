# backend/core/game_engine.py
import os
import json
import time
from typing import Dict, Optional, List
from pathlib import Path
from backend.models.game_session import GameSession
from backend.core.persona_manager import PersonaManager
from backend.core.content_generator import ContentGenerator
from backend.schemas.game import GameDifficulty, GameStatus


class GameEngine:
    def __init__(self):
        self.sessions: Dict[str, GameSession] = {}
        self.persona_manager = PersonaManager()
        self.content_generator = ContentGenerator()
        self.save_directory = (
            Path(os.getenv("AI_CACHE_ROOT", "/tmp"))
            / "outputs"
            / "multi-modal-lab"
            / "game_saves"
        )
        self.save_directory.mkdir(parents=True, exist_ok=True)

    def create_session(
        self,
        persona: str,
        setting: str = "modern",
        difficulty: GameDifficulty = GameDifficulty.NORMAL,
        player_name: Optional[str] = None,
    ) -> GameSession:
        """Create a new game session"""
        # Validate persona
        if persona not in self.persona_manager.list_personas():
            raise ValueError(f"Unknown persona: {persona}")

        # Create session
        session = GameSession(persona, setting, difficulty, player_name)

        # Generate opening scene
        scene, choices = self.content_generator.generate_opening_scene(session)
        session.update_scene(scene, choices)

        # Store session
        self.sessions[session.id] = session

        # Add to memory
        persona_info = self.persona_manager.get_persona(persona)
        session.add_to_memory(
            f"開始了與{persona_info['display_name']}的冒險", importance=5
        )

        return session

    def get_session(self, session_id: str) -> Optional[GameSession]:
        """Get session by ID"""
        return self.sessions.get(session_id)

    def process_action(
        self, session_id: str, action: str, message: Optional[str] = None
    ) -> GameSession:
        """Process player action and update game state"""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        if session.status != GameStatus.ACTIVE:
            raise ValueError(f"Session is not active: {session.status}")

        # Combine action and message
        full_action = action
        if message:
            full_action += f" ({message})"

        # Generate next scene
        try:
            scene, choices, character_response = (
                self.content_generator.generate_next_scene(session, full_action)
            )

            # Update session
            session.update_scene(scene, choices)
            session.add_to_history(full_action, scene.description, character_response)

            # Update player state based on action and difficulty
            self._update_player_state(session, action)

            # Add to memory if significant
            if session.turn_number % 5 == 0:  # Every 5 turns
                session.add_to_memory(f"重要時刻：{scene.title}", importance=3)

            return session

        except Exception as e:
            print(f"[GameEngine] Action processing failed: {e}")
            # Fallback to safe state
            session.status = GameStatus.PAUSED
            raise e

    def _update_player_state(self, session: GameSession, action: str):
        """Update player state based on action and difficulty"""
        # Energy management
        energy_cost = 5
        if session.difficulty == GameDifficulty.HARD:
            energy_cost = 8
        elif session.difficulty == GameDifficulty.EASY:
            energy_cost = 3

        session.player_state.energy = max(0, session.player_state.energy - energy_cost)

        # Health regeneration over time
        if session.turn_number % 3 == 0 and session.player_state.health < 100:
            regen = 5 if session.difficulty == GameDifficulty.EASY else 3
            session.player_state.health = min(100, session.player_state.health + regen)

        # Mood changes based on action type
        mood_map = {
            "greet": "friendly",
            "observe": "curious",
            "ask": "inquisitive",
            "continue": "determined",
            "rest": "peaceful",
            "think": "contemplative",
        }

        for key, mood in mood_map.items():
            if key in action.lower():
                session.player_state.mood = mood
                break

    def save_session(self, session_id: str, save_name: str) -> bool:
        """Save game session to file"""
        session = self.get_session(session_id)
        if not session:
            return False

        try:
            save_path = self.save_directory / f"{save_name}.json"
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)

            print(f"[GameEngine] Session saved: {save_path}")
            return True

        except Exception as e:
            print(f"[GameEngine] Save failed: {e}")
            return False

    def load_session(self, save_name: str) -> Optional[GameSession]:
        """Load game session from file"""
        try:
            save_path = self.save_directory / f"{save_name}.json"
            if not save_path.exists():
                return None

            with open(save_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            session = GameSession.from_dict(data)
            self.sessions[session.id] = session

            print(f"[GameEngine] Session loaded: {save_path}")
            return session

        except Exception as e:
            print(f"[GameEngine] Load failed: {e}")
            return None

    def list_saves(self) -> List[str]:
        """List available save files"""
        try:
            saves = []
            for save_file in self.save_directory.glob("*.json"):
                saves.append(save_file.stem)
            return sorted(saves)
        except Exception as e:
            print(f"[GameEngine] Failed to list saves: {e}")
            return []

    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Remove old inactive sessions"""
        current_time = time.time()
        to_remove = []

        for session_id, session in self.sessions.items():
            age_hours = (current_time - session.last_action_time) / 3600
            if age_hours > max_age_hours and session.status != GameStatus.ACTIVE:
                to_remove.append(session_id)

        for session_id in to_remove:
            del self.sessions[session_id]

        if to_remove:
            print(f"[GameEngine] Cleaned up {len(to_remove)} old sessions")
