# backend/core/conversation_manager.py
import json
import time
import uuid
from typing import Dict, List, Optional
from pathlib import Path
from backend.schemas.chat import ChatMessage, MessageRole, SessionInfo


class ConversationManager:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir) / "conversations"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sessions: Dict[str, List[ChatMessage]] = {}
        self.session_info: Dict[str, SessionInfo] = {}

    def create_session(self, persona_id: Optional[str] = None) -> str:
        """Create a new conversation session"""
        session_id = str(uuid.uuid4())
        now = time.time()

        self.sessions[session_id] = []
        self.session_info[session_id] = SessionInfo(
            session_id=session_id,
            message_count=0,
            created_at=now,
            last_activity=now,
            persona_id=persona_id,
        )

        return session_id

    def add_message(self, session_id: str, message: ChatMessage) -> None:
        """Add message to session history"""
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        message.timestamp = time.time()
        self.sessions[session_id].append(message)

        # Update session info
        if session_id in self.session_info:
            self.session_info[session_id].message_count += 1
            self.session_info[session_id].last_activity = message.timestamp

        # Auto-save to disk
        self._save_session(session_id)

    def get_history(self, session_id: str, limit: int = 20) -> List[ChatMessage]:
        """Get conversation history for session"""
        if session_id not in self.sessions:
            self._load_session(session_id)

        messages = self.sessions.get(session_id, [])
        return messages[-limit:] if limit > 0 else messages

    def get_context_for_generation(
        self, session_id: str, max_context: int = 10
    ) -> List[ChatMessage]:
        """Get optimized context for generation"""
        history = self.get_history(session_id, max_context)

        # Keep system message + recent user/assistant pairs
        context = []
        for msg in history:
            if msg.role == MessageRole.SYSTEM or len(context) < max_context:
                context.append(msg)

        return context

    def delete_session(self, session_id: str) -> bool:
        """Delete conversation session"""
        try:
            if session_id in self.sessions:
                del self.sessions[session_id]
            if session_id in self.session_info:
                del self.session_info[session_id]

            # Delete from disk
            session_file = self.cache_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()

            return True
        except Exception:
            return False

    def list_sessions(self) -> List[SessionInfo]:
        """List all active sessions"""
        return list(self.session_info.values())

    def _save_session(self, session_id: str) -> None:
        """Save session to disk"""
        try:
            session_file = self.cache_dir / f"{session_id}.json"
            session_data = {
                "info": self.session_info[session_id].dict(),
                "messages": [msg.dict() for msg in self.sessions[session_id]],
            }

            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save session {session_id}: {e}")

    def _load_session(self, session_id: str) -> None:
        """Load session from disk"""
        try:
            session_file = self.cache_dir / f"{session_id}.json"
            if not session_file.exists():
                return

            with open(session_file, "r", encoding="utf-8") as f:
                session_data = json.load(f)

            # Restore session info
            self.session_info[session_id] = SessionInfo(**session_data["info"])

            # Restore messages
            self.sessions[session_id] = [
                ChatMessage(**msg_data) for msg_data in session_data["messages"]
            ]

        except Exception as e:
            print(f"Failed to load session {session_id}: {e}")
