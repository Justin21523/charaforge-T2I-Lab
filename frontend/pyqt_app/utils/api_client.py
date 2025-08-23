# frontend/pyqt_app/utils/api_client.py
import requests
import json
from typing import Dict, List, Optional
import os


class APIClient:
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv(
            "API_BASE_URL", "http://localhost:8000/api/v1"
        )
        self.session = requests.Session()

    def health_check(self) -> Dict:
        """Check API health"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def chat_completion(
        self,
        messages: List[Dict],
        session_id: Optional[str] = None,
        persona_id: Optional[str] = None,
        max_length: int = 200,
        temperature: float = 0.7,
    ) -> Dict:
        """Send chat completion request"""
        try:
            payload = {
                "messages": messages,
                "max_length": max_length,
                "temperature": temperature,
                "session_id": session_id,
                "persona_id": persona_id,
            }

            response = self.session.post(
                f"{self.base_url}/chat", json=payload, timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def list_chat_models(self) -> Dict:
        """List available chat models"""
        try:
            response = self.session.get(f"{self.base_url}/chat/models")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def get_session_history(self, session_id: str) -> Dict:
        """Get chat session history"""
        try:
            response = self.session.get(
                f"{self.base_url}/chat/sessions/{session_id}/history"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
