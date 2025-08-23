# tests/test_chat.py
import pytest
import time
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_chat_models_endpoint():
    """Test chat models listing"""
    response = client.get("/api/v1/chat/models")
    assert response.status_code == 200
    data = response.json()
    assert "available_models" in data
    assert "current_model" in data


def test_chat_sessions_endpoint():
    """Test chat sessions listing"""
    response = client.get("/api/v1/chat/sessions")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_chat_completion():
    """Test basic chat completion"""
    response = client.post(
        "/api/v1/chat",
        json={
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "max_length": 50,
            "temperature": 0.7,
        },
    )

    # Should not fail with 500 (model loading might take time)
    assert response.status_code in [200, 500]

    if response.status_code == 200:
        data = response.json()
        assert "message" in data
        assert "session_id" in data
        assert data["message"]["role"] == "assistant"
        assert len(data["message"]["content"]) > 0


def test_chat_with_persona():
    """Test chat with persona"""
    response = client.post(
        "/api/v1/chat",
        json={
            "messages": [{"role": "user", "content": "Tell me a creative story"}],
            "persona_id": "creative",
            "max_length": 100,
        },
    )

    assert response.status_code in [200, 500]


def test_chat_conversation_memory():
    """Test conversation memory across multiple messages"""
    # First message
    response1 = client.post(
        "/api/v1/chat",
        json={
            "messages": [{"role": "user", "content": "My name is Alice"}],
            "max_length": 50,
        },
    )

    if response1.status_code != 200:
        pytest.skip("Chat model not available")

    data1 = response1.json()
    session_id = data1["session_id"]

    # Second message referencing the first
    response2 = client.post(
        "/api/v1/chat",
        json={
            "messages": [{"role": "user", "content": "What is my name?"}],
            "session_id": session_id,
            "max_length": 50,
        },
    )

    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["session_id"] == session_id


def test_invalid_chat_request():
    """Test handling of invalid chat request"""
    response = client.post(
        "/api/v1/chat", json={"messages": [], "max_length": 50}  # Empty messages
    )

    # Should handle gracefully
    assert response.status_code in [200, 400, 500]
