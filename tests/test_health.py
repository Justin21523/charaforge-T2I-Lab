# tests/test_health.py
"""Health endpoint smoke tests"""
import pytest
from fastapi.testclient import TestClient
import os
import tempfile

# Set test cache root before importing app
os.environ["AI_CACHE_ROOT"] = tempfile.mkdtemp()

from backend.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"] == "Multi-Modal Lab API"


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "gpu_available" in data
    assert "gpu_count" in data
    assert "memory_usage" in data
    assert "cache_root" in data

    # Verify memory usage structure
    memory = data["memory_usage"]
    assert "total_gb" in memory
    assert "available_gb" in memory
    assert "usage_percent" in memory


def test_cache_directories_created():
    """Test that cache directories are created properly"""
    from backend.core.cache import setup_shared_cache

    cache_info = setup_shared_cache()

    import pathlib

    cache_root = cache_info["AI_CACHE_ROOT"]

    # Check key directories exist
    expected_dirs = [
        f"{cache_root}/hf/transformers",
        f"{cache_root}/models/blip2",
        f"{cache_root}/datasets/metadata",
        f"{cache_root}/outputs/multi-modal-lab",
    ]

    for dir_path in expected_dirs:
        assert pathlib.Path(dir_path).exists(), f"Directory {dir_path} not created"
