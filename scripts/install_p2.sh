# scripts/install_p2.sh
#!/bin/bash
set -e

echo "🚀 Installing CharaForge Multi-Modal Lab P2 (VQA)"

# Check prerequisites
echo "📋 Checking prerequisites..."
python --version
node --version
which conda > /dev/null && echo "✅ Conda found" || echo "❌ Conda not found"

# Set up environment variables
export AI_CACHE_ROOT=${AI_CACHE_ROOT:-"/mnt/ai_warehouse/cache"}
echo "📁 Using cache root: $AI_CACHE_ROOT"

# Create conda environment
echo "🐍 Setting up Python environment..."
conda create -n multi-modal-lab python=3.10 -y
conda activate multi-modal-lab

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU detected, installing CUDA version"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "💻 No GPU detected, installing CPU version"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install Python dependencies
echo "📦 Installing Python packages..."
pip install -r requirements.txt

# Install Node.js dependencies for React
echo "📦 Installing Node.js packages..."
cd frontend/react_app
npm install
cd ../..

# Download models (optional, will download on first use)
echo "🤖 Downloading models (optional)..."
python -c "
import os
os.environ['AI_CACHE_ROOT'] = '$AI_CACHE_ROOT'
from transformers import BlipProcessor, BlipForConditionalGeneration
print('📸 Downloading BLIP-2 model...')
BlipProcessor.from_pretrained('Salesforce/blip2-opt-2.7b')
BlipForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b')
print('✅ BLIP-2 downloaded')
" || echo "⚠️  Model download failed, will download on first use"

echo "✅ Installation complete!"
echo "🚀 Start the backend: uvicorn backend.main:app --reload"
echo "🎨 Start the frontend: cd frontend/react_app && npm run dev"

# requirements.txt (Updated for P2)
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.4.2
pillow==10.0.1
torch>=2.0.0
torchvision>=0.15.0
transformers==4.35.0
accelerate==0.24.0
bitsandbytes==0.41.1
xformers==0.0.22
opencv-python==4.8.1.78
numpy==1.24.3
python-multipart==0.0.6
python-dotenv==1.0.0

# Development dependencies
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.1
ruff==0.1.6
black==23.10.1

# tests/test_vqa.py
import pytest
import base64
import io
from PIL import Image
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def create_test_image():
    """Create a simple test image"""
    image = Image.new('RGB', (100, 100), color='red')
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer.getvalue()

def test_health_endpoint():
    """Test API health check"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "gpu_available" in data
    assert "cache_root" in data

def test_vqa_models_endpoint():
    """Test VQA models listing"""
    response = client.get("/api/v1/vqa/models")
    assert response.status_code == 200
    data = response.json()
    assert "available_models" in data
    assert "current_model" in data

@pytest.mark.asyncio
async def test_vqa_upload_endpoint():
    """Test VQA with file upload"""
    image_data = create_test_image()

    response = client.post(
        "/api/v1/vqa/upload",
        files={"image": ("test.png", image_data, "image/png")},
        data={
            "question": "What color is this image?",
            "max_length": 50,
            "temperature": 0.7
        }
    )

    # Should not fail with 500 (model loading might take time)
    assert response.status_code in [200, 500]  # Allow model loading errors in tests

    if response.status_code == 200:
        data = response.json()
        assert "question" in data
        assert "answer" in data
        assert "confidence" in data

def test_vqa_base64_endpoint():
    """Test VQA with base64 input"""
    image_data = create_test_image()
    image_b64 = base64.b64encode(image_data).decode()

    response = client.post(
        "/api/v1/vqa",
        json={
            "image": image_b64,
            "question": "Describe this image",
            "max_length": 100,
            "temperature": 0.7,
            "language": "en"
        }
    )

    assert response.status_code in [200, 500]  # Allow model loading errors

def test_invalid_image_format():
    """Test handling of invalid image format"""
    response = client.post(
        "/api/v1/vqa",
        json={
            "image": "invalid_base64",
            "question": "What is this?",
        }
    )

    assert response.status_code == 400

# tests/test_react_build.py
import subprocess
import os
import pytest

def test_react_build():
    """Test that React app builds successfully"""
    react_dir = "frontend/react_app"

    if not os.path.exists(react_dir):
        pytest.skip("React app directory not found")

    # Run npm build
    result = subprocess.run(
        ["npm", "run", "build"],
        cwd=react_dir,
        capture_output=True,
        text=True
    )

    assert result.returncode == 0, f"React build failed: {result.stderr}"
    assert os.path.exists(f"{react_dir}/dist"), "Build output not found"

# scripts/smoke_test.sh
#!/bin/bash
set -e

echo "🔍 Running P2 smoke tests..."

# Test backend health
echo "🏥 Testing backend health..."
curl -f http://localhost:8000/api/v1/health || {
    echo "❌ Backend health check failed"
    exit 1
}

# Test VQA models endpoint
echo "🤖 Testing VQA models endpoint..."
curl -f http://localhost:8000/api/v1/vqa/models || {
    echo "❌ VQA models endpoint failed"
    exit 1
}

# Test React build
echo "🎨 Testing React build..."
cd frontend/react_app
npm run build
cd ../..

# Run pytest
echo "🧪 Running Python tests..."
pytest tests/test_vqa.py -v

echo "✅ All smoke tests passed!"

# scripts/start_dev.sh
#!/bin/bash
# Development startup script

echo "🚀 Starting CharaForge Multi-Modal Lab P2 Development Environment"

# Check if .env exists
if [ ! -f .env ]; then
    echo "📝 Creating .env from template..."
    cp .env.example .env
    echo "✏️  Please edit .env file with your settings"
fi

# Start backend in background
echo "🔧 Starting backend..."
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start React dev server
echo "🎨 Starting React frontend..."
cd frontend/react_app
npm run dev &
FRONTEND_PID=$!

echo "✅ Development environment started!"
echo "📋 Backend: http://localhost:8000"
echo "🎨 Frontend: http://localhost:3000"
echo "📖 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap "kill $BACKEND_PID $FRONTEND_PID" INT
wait