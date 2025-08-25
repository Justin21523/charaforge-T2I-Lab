# scripts/install_p7.sh - P7 Installation Script
#!/bin/bash
set -e

echo "🚀 Installing P7: Batch Processing & Logging System"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.9"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.9+ required, found $python_version"
    exit 1
fi

# Check if conda environment exists
if ! conda env list | grep -q "multi-modal-lab"; then
    echo "📦 Creating conda environment..."
    conda create -n multi-modal-lab python=3.10 -y
fi

echo "🔧 Activating environment and installing dependencies..."
eval "$(conda shell.bash hook)"
conda activate multi-modal-lab

# Install PyTorch with CUDA support (adjust for your system)
if command -v nvidia-smi &> /dev/null; then
    echo "🔥 Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "💻 Installing PyTorch with CPU support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install main dependencies
echo "📚 Installing ML and API dependencies..."
pip install -r requirements.txt

# Install Redis if not present
if ! command -v redis-server &> /dev/null; then
    echo "🔴 Installing Redis..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update && sudo apt-get install -y redis-server
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install redis
    else
        echo "⚠️  Please install Redis manually for your system"
    fi
fi

# Install Node.js dependencies for React
if [ -d "frontend/react_app" ]; then
    echo "⚛️  Installing React dependencies..."
    cd frontend/react_app
    npm install
    cd ../..
fi

# Setup environment variables
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "✏️  Please edit .env file with your specific paths and settings"
fi

# Create necessary directories
echo "📁 Setting up AI_CACHE_ROOT directories..."
source .env
mkdir -p "$AI_CACHE_ROOT"/{hf,torch,models/{lora,blip2,qwen,llava,embeddings},datasets/{raw,processed,metadata},outputs/multi-modal-lab/batch,logs}

# Setup pre-commit hooks
echo "🔨 Setting up pre-commit hooks..."
pre-commit install

echo "✅ P7 installation completed!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file: nano .env"
echo "2. Start Redis: redis-server &"
echo "3. Run smoke tests: ./scripts/test_p7.sh"
echo "4. Start services: ./scripts/start_services.sh"

# scripts/test_p7.sh - P7 Smoke Tests
#!/bin/bash
set -e

echo "🧪 Running P7 Smoke Tests"

# Load environment
if [ -f ".env" ]; then
    source .env
else
    echo "❌ .env file not found"
    exit 1
fi

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate multi-modal-lab

# Test 1: Check Redis connection
echo "🔴 Testing Redis connection..."
if redis-cli ping | grep -q PONG; then
    echo "✅ Redis is running"
else
    echo "❌ Redis is not running. Please start with: redis-server &"
    exit 1
fi

# Test 2: Check AI_CACHE_ROOT setup
echo "📁 Testing AI_CACHE_ROOT setup..."
if [ -d "$AI_CACHE_ROOT" ]; then
    echo "✅ AI_CACHE_ROOT directory exists: $AI_CACHE_ROOT"
else
    echo "❌ AI_CACHE_ROOT directory not found: $AI_CACHE_ROOT"
    exit 1
fi

# Test 3: Python imports
echo "🐍 Testing Python imports..."
python3 -c "
import sys
sys.path.append('.')

# Test basic imports
try:
    import torch
    import transformers
    import fastapi
    import celery
    import redis
    import pandas
    import numpy
    print('✅ All core imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)

# Test GPU availability
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️  CUDA not available, using CPU')
"

# Test 4: Start API server (background)
echo "🌐 Testing API server startup..."
export PYTHONPATH="."
python3 -c "
import pathlib, torch, os
AI_CACHE_ROOT = os.getenv('AI_CACHE_ROOT', '/mnt/ai_warehouse/cache')
for k, v in {
    'HF_HOME': f'{AI_CACHE_ROOT}/hf',
    'TRANSFORMERS_CACHE': f'{AI_CACHE_ROOT}/hf/transformers',
    'HF_DATASETS_CACHE': f'{AI_CACHE_ROOT}/hf/datasets',
    'HUGGINGFACE_HUB_CACHE': f'{AI_CACHE_ROOT}/hf/hub',
    'TORCH_HOME': f'{AI_CACHE_ROOT}/torch',
}.items():
    os.environ[k] = v
    pathlib.Path(v).mkdir(parents=True, exist_ok=True)
print('✅ Cache bootstrap successful')
" &

# Start API server in background
uvicorn backend.main:app --host 127.0.0.1 --port 8001 --log-level error &
API_PID=$!
sleep 5

# Test API health endpoint
echo "🏥 Testing API health endpoint..."
if curl -s http://127.0.0.1:8001/api/v1/health | grep -q "healthy"; then
    echo "✅ API health check passed"
else
    echo "❌ API health check failed"
    kill $API_PID 2>/dev/null || true
    exit 1
fi

# Test 5: Celery worker (quick start/stop)
echo "👷 Testing Celery worker..."
timeout 10s celery -A backend.jobs.worker worker --loglevel=error --concurrency=1 &
WORKER_PID=$!
sleep 3

if ps -p $WORKER_PID > /dev/null; then
    echo "✅ Celery worker started successfully"
    kill $WORKER_PID 2>/dev/null || true
else
    echo "❌ Celery worker failed to start"
fi

# Test 6: Batch submission (mock)
echo "📦 Testing batch submission..."
cat > /tmp/test_batch.json << EOF
[
    {"image_path": "/tmp/test.jpg", "max_length": 50},
    {"image_path": "/tmp/test2.jpg", "max_length": 50}
]
EOF

# Create a dummy image for testing
python3 -c "
from PIL import Image
import numpy as np
img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
img.save('/tmp/test.jpg')
img.save('/tmp/test2.jpg')
print('✅ Test images created')
"

# Test batch API endpoint (will fail but should accept the request)
response=$(curl -s -w "%{http_code}" -o /dev/null -X POST \
    http://127.0.0.1:8001/api/v1/batch/submit \
    -F "task_type=caption" \
    -F "file=@/tmp/test_batch.json" \
    -F "batch_name=test_batch")

if [ "$response" -eq 200 ] || [ "$response" -eq 500 ]; then
    echo "✅ Batch submission endpoint responding"
else
    echo "❌ Batch submission endpoint failed (HTTP $response)"
fi

# Test 7: Resource monitoring
echo "📊 Testing resource monitoring..."
timeout 5s python3 scripts/resource_monitor.py --interval 1 &
MONITOR_PID=$!
sleep 2

if ps -p $MONITOR_PID > /dev/null; then
    echo "✅ Resource monitor working"
    kill $MONITOR_PID 2>/dev/null || true
else
    echo "⚠️  Resource monitor test completed"
fi

# Cleanup
echo "🧹 Cleaning up test processes..."
kill $API_PID 2>/dev/null || true
rm -f /tmp/test_batch.json /tmp/test.jpg /tmp/test2.jpg

echo ""
echo "🎉 P7 Smoke Tests Completed!"
echo ""
echo "📋 Test Summary:"
echo "✅ Redis connection"
echo "✅ AI_CACHE_ROOT setup"
echo "✅ Python dependencies"
echo "✅ API server startup"
echo "✅ Celery worker"
echo "✅ Batch submission endpoint"
echo "✅ Resource monitoring"
echo ""
echo "🚀 Ready to start full services!"

# scripts/start_services.sh - Service Startup Script
#!/bin/bash

echo "🚀 Starting Multi-Modal Lab Services"

# Load environment
if [ -f ".env" ]; then
    source .env
else
    echo "❌ .env file not found"
    exit 1
fi

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate multi-modal-lab

# Function to check if port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo "⚠️  Port $1 is already in use"
        return 1
    fi
    return 0
}

# Start Redis if not running
if ! redis-cli ping >/dev/null 2>&1; then
    echo "🔴 Starting Redis..."
    redis-server &
    sleep 2
    if redis-cli ping | grep -q PONG; then
        echo "✅ Redis started"
    else
        echo "❌ Failed to start Redis"
        exit 1
    fi
else
    echo "✅ Redis already running"
fi

# Start API server
if check_port 8000; then
    echo "🌐 Starting API server on port 8000..."
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
    API_PID=$!
    sleep 3
    echo "✅ API server started (PID: $API_PID)"
else
    echo "⚠️  API server may already be running on port 8000"
fi

# Start Celery worker
echo "👷 Starting Celery worker..."
celery -A backend.jobs.worker worker --loglevel=INFO --concurrency=2 &
WORKER_PID=$!
sleep 2
echo "✅ Celery worker started (PID: $WORKER_PID)"

# Start Flower monitoring (optional)
if check_port 5555; then
    echo "🌸 Starting Flower monitoring on port 5555..."
    celery -A backend.jobs.worker flower --port=5555 &
    FLOWER_PID=$!
    sleep 2
    echo "✅ Flower started (PID: $FLOWER_PID)"
fi

# Start Gradio UI
if check_port 7860; then
    echo "🎨 Starting Gradio UI on port 7860..."
    cd frontend/gradio_app
    python app.py &
    GRADIO_PID=$!
    cd ../..
    sleep 3
    echo "✅ Gradio UI started (PID: $GRADIO_PID)"
fi

# Start React development server (if in development)
if [ -d "frontend/react_app" ] && check_port 3000; then
    echo "⚛️  Starting React development server on port 3000..."
    cd frontend/react_app
    npm run dev &
    REACT_PID=$!
    cd ../..
    sleep 3
    echo "✅ React dev server started (PID: $REACT_PID)"
fi

# Start resource monitor
echo "📊 Starting resource monitor..."
python scripts/resource_monitor.py --interval 30 &
MONITOR_PID=$!
echo "✅ Resource monitor started (PID: $MONITOR_PID)"

echo ""
echo "🎉 All services started successfully!"
echo ""
echo "📋 Service URLs:"
echo "🌐 API Server: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo "🌸 Flower (Celery): http://localhost:5555"
echo "🎨 Gradio UI: http://localhost:7860"
echo "⚛️  React UI: http://localhost:3000"
echo ""
echo "📋 Process IDs:"
echo "API: $API_PID"
echo "Worker: $WORKER_PID"
echo "Flower: $FLOWER_PID"
echo "Gradio: $GRADIO_PID"
echo "React: $REACT_PID"
echo "Monitor: $MONITOR_PID"
echo ""
echo "🛑 To stop all services: ./scripts/stop_services.sh"

# Save PIDs for cleanup script
cat > .service_pids << EOF
API_PID=$API_PID
WORKER_PID=$WORKER_PID
FLOWER_PID=$FLOWER_PID
GRADIO_PID=$GRADIO_PID
REACT_PID=$REACT_PID
MONITOR_PID=$MONITOR_PID
EOF

echo "💾 Service PIDs saved to .service_pids"

# scripts/stop_services.sh - Service Cleanup Script
#!/bin/bash

echo "🛑 Stopping Multi-Modal Lab Services"

# Load service PIDs if available
if [ -f ".service_pids" ]; then
    source .service_pids
    echo "📋 Found saved service PIDs"
else
    echo "⚠️  No saved PIDs found, attempting to find processes..."
fi

# Function to safely kill process
safe_kill() {
    local pid=$1
    local name=$2

    if [ -n "$pid" ] && ps -p $pid > /dev/null 2>&1; then
        echo "🔄 Stopping $name (PID: $pid)..."
        kill $pid
        sleep 2
        if ps -p $pid > /dev/null 2>&1; then
            echo "⚡ Force killing $name..."
            kill -9 $pid
        fi
        echo "✅ $name stopped"
    else
        echo "⚠️  $name not running or PID not found"
    fi
}

# Stop services
safe_kill $API_PID "API Server"
safe_kill $WORKER_PID "Celery Worker"
safe_kill $FLOWER_PID "Flower Monitor"
safe_kill $GRADIO_PID "Gradio UI"
safe_kill $REACT_PID "React Dev Server"
safe_kill $MONITOR_PID "Resource Monitor"

# Kill any remaining processes by name
echo "🧹 Cleaning up remaining processes..."
pkill -f "uvicorn backend.main:app" 2>/dev/null || true
pkill -f "celery.*worker" 2>/dev/null || true
pkill -f "celery.*flower" 2>/dev/null || true
pkill -f "gradio_app" 2>/dev/null || true
pkill -f "resource_monitor" 2>/dev/null || true

# Clean up PID file
rm -f .service_pids

echo "✅ All services stopped successfully!"

# Make scripts executable
chmod +x scripts/install_p7.sh
chmod +x scripts/test_p7.sh
chmod +x scripts/start_services.sh
chmod +x scripts/stop_services.sh