#!/bin/bash
# scripts/start_t2i_system.sh - SagaForge T2I System Startup Script

set -e  # Exit on any error

echo "🚀 Starting SagaForge T2I Lab System"
echo "=================================="

# Configuration
API_HOST=${API_HOST:-"0.0.0.0"}
API_PORT=${API_PORT:-8000}
REDIS_URL=${REDIS_URL:-"redis://localhost:6379/0"}
AI_CACHE_ROOT=${AI_CACHE_ROOT:-"../ai_warehouse/cache"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependency() {
    if command -v $1 &> /dev/null; then
        log_success "$1 is available"
        return 0
    else
        log_error "$1 is not installed"
        return 1
    fi
}

check_python_package() {
    if python -c "import $1" 2>/dev/null; then
        log_success "Python package '$1' is available"
        return 0
    else
        log_error "Python package '$1' is not installed"
        return 1
    fi
}

# Step 1: Check dependencies
log_info "Step 1: Checking system dependencies..."

DEPS_OK=true

# Check basic commands
for cmd in python redis-server curl; do
    if ! check_dependency $cmd; then
        DEPS_OK=false
    fi
done

# Check Python packages
for pkg in torch diffusers transformers fastapi uvicorn celery redis; do
    if ! check_python_package $pkg; then
        DEPS_OK=false
    fi
done

if [ "$DEPS_OK" = false ]; then
    log_error "Missing dependencies. Please install them first."
    echo ""
    echo "Install missing packages:"
    echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    echo "  pip install diffusers transformers fastapi uvicorn celery redis"
    echo ""
    echo "Install Redis:"
    echo "  # Ubuntu/Debian: sudo apt install redis-server"
    echo "  # macOS: brew install redis"
    echo "  # Windows: Download from https://redis.io/download"
    exit 1
fi

# Step 2: Setup environment
log_info "Step 2: Setting up environment..."

# Export environment variables
export AI_CACHE_ROOT="$AI_CACHE_ROOT"
export HF_HOME="$AI_CACHE_ROOT/hf"
export TRANSFORMERS_CACHE="$AI_CACHE_ROOT/hf/transformers"
export HF_DATASETS_CACHE="$AI_CACHE_ROOT/hf/datasets"
export HUGGINGFACE_HUB_CACHE="$AI_CACHE_ROOT/hf/hub"
export TORCH_HOME="$AI_CACHE_ROOT/torch"

# Create cache directories
log_info "Creating cache directories..."
mkdir -p "$AI_CACHE_ROOT"/{hf,torch,models,datasets,outputs,runs}
mkdir -p "$AI_CACHE_ROOT"/models/{sd15,sdxl,controlnet,lora,embeddings}
mkdir -p "$AI_CACHE_ROOT"/datasets/{raw,processed,metadata}
mkdir -p "$AI_CACHE_ROOT"/outputs/{images,exports,batch}

log_success "Environment setup completed"
log_info "Cache root: $AI_CACHE_ROOT"

# Step 3: Check GPU availability
log_info "Step 3: Checking GPU availability..."

if python -c "import torch; print('CUDA available:', torch.cuda.is_available())" | grep -q "True"; then
    GPU_INFO=$(python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory//1024**3}GB)')" 2>/dev/null)
    log_success "$GPU_INFO"
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
else
    log_warning "CUDA not available - using CPU mode"
    export CUDA_VISIBLE_DEVICES=""
fi

# Step 4: Start Redis (if not running)
log_info "Step 4: Checking Redis server..."

if ! curl -s $REDIS_URL > /dev/null 2>&1; then
    log_info "Starting Redis server..."

    # Try to start Redis in background
    if command -v redis-server &> /dev/null; then
        redis-server --daemonize yes --bind 127.0.0.1 --port 6379
        sleep 2

        if redis-cli ping > /dev/null 2>&1; then
            log_success "Redis server started"
        else
            log_warning "Failed to start Redis automatically. Please start it manually:"
            echo "  redis-server"
        fi
    else
        log_warning "Redis not found. Please install and start Redis server"
    fi
else
    log_success "Redis server is already running"
fi

# Step 5: Run smoke tests
log_info "Step 5: Running system smoke tests..."

if python scripts/smoke_test_t2i.py --skip-api; then
    log_success "Core system smoke tests passed"
else
    log_warning "Some smoke tests failed. System may have issues."
fi

# Step 6: Start Celery workers (in background)
log_info "Step 6: Starting Celery workers..."

# Kill existing Celery workers
pkill -f "celery worker" 2>/dev/null || true

# Start Celery worker in background
log_info "Starting Celery worker for T2I tasks..."
celery -A workers.celery_app worker \
    --loglevel=info \
    --concurrency=1 \
    --queues=training,generation,batch \
    --detach \
    --pidfile=celery_worker.pid \
    --logfile=logs/celery_worker.log

if [ $? -eq 0 ]; then
    log_success "Celery worker started"
else
    log_warning "Failed to start Celery worker"
fi

# Step 7: Start API server
log_info "Step 7: Starting FastAPI server..."

log_info "API will be available at: http://$API_HOST:$API_PORT"
log_info "API documentation: http://$API_HOST:$API_PORT/docs"

# Create logs directory
mkdir -p logs

# Start FastAPI server
log_success "Starting SagaForge T2I API server..."
exec uvicorn api.main:app \
    --host "$API_HOST" \
    --port "$API_PORT" \
    --reload \
    --log-level info \
    --access-log \
    --log-config logging.yaml

# Note: exec replaces the shell process, so anything after this won't run

# Cleanup function (won't run due to exec above, but kept for reference)
cleanup() {
    log_info "Shutting down services..."

    # Stop Celery workers
    if [ -f celery_worker.pid ]; then
        kill $(cat celery_worker.pid) 2>/dev/null || true
        rm -f celery_worker.pid
    fi

    # Stop Redis if we started it
    # redis-cli shutdown 2>/dev/null || true

    log_success "Cleanup completed"
}

# Set trap for cleanup (won't trigger due to exec)
trap cleanup EXIT