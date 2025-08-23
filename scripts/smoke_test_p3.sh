# scripts/smoke_test_p3.sh
#!/bin/bash
set -e

echo "🔍 Running P3 smoke tests..."

# Test backend health
echo "🏥 Testing backend health..."
curl -f http://localhost:8000/api/v1/health || {
    echo "❌ Backend health check failed"
    exit 1
}

# Test chat models endpoint
echo "🤖 Testing chat models endpoint..."
curl -f http://localhost:8000/api/v1/chat/models || {
    echo "❌ Chat models endpoint failed"
    exit 1
}

# Test chat sessions endpoint
echo "💬 Testing chat sessions endpoint..."
curl -f http://localhost:8000/api/v1/chat/sessions || {
    echo "❌ Chat sessions endpoint failed"
    exit 1
}

# Test simple chat completion
echo "🗨️ Testing chat completion..."
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "max_length": 50,
    "temperature": 0.7
  }' || {
    echo "⚠️  Chat completion test failed (model may not be loaded)"
}

# Test React build
echo "🎨 Testing React build..."
cd frontend/react_app
npm run build
cd ../..

# Test PyQt app (basic import)
echo "🖥️ Testing PyQt app..."
python -c "
try:
    from frontend.pyqt_app.main import MainWindow
    print('✅ PyQt app can be imported')
except ImportError as e:
    print(f'❌ PyQt import failed: {e}')
    print('   Install PyQt6: pip install PyQt6')
except Exception as e:
    print(f'⚠️  PyQt test warning: {e}')
"

# Run pytest
echo "🧪 Running Python tests..."
pytest tests/test_chat.py -v --tb=short

echo "✅ All P3 smoke tests completed!"

# scripts/start_dev_p3.sh
#!/bin/bash
# Development startup script for P3

echo "🚀 Starting CharaForge Multi-Modal Lab P3 Development Environment"

# Check if .env exists
if [ ! -f .env ]; then
    echo "📝 Creating .env from template..."
    cp .env.example .env
    echo "✏️  Please edit .env file with your settings"
fi

# Function to kill background processes on exit
cleanup() {
    echo "🛑 Stopping all services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    echo "👋 Goodbye!"
}
trap cleanup INT TERM EXIT

# Start backend in background
echo "🔧 Starting backend..."
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
for i in {1..30}; do
    if curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
        echo "✅ Backend is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Backend failed to start after 30 seconds"
        exit 1
    fi
    sleep 1
done

# Start React dev server
echo "🎨 Starting React frontend..."
cd frontend/react_app
npm run dev &
FRONTEND_PID=$!
cd ../..

# Wait a bit for frontend to start
sleep 3

echo ""
echo "✅ Development environment started!"
echo ""
echo "📊 Services:"
echo "   🔧 Backend API:  http://localhost:8000"
echo "   🎨 React UI:     http://localhost:3000"
echo "   📖 API Docs:     http://localhost:8000/docs"
echo "   🖥️  PyQt App:     python frontend/pyqt_app/main.py"
echo ""
echo "🧪 Quick Tests:"
echo "   curl http://localhost:8000/api/v1/health"
echo "   curl http://localhost:8000/api/v1/chat/models"
echo ""
echo "💬 Chat Test:"
echo '   curl -X POST http://localhost:8000/api/v1/chat \'
echo '     -H "Content-Type: application/json" \'
echo '     -d '\''{"messages":[{"role":"user","content":"Hello!"}]}'\'''
echo ""
echo "Press Ctrl+C to stop all services"

# Keep script running
wait