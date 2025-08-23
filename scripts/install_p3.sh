# scripts/install_p3.sh
#!/bin/bash
set -e

echo "🚀 Installing CharaForge Multi-Modal Lab P3 (Chat)"

# Check prerequisites
echo "📋 Checking prerequisites..."
python --version
node --version
which conda > /dev/null && echo "✅ Conda found" || echo "❌ Conda not found"

# Set up environment variables
export AI_CACHE_ROOT=${AI_CACHE_ROOT:-"/mnt/ai_warehouse/cache"}
echo "📁 Using cache root: $AI_CACHE_ROOT"

# Create/update conda environment
echo "🐍 Setting up Python environment..."
if conda env list | grep -q "multi-modal-lab"; then
    echo "📦 Updating existing environment..."
    conda activate multi-modal-lab
    pip install -r requirements.txt --upgrade
else
    echo "📦 Creating new environment..."
    conda create -n multi-modal-lab python=3.10 -y
    conda activate multi-modal-lab

    # Install PyTorch
    if command -v nvidia-smi &> /dev/null; then
        echo "🎮 GPU detected, installing CUDA version"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo "💻 No GPU detected, installing CPU version"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi

    # Install other dependencies
    pip install -r requirements.txt
fi

# Install PyQt6 for desktop app
echo "🖥️ Installing PyQt6 for desktop app..."
pip install PyQt6

# Update Node.js dependencies
echo "📦 Updating Node.js packages..."
cd frontend/react_app
npm install
# Add new dependencies for chat functionality
npm install react-textarea-autosize @heroicons/react
cd ../..

# Create config directories and files
echo "⚙️ Setting up configuration files..."
mkdir -p configs

# Copy sample configs if they don't exist
if [ ! -f configs/personas.json ]; then
    echo "📝 Creating personas.json..."
    # The personas.json content would be created here
fi

if [ ! -f configs/safety_rules.yaml ]; then
    echo "📝 Creating safety_rules.yaml..."
    # The safety_rules.yaml content would be created here
fi

# Download chat models (optional)
echo "🤖 Downloading chat models (optional)..."
python -c "
import os
os.environ['AI_CACHE_ROOT'] = '$AI_CACHE_ROOT'
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print('💬 Downloading DialoGPT model...')
    AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
    print('✅ Chat model downloaded')
except Exception as e:
    print(f'⚠️  Chat model download failed: {e}')
    print('   Models will download on first use')
" || echo "⚠️  Model download failed, will download on first use"

echo "✅ P3 Installation complete!"
echo ""
echo "🚀 Start services:"
echo "   Backend: uvicorn backend.main:app --reload"
echo "   React:   cd frontend/react_app && npm run dev"
echo "   PyQt:    cd frontend/pyqt_app && python main.py"
echo ""
echo "📖 API Docs: http://localhost:8000/docs"
