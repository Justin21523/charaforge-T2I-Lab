#!/usr/bin/env python3
"""
CharaForge T2I Lab Quick Setup Script
自動設置開發環境和依賴套件
"""

import os
import platform
import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True, shell=False):
    """執行命令"""
    if isinstance(cmd, str) and not shell:
        cmd = cmd.split()

    return subprocess.run(cmd, capture_output=True, text=True, check=check, shell=shell)


def check_python_version():
    """檢查 Python 版本"""
    print("🐍 Checking Python version...")

    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        sys.exit(1)

    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")


def detect_system():
    """檢測系統環境"""
    system = platform.system().lower()
    arch = platform.machine().lower()

    print(f"🖥️ System: {system} {arch}")

    # Check if we're in WSL
    is_wsl = "microsoft" in platform.uname().release.lower()
    if is_wsl:
        print("📝 WSL detected")

    return system, arch, is_wsl


def check_gpu():
    """檢查 GPU 可用性"""
    print("🎮 Checking GPU availability...")

    try:
        # Try to import torch and check CUDA
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            return True, gpu_name, gpu_memory
        else:
            print("⚠️ CUDA not available - will use CPU mode")
            return False, None, 0
    except ImportError:
        print("⚠️ PyTorch not installed - GPU check skipped")
        return False, None, 0


def setup_conda_env():
    """設置 Conda 環境"""
    print("🔧 Setting up Conda environment...")

    # Check if conda is available
    try:
        run_command("conda --version")
    except Exception:
        print("❌ Conda not found. Please install Anaconda or Miniconda first.")
        print("   Download from: https://docs.conda.io/en/latest/miniconda.html")
        return False

    env_name = "ai_env"

    # Check if environment already exists
    result = run_command("conda env list", check=False)
    if env_name in result.stdout:
        print(f"📝 Environment '{env_name}' already exists")

        response = input(f"🤔 Recreate environment '{env_name}'? (y/N): ").lower()
        if response == "y":
            print("🗑️ Removing existing environment...")
            run_command(f"conda env remove -n {env_name} -y")
        else:
            print(f"✅ Using existing environment '{env_name}'")
            return True

    # Create new environment
    print(f"🆕 Creating new environment '{env_name}'...")
    run_command(f"conda create -n {env_name} python=3.10 -y")

    print(f"✅ Environment '{env_name}' created")
    print(f"📝 Activate with: conda activate {env_name}")

    return True


def install_pytorch():
    """安裝 PyTorch"""
    print("⚡ Installing PyTorch...")

    system, arch, is_wsl = detect_system()

    # Determine PyTorch installation command
    if system == "darwin":  # macOS
        if "arm" in arch:  # Apple Silicon
            torch_cmd = "pip install torch torchvision torchaudio"
        else:  # Intel Mac
            torch_cmd = "pip install torch torchvision torchaudio"
    else:  # Linux/Windows
        # Try to detect CUDA version
        cuda_version = None
        try:
            result = run_command("nvidia-smi", check=False)
            if result.returncode == 0 and "CUDA Version" in result.stdout:
                # Extract CUDA version (simplified)
                cuda_version = "cu121"  # Default to recent version
        except Exception:
            pass

        if cuda_version:
            torch_cmd = f"pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/{cuda_version}"
        else:
            torch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"

    print(f"📦 Installing: {torch_cmd}")
    run_command(torch_cmd, shell=True)

    # Verify installation
    try:
        import torch

        print(f"✅ PyTorch {torch.__version__} installed")

        if torch.cuda.is_available():
            print(f"✅ CUDA {torch.version.cuda} available")  # type: ignore
        else:
            print("📝 CPU-only PyTorch installed")

        return True
    except ImportError:
        print("❌ PyTorch installation verification failed")
        return False


def install_diffusers_packages():
    """安裝 Diffusers 相關套件"""
    print("🎨 Installing Diffusers and AI packages...")

    packages = [
        "diffusers>=0.25.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "peft>=0.6.0",
        "bitsandbytes",
        "xformers",  # Will fallback gracefully if fails
        "safetensors",
        "controlnet_aux",
    ]

    for package in packages:
        try:
            print(f"📦 Installing {package}...")
            run_command(f"pip install {package}")
            print(f"✅ {package} installed")
        except Exception:
            if package == "xformers":
                print(f"⚠️ {package} installation failed (optional)")
            else:
                print(f"❌ {package} installation failed")


def install_api_packages():
    """安裝 API 相關套件"""
    print("🌐 Installing API and web packages...")

    packages = [
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0",
        "celery>=5.3.0",
        "redis>=5.0.0",
        "python-multipart",
        "aiofiles",
    ]

    for package in packages:
        try:
            print(f"📦 Installing {package}...")
            run_command(f"pip install {package}")
            print(f"✅ {package} installed")
        except Exception:
            print(f"❌ {package} installation failed")


def install_utility_packages():
    """安裝工具套件"""
    print("🔧 Installing utility packages...")

    packages = [
        "pillow>=10.0.0",
        "opencv-python",
        "numpy>=1.24.0",
        "scipy",
        "matplotlib",
        "requests",
        "tqdm",
        "psutil",
        "pyyaml",
        "python-dotenv",
    ]

    for package in packages:
        try:
            print(f"📦 Installing {package}...")
            run_command(f"pip install {package}")
            print(f"✅ {package} installed")
        except Exception:
            print(f"❌ {package} installation failed")


def setup_environment_files():
    """設置環境檔案"""
    print("📝 Setting up environment files...")

    # Setup .env file
    env_file = Path(".env")
    if not env_file.exists():
        env_content = """# CharaForge T2I Lab Environment Configuration

# Cache and Storage
AI_CACHE_ROOT=../ai_warehouse/cache

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# Celery Configuration
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Model Configuration
MODEL_DEFAULT_SD15_MODEL=runwayml/stable-diffusion-v1-5
MODEL_DEFAULT_SDXL_MODEL=stabilityai/stable-diffusion-xl-base-1.0
MODEL_LOW_VRAM_MODE=false
MODEL_USE_FP16=true
MODEL_ENABLE_XFORMERS=true

# Training Configuration
TRAIN_LORA_RANK=16
TRAIN_LEARNING_RATE=0.0001
TRAIN_BATCH_SIZE=1
TRAIN_NUM_EPOCHS=10

# Development
DEBUG=true
LOG_LEVEL=INFO
ENVIRONMENT=development
"""
        env_file.write_text(env_content)
        print("✅ .env file created")
    else:
        print("📝 .env file already exists")

    # Setup basic config directory
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)

    # Create basic app.yaml
    app_config = config_dir / "app.yaml"
    if not app_config.exists():
        config_content = """# CharaForge T2I Lab Configuration

app:
  name: "CharaForge T2I Lab"
  version: "0.2.0"
  description: "Text-to-Image generation and LoRA fine-tuning"

performance:
  enable_memory_optimization: true
  enable_attention_slicing: true
  enable_vae_slicing: true

safety:
  nsfw_detection: true
  content_filtering: true

watermark:
  enable_text_watermark: true
  enable_metadata: true
  text: "Generated by CharaForge T2I"
"""
        app_config.write_text(config_content)
        print("✅ Basic config files created")


def setup_cache_directories():
    """設置快取目錄"""
    print("📁 Setting up cache directories...")

    cache_root = os.getenv("AI_CACHE_ROOT", "../ai_warehouse/cache")
    cache_path = Path(cache_root)

    directories = [
        "hf",
        "torch",
        "models/sd15",
        "models/sdxl",
        "models/controlnet",
        "models/lora",
        "models/embeddings",
        "datasets/raw",
        "datasets/processed",
        "datasets/metadata",
        "outputs/images",
        "outputs/exports",
        "outputs/batch",
        "runs/training",
    ]

    for dir_name in directories:
        dir_path = cache_path / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)

    print(f"✅ Cache directories created at: {cache_path}")

    # Set environment variables
    env_vars = {
        "AI_CACHE_ROOT": str(cache_path),
        "HF_HOME": str(cache_path / "hf"),
        "TRANSFORMERS_CACHE": str(cache_path / "hf" / "transformers"),
        "HF_DATASETS_CACHE": str(cache_path / "hf" / "datasets"),
        "HUGGINGFACE_HUB_CACHE": str(cache_path / "hf" / "hub"),
        "TORCH_HOME": str(cache_path / "torch"),
    }

    for key, value in env_vars.items():
        os.environ[key] = value

    print("✅ Environment variables set")


def install_redis():
    """安裝或檢查 Redis"""
    print("🔴 Setting up Redis...")

    # Check if Redis is already available
    try:
        run_command("redis-server --version", check=False)
        print("✅ Redis is already installed")
        return True
    except Exception:
        pass

    system, _, _ = detect_system()

    if system == "darwin":  # macOS
        print("📦 Installing Redis via Homebrew...")
        try:
            run_command("brew install redis")
            print("✅ Redis installed via Homebrew")
            return True
        except Exception:
            print("❌ Failed to install Redis via Homebrew")

    elif system == "linux":  # Linux
        print("📦 Installing Redis via apt...")
        try:
            run_command("sudo apt update", shell=True)
            run_command("sudo apt install -y redis-server", shell=True)
            print("✅ Redis installed via apt")
            return True
        except Exception:
            print("❌ Failed to install Redis via apt")

    else:  # Windows
        print("⚠️ Windows detected - Redis installation requires manual setup")
        print("   Please download Redis from: https://redis.io/download")
        print("   Or use Windows Subsystem for Linux (WSL)")

    return False


def run_smoke_test():
    """執行煙霧測試"""
    print("🧪 Running smoke test...")

    try:
        # Run the smoke test
        result = run_command(
            [sys.executable, "scripts/smoke_test_t2i.py", "--skip-api"], check=False
        )

        if result.returncode == 0:
            print("✅ Smoke test passed")
            return True
        else:
            print("⚠️ Some smoke tests failed")
            print("   This is normal for first-time setup")
            return False

    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        return False


def create_startup_scripts():
    """創建啟動腳本"""
    print("📜 Creating startup scripts...")

    # Make sure scripts are executable
    scripts_dir = Path("scripts")

    if (scripts_dir / "start_t2i_system.sh").exists():
        run_command("chmod +x scripts/start_t2i_system.sh", shell=True, check=False)

    if (scripts_dir / "test_api.sh").exists():
        run_command("chmod +x scripts/test_api.sh", shell=True, check=False)

    # Create simple Python startup script
    startup_script = scripts_dir / "start_system.py"
    startup_content = '''#!/usr/bin/env python3
"""Simple Python startup script for CharaForge T2I"""

import subprocess
import sys
from pathlib import Path

def main():
    print("🚀 Starting CharaForge T2I Lab...")

    # Start the system using uvicorn
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\\n👋 CharaForge T2I Lab stopped")

if __name__ == "__main__":
    main()
'''

    startup_script.write_text(startup_content)
    run_command(f"chmod +x {startup_script}", shell=True, check=False)

    print("✅ Startup scripts ready")


def main():
    """主函數"""
    print("🚀 CharaForge T2I Lab Quick Setup")
    print("=" * 50)

    # Check Python version
    check_python_version()

    # Detect system
    system, arch, is_wsl = detect_system()

    # Setup steps
    steps = [
        ("Setting up cache directories", setup_cache_directories),
        ("Installing PyTorch", install_pytorch),
        ("Installing Diffusers packages", install_diffusers_packages),
        ("Installing API packages", install_api_packages),
        ("Installing utility packages", install_utility_packages),
        ("Setting up environment files", setup_environment_files),
        ("Installing Redis", install_redis),
        ("Creating startup scripts", create_startup_scripts),
    ]

    success_count = 0

    for step_name, step_func in steps:
        print(f"\n📋 {step_name}...")
        try:
            if step_func():
                success_count += 1
                print(f"✅ {step_name} completed")
            else:
                print(f"⚠️ {step_name} completed with warnings")
                success_count += 0.5
        except Exception as e:
            print(f"❌ {step_name} failed: {e}")

    # Check GPU after PyTorch installation
    print("\n🎮 Final GPU check...")
    has_gpu, gpu_name, gpu_memory = check_gpu()

    # Run smoke test
    print("\n🧪 Running smoke test...")
    smoke_test_passed = run_smoke_test()

    # Final report
    print("\n" + "=" * 50)
    print("🎯 Setup Complete!")
    print("=" * 50)

    print(f"✅ Steps completed: {int(success_count)}/{len(steps)}")

    if has_gpu:
        print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("💻 CPU-only mode")

    if smoke_test_passed:
        print("🧪 Smoke test: ✅ PASSED")
    else:
        print("🧪 Smoke test: ⚠️ Some issues (normal for first setup)")

    print("\n📋 Next Steps:")
    print("1. Activate environment: conda activate ai_env")
    print("2. Start API: bash scripts/start_api.sh")
    print("3. Start worker: bash scripts/start_worker.sh")
    print("4. Access API docs: http://localhost:8000/docs")
    print("5. Run tests: bash scripts/test_api.sh")

    print("\n📚 Documentation:")
    print("- API docs: http://localhost:8000/docs")
    print("- Health check: http://localhost:8000/api/v1/health")
    print("- Models: http://localhost:8000/api/v1/models")

    if success_count >= len(steps) * 0.8:
        print("\n🎉 Setup successful! Ready to use CharaForge T2I Lab.")
    else:
        print("\n⚠️ Setup completed with some issues. Check error messages above.")


if __name__ == "__main__":
    main()
