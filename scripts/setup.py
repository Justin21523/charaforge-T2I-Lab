# scripts/setup.py - 環境設置腳本

import os
import sys
from pathlib import Path


def check_python_version():
    """檢查 Python 版本"""
    if sys.version_info < (3, 10):
        print("錯誤: 需要 Python 3.10 或更高版本")
        sys.exit(1)
    print(f"✓ Python 版本: {sys.version}")


def setup_environment():
    """設置環境變數和目錄"""
    print("設置環境...")

    # AI_WAREHOUSE 3.0 defaults (see ~/Desktop/data_model_structure.md)
    project_slug = os.getenv("PROJECT_SLUG", "charaforge-t2i-lab")
    models_root = Path(os.getenv("AI_MODELS_ROOT", "/mnt/c/ai_models")).absolute()
    cache_root = Path(os.getenv("AI_CACHE_ROOT", "/mnt/c/ai_cache")).absolute()
    datasets_root = Path(os.getenv("AI_DATASETS_ROOT", "/mnt/data/datasets")).absolute()
    training_root = Path(os.getenv("AI_TRAINING_ROOT", "/mnt/data/training")).absolute()

    # Framework caches (force out of $HOME)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))

    project_datasets = datasets_root / project_slug
    project_runs = training_root / "runs" / project_slug

    print(f"Models:   {models_root}")
    print(f"Caches:   {cache_root}")
    print(f"Datasets: {project_datasets}")
    print(f"Runs:     {project_runs}")

    directories = [
        # /mnt/c (models + caches)
        models_root / "stable-diffusion" / "sd15",
        models_root / "stable-diffusion" / "sdxl",
        models_root / "controlnet",
        models_root / "lora",
        models_root / "lora_sdxl",
        models_root / "embeddings",
        cache_root / "huggingface",
        cache_root / "torch",
        cache_root / "pip",
        # /mnt/data (datasets + outputs)
        project_datasets / "raw",
        project_datasets / "processed",
        training_root / "logs" / project_slug,
        project_runs / "outputs" / "generation",
        project_runs / "outputs" / "batch",
        project_runs / "outputs" / "exports",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")

    # 創建 .env 檔案 (如果不存在)
    env_file = Path(".env")
    if not env_file.exists():
        print("創建 .env 檔案...")
        with open(env_file, "w") as f:
            f.write(
                f"""# CharaForge T2I Lab 環境設定
PROJECT_SLUG={project_slug}
AI_MODELS_ROOT={models_root}
AI_CACHE_ROOT={cache_root}
XDG_CACHE_HOME={cache_root}
HF_HOME={cache_root / 'huggingface'}
TRANSFORMERS_CACHE={cache_root / 'huggingface'}
TORCH_HOME={cache_root / 'torch'}
AI_DATASETS_ROOT={datasets_root}
AI_TRAINING_ROOT={training_root}
CUDA_VISIBLE_DEVICES=0
API_HOST=0.0.0.0
API_PORT=8000
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
LOG_LEVEL=INFO
"""
            )
        print(f"  ✓ {env_file}")


def check_dependencies():
    """檢查依賴套件"""
    print("檢查 Python 依賴...")

    required_packages = [
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "accelerate",
        "peft",
        "fastapi",
        "uvicorn",
        "celery",
        "redis",
        "pillow",
        "numpy",
        "pandas",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (缺少)")
            missing_packages.append(package)

    if missing_packages:
        print("\n請安裝缺少的套件:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    return True


def check_gpu():
    """檢查 GPU 可用性"""
    print("檢查 GPU...")

    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
            print(f"  ✓ GPU: {gpu_name} ({gpu_memory}GB VRAM)")
            return True
        else:
            print("  ⚠ GPU 不可用，將使用 CPU 模式")
            return False
    except ImportError:
        print("  ✗ PyTorch 未安裝")
        return False


def check_redis():
    """檢查 Redis 連接"""
    print("檢查 Redis...")

    try:
        import redis

        client = redis.Redis(host="localhost", port=6379, db=0)
        client.ping()
        print("  ✓ Redis 連接成功")
        return True
    except Exception as e:
        print(f"  ✗ Redis 連接失敗: {e}")
        print("  請啟動 Redis: redis-server")
        return False


def main():
    """主設置流程"""
    print("=== CharaForge T2I Lab 環境設置 ===\n")

    # 檢查 Python 版本
    check_python_version()

    # 設置環境
    setup_environment()

    # 檢查依賴
    deps_ok = check_dependencies()

    # 檢查 GPU
    check_gpu()

    # 檢查 Redis
    redis_ok = check_redis()

    print("\n=== 設置完成 ===")

    if not deps_ok:
        print("⚠ 請先安裝缺少的 Python 套件")
        return False

    if not redis_ok:
        print("⚠ 請啟動 Redis 服務")

    print("\n啟動指令:")
    print("  API 服務器: bash scripts/start_api.sh")
    print("  Worker: bash scripts/start_worker.sh")
    print("  測試 API: bash scripts/test_api.sh")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
