# scripts/setup.py - 環境設置腳本

import os
import sys
from pathlib import Path
import subprocess


def check_python_version():
    """檢查 Python 版本"""
    if sys.version_info < (3, 10):
        print("錯誤: 需要 Python 3.10 或更高版本")
        sys.exit(1)
    print(f"✓ Python 版本: {sys.version}")


def setup_environment():
    """設置環境變數和目錄"""
    print("設置環境...")

    # 設置 AI_CACHE_ROOT
    cache_root = os.getenv("AI_CACHE_ROOT", "../ai_warehouse/cache")
    cache_path = Path(cache_root).absolute()

    print(f"創建快取目錄: {cache_path}")

    # 創建必要目錄
    directories = [
        cache_path,
        cache_path / "models" / "sd",
        cache_path / "models" / "sdxl",
        cache_path / "models" / "controlnet",
        cache_path / "models" / "lora",
        cache_path / "models" / "ipadapter",
        cache_path / "datasets" / "raw",
        cache_path / "datasets" / "processed",
        cache_path / "datasets" / "metadata",
        cache_path / "cache" / "hf",
        cache_path / "cache" / "torch",
        cache_path / "runs",
        cache_path / "outputs" / "t2i",
        cache_path / "outputs" / "batch",
        cache_path / "outputs" / "exports",
        Path("logs"),
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
AI_CACHE_ROOT={cache_path}
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
        print(f"\n請安裝缺少的套件:")
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
    gpu_ok = check_gpu()

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
