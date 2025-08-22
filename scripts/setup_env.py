# scripts/setup_env.py
"""Helper script to create conda environment"""
import subprocess
import sys
import os


def create_conda_env():
    """Create multi-modal-lab conda environment"""
    env_name = "multi-modal-lab"

    # Check if conda is available
    try:
        subprocess.run(["conda", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Conda not found. Please install Anaconda/Miniconda first.")
        sys.exit(1)

    # Create environment
    print(f"🔧 Creating conda environment: {env_name}")
    subprocess.run(["conda", "create", "-n", env_name, "python=3.10", "-y"], check=True)

    # Install PyTorch (adjust for your CUDA version)
    print("🔧 Installing PyTorch...")
    subprocess.run(
        [
            "conda",
            "run",
            "-n",
            env_name,
            "pip",
            "install",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            "https://download.pytorch.org/whl/cu118",
        ],
        check=True,
    )

    # Install other requirements
    print("🔧 Installing requirements...")
    subprocess.run(
        ["conda", "run", "-n", env_name, "pip", "install", "-r", "requirements.txt"],
        check=True,
    )

    print(f"✅ Environment '{env_name}' created successfully!")
    print(f"🚀 Activate with: conda activate {env_name}")


if __name__ == "__main__":
    create_conda_env()
