# ===== scripts/start_workers.py =====
"""
Workers 啟動腳本
支援不同模式的 Celery worker 啟動
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add project root
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


def start_worker(queue="all", concurrency=1, loglevel="INFO", detach=False):
    """啟動 Celery worker"""

    # Build command
    cmd = [
        sys.executable,
        "-m",
        "celery",
        "-A",
        "workers.celery_app",
        "worker",
        "--loglevel",
        loglevel,
        "--concurrency",
        str(concurrency),
    ]

    if queue != "all":
        cmd.extend(["-Q", queue])

    if detach:
        cmd.append("--detach")

    print(f"🚀 Starting Celery worker: {' '.join(cmd)}")

    try:
        if detach:
            subprocess.Popen(cmd, cwd=ROOT_DIR)
            print("✅ Worker started in background")
        else:
            subprocess.run(cmd, cwd=ROOT_DIR, check=True)

    except KeyboardInterrupt:
        print("\n🛑 Worker stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Worker failed: {e}")
        sys.exit(1)


def start_flower(port=5555):
    """啟動 Flower 監控介面"""

    cmd = [
        sys.executable,
        "-m",
        "celery",
        "-A",
        "workers.celery_app",
        "flower",
        "--port",
        str(port),
    ]

    print(f"🌸 Starting Flower monitoring: {' '.join(cmd)}")
    print(f"📊 Access at: http://localhost:{port}")

    try:
        subprocess.run(cmd, cwd=ROOT_DIR, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Flower stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Flower failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="SagaForge Workers Manager")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Worker command
    worker_parser = subparsers.add_parser("worker", help="Start Celery worker")
    worker_parser.add_argument(
        "-q",
        "--queue",
        default="all",
        choices=["all", "training", "generation", "batch"],
        help="Queue to process",
    )
    worker_parser.add_argument(
        "-c", "--concurrency", type=int, default=1, help="Number of concurrent workers"
    )
    worker_parser.add_argument(
        "-l",
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    worker_parser.add_argument(
        "-d", "--detach", action="store_true", help="Run in background"
    )

    # Flower command
    flower_parser = subparsers.add_parser("flower", help="Start Flower monitoring")
    flower_parser.add_argument(
        "-p", "--port", type=int, default=5555, help="Port for Flower web interface"
    )

    # Test command
    subparsers.add_parser("test", help="Run test suite")

    args = parser.parse_args()

    if args.command == "worker":
        start_worker(
            queue=args.queue,
            concurrency=args.concurrency,
            loglevel=args.loglevel,
            detach=args.detach,
        )
    elif args.command == "flower":
        start_flower(port=args.port)
    elif args.command == "test":
        subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT_DIR, check=True)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
