# ===== scripts/test_workers.py =====
"""
Workers 功能測試腳本
測試 Celery 任務執行、GPU 管理、進度追蹤
"""

import os
import sys
import time
import json
import asyncio
from pathlib import Path

# Add project root
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.shared_cache import bootstrap_cache
from core.config import bootstrap_config
from workers.utils.gpu_manager import GPUManager
from workers.utils.job_tracker import JobTracker, JobStatus
from workers.utils.task_progress import TaskProgress


def test_gpu_manager():
    """測試 GPU 管理器"""
    print("🔧 Testing GPU Manager...")

    gpu_manager = GPUManager(low_vram_mode=True)

    print(f"Device: {gpu_manager.device}")
    print(f"Initial memory: {gpu_manager.initial_memory}")

    # Test inference context
    with gpu_manager.inference_context() as device:
        print(f"✅ Inference context ready: {device}")
        time.sleep(1)

    # Test training context
    with gpu_manager.training_context() as device:
        print(f"✅ Training context ready: {device}")
        time.sleep(1)

    print("✅ GPU Manager test completed")


def test_job_tracker():
    """測試工作追蹤器"""
    print("📊 Testing Job Tracker...")

    try:
        tracker = JobTracker()

        # Create test job
        job_id = "test_job_123"
        job_info = tracker.create_job(job_id, "test_task", {"param": "value"})
        print(f"Created job: {job_info.job_id}")

        # Update job progress
        tracker.update_job(
            job_id, status=JobStatus.PROGRESS, progress=50, message="Half done"
        )

        # Get job info
        updated_job = tracker.get_job(job_id)
        print(f"Job status: {updated_job.status}, Progress: {updated_job.progress}%")

        # Complete job
        tracker.update_job(
            job_id,
            status=JobStatus.SUCCESS,
            progress=100,
            result={"output": "test_result"},
        )

        # List jobs
        jobs = tracker.list_jobs(limit=10)
        print(f"Found {len(jobs)} jobs")

        print("✅ Job Tracker test completed")

    except Exception as e:
        print(f"❌ Job Tracker test failed: {e}")


def test_celery_tasks():
    """測試 Celery 任務"""
    print("⚙️ Testing Celery Tasks...")

    try:
        # Import tasks (this will also test imports)
        from workers.tasks.training import train_lora
        from workers.tasks.generation import generate_image_task

        print("✅ Task imports successful")

        # Test configuration
        from workers.celery_app import celery_app

        print(f"Celery app configured: {celery_app.main}")
        print(f"Broker: {celery_app.conf.broker_url}")
        print(f"Backend: {celery_app.conf.result_backend}")

        print("✅ Celery configuration test completed")

    except Exception as e:
        print(f"❌ Celery test failed: {e}")
        import traceback

        traceback.print_exc()


def test_mock_training():
    """測試模擬訓練流程"""
    print("🎯 Testing Mock Training Flow...")

    try:
        # Mock task progress (simulating Celery task)
        class MockTask:
            def __init__(self):
                self.request = type("obj", (object,), {"id": "mock_task_123"})

            def update_state(self, state, meta):
                print(f"Task update: {state} - {meta.get('message', '')}")

        mock_task = MockTask()
        progress = TaskProgress(mock_task, total_steps=100)

        # Simulate training steps
        progress.update(10, "Loading dataset...")
        time.sleep(0.5)

        progress.update(30, "Initializing model...")
        time.sleep(0.5)

        progress.update(50, "Training started...")
        time.sleep(1)

        progress.update(80, "Training completed...")
        time.sleep(0.5)

        progress.update(95, "Saving model...")
        time.sleep(0.5)

        result = progress.complete({"model_path": "/path/to/model", "loss": 0.023})
        print(f"Training completed: {result}")

        print("✅ Mock training test completed")

    except Exception as e:
        print(f"❌ Mock training test failed: {e}")


def test_worker_startup():
    """測試 Worker 啟動流程"""
    print("🚀 Testing Worker Startup...")

    try:
        # Bootstrap configuration
        print("Bootstrapping config...")
        bootstrap_config(verbose=True)

        print("Bootstrapping cache...")
        bootstrap_cache(verbose=True)

        print("✅ Worker startup test completed")

    except Exception as e:
        print(f"❌ Worker startup test failed: {e}")
        import traceback

        traceback.print_exc()


def main():
    """執行所有測試"""
    print("🧪 SagaForge Workers Test Suite")
    print("=" * 50)

    tests = [
        test_worker_startup,
        test_gpu_manager,
        test_job_tracker,
        test_celery_tasks,
        test_mock_training,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            print(f"\n{test_func.__name__}")
            print("-" * 30)
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed} passed, {failed} failed")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
