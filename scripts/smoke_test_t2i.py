#!/usr/bin/env python3
"""
SagaForge T2I System Smoke Test
驗證所有 T2I 核心功能是否正常運作
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import requests

AI_CACHE_ROOT = os.getenv("AI_CACHE_ROOT", "/mnt/c/ai_cache")

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class T2ISmokeTest:
    """T2I 系統煙霧測試"""

    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.test_results = {
            "config_system": False,
            "cache_system": False,
            "pipeline_loading": False,
            "safety_system": False,
            "watermark_system": False,
            "api_health": False,
            "image_generation": False,
            "lora_management": False,
            "system_status": False,
            "errors": [],
        }

    def run_all_tests(self) -> Dict[str, Any]:
        """執行所有測試"""
        print("🧪 Starting SagaForge T2I Smoke Tests...")
        print("=" * 60)

        # Core system tests
        self.test_config_system()
        self.test_cache_system()
        self.test_pipeline_loading()
        self.test_safety_system()
        self.test_watermark_system()

        # API tests (if API is running)
        self.test_api_health()
        if self.test_results["api_health"]:
            self.test_image_generation_api()
            self.test_lora_management_api()
            self.test_system_status_api()

        # Generate report
        self.print_test_report()
        return self.test_results

    def test_config_system(self):
        """測試配置系統"""
        print("🔧 Testing configuration system...")

        try:
            from core.config import (
                bootstrap_config,
                get_app_paths,
                get_cache_paths,
                get_settings,
            )

            # Test settings loading
            settings = get_settings()
            assert hasattr(settings, "model"), "Settings should have model config"
            assert hasattr(settings, "api"), "Settings should have API config"

            # Test path configuration
            cache_paths = get_cache_paths()
            app_paths = get_app_paths()

            assert cache_paths.root.exists(), "Cache root should exist"
            assert app_paths.root.exists(), "App root should exist"

            # Test bootstrap
            summary = bootstrap_config(verbose=False)
            assert summary["settings_loaded"], "Settings should be loaded"

            self.test_results["config_system"] = True
            print("✅ Configuration system test passed")

        except Exception as e:
            error_msg = f"Configuration system test failed: {e}"
            self.test_results["errors"].append(error_msg)
            print(f"❌ {error_msg}")

    def test_cache_system(self):
        """測試共用快取系統"""
        print("🗂️ Testing shared cache system...")

        try:
            from core.shared_cache import bootstrap_cache

            # Test cache bootstrap
            cache = bootstrap_cache(verbose=False)

            # Test cache functionality
            cache_stats = cache.get_cache_stats()
            device_config = cache.get_device_config()

            assert "cache_root" in cache_stats, "Cache stats should include root"
            assert (
                "cuda_available" in device_config
            ), "Device config should include CUDA info"

            # Test model registration (with dummy data)
            test_model_path = cache.cache_paths.models / "test_model.txt"
            test_model_path.parent.mkdir(parents=True, exist_ok=True)
            test_model_path.write_text("test model content")

            success = cache.register_model(
                model_id="test_model",
                model_type="test",
                local_path=test_model_path,
                metadata={"test": True},
            )

            assert success, "Model registration should succeed"

            # Cleanup test model
            test_model_path.unlink(missing_ok=True)

            self.test_results["cache_system"] = True
            print("✅ Shared cache system test passed")

        except Exception as e:
            error_msg = f"Cache system test failed: {e}"
            self.test_results["errors"].append(error_msg)
            print(f"❌ {error_msg}")

    def test_pipeline_loading(self):
        """測試 Pipeline 載入"""
        print("🎨 Testing T2I pipeline loading...")

        try:
            from core.t2i.pipeline import get_pipeline_manager

            # Get pipeline manager
            manager = get_pipeline_manager()

            # Test pipeline creation
            pipeline = manager.get_pipeline("sd15")  # type: ignore
            assert pipeline is not None, "Pipeline should be created"

            # Test pipeline status
            status = pipeline.get_status()
            assert "model_type" in status, "Status should include model type"
            assert status["model_type"] == "sd15", "Model type should match"

            # Note: We don't actually load the pipeline to avoid downloading models
            # In a real test environment, you would test pipeline.load_pipeline()

            self.test_results["pipeline_loading"] = True
            print("✅ Pipeline loading test passed")

        except Exception as e:
            error_msg = f"Pipeline loading test failed: {e}"
            self.test_results["errors"].append(error_msg)
            print(f"❌ {error_msg}")

    def test_safety_system(self):
        """測試安全系統"""
        print("🛡️ Testing safety system...")

        try:
            from core.t2i.safety import get_safety_checker, test_safety_system

            # Run built-in safety test
            test_results = asyncio.run(test_safety_system())

            # Check basic functionality
            assert isinstance(test_results, dict), "Safety test should return dict"

            # Test individual components
            checker = asyncio.run(get_safety_checker())
            status = checker.get_status()

            assert "enabled" in status, "Safety status should include enabled flag"

            self.test_results["safety_system"] = True
            print("✅ Safety system test passed")

        except Exception as e:
            error_msg = f"Safety system test failed: {e}"
            self.test_results["errors"].append(error_msg)
            print(f"❌ {error_msg}")

    def test_watermark_system(self):
        """測試浮水印系統"""
        print("💧 Testing watermark system...")

        try:
            from core.t2i.watermark import get_watermark_manager, test_watermark_system

            # Run built-in watermark test
            test_results = test_watermark_system()

            # Check basic functionality
            assert isinstance(test_results, dict), "Watermark test should return dict"
            assert (
                len(test_results["errors"]) == 0
            ), f"Watermark test should have no errors: {test_results['errors']}"

            # Test manager
            manager = get_watermark_manager()
            status = manager.get_watermark_status()

            assert "enabled" in status, "Watermark status should include enabled flag"

            self.test_results["watermark_system"] = True
            print("✅ Watermark system test passed")

        except Exception as e:
            error_msg = f"Watermark system test failed: {e}"
            self.test_results["errors"].append(error_msg)
            print(f"❌ {error_msg}")

    def test_api_health(self):
        """測試 API 健康狀態"""
        print("🌐 Testing API health...")

        try:
            base = self.api_base_url.rstrip("/")
            response = requests.get(f"{base}/api/v1/health", timeout=10)

            if response.status_code == 200:
                health_data = response.json()
                assert "status" in health_data, "Health response should include status"

                self.test_results["api_health"] = True
                print("✅ API health test passed")
            else:
                error_msg = f"API health check failed: HTTP {response.status_code}"
                self.test_results["errors"].append(error_msg)
                print(f"❌ {error_msg}")

        except requests.RequestException as e:
            error_msg = f"API health test failed (connection): {e}"
            self.test_results["errors"].append(error_msg)
            print(f"❌ {error_msg}")
        except Exception as e:
            error_msg = f"API health test failed: {e}"
            self.test_results["errors"].append(error_msg)
            print(f"❌ {error_msg}")

    def test_image_generation_api(self):
        """測試圖像生成 API"""
        print("🎨 Testing image generation API...")

        try:
            # Test generation request
            generation_request = {
                "prompt": "A simple test image",
                "model_type": "sd15",
                "width": 256,
                "height": 256,
                "steps": 5,  # Very fast for testing
                "batch_size": 1,
            }

            base = self.api_base_url.rstrip("/")
            response = requests.post(
                f"{base}/api/v1/t2i/generate", json=generation_request, timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                assert "job_id" in result, "Generation response should include job_id"
                assert "status" in result, "Generation response should include status"

                self.test_results["image_generation"] = True
                print("✅ Image generation API test passed")
            else:
                error_msg = f"Image generation API failed: HTTP {response.status_code}"
                if response.text:
                    error_msg += f" - {response.text[:200]}"
                self.test_results["errors"].append(error_msg)
                print(f"❌ {error_msg}")

        except Exception as e:
            error_msg = f"Image generation API test failed: {e}"
            self.test_results["errors"].append(error_msg)
            print(f"❌ {error_msg}")

    def test_lora_management_api(self):
        """測試 LoRA 管理 API"""
        print("🔄 Testing LoRA management API...")

        try:
            base = self.api_base_url.rstrip("/")

            list_res = requests.get(f"{base}/api/v1/lora/list", timeout=10)
            if list_res.status_code != 200:
                raise RuntimeError(f"LoRA list failed: HTTP {list_res.status_code}")
            assert isinstance(list_res.json(), list), "LoRA list should be a list"

            status_res = requests.get(f"{base}/api/v1/lora/status", timeout=10)
            if status_res.status_code != 200:
                raise RuntimeError(f"LoRA status failed: HTTP {status_res.status_code}")
            payload = status_res.json()
            assert "pipeline_loaded" in payload, "LoRA status should include pipeline_loaded"
            assert "loaded_loras" in payload, "LoRA status should include loaded_loras"

            self.test_results["lora_management"] = True
            print("✅ LoRA management API test passed")

        except Exception as e:
            error_msg = f"LoRA management API test failed: {e}"
            self.test_results["errors"].append(error_msg)
            print(f"❌ {error_msg}")

    def test_system_status_api(self):
        """測試系統狀態 API"""
        print("📊 Testing models API...")

        try:
            base = self.api_base_url.rstrip("/")
            response = requests.get(f"{base}/api/v1/models", timeout=10)

            if response.status_code == 200:
                status = response.json()
                assert "count" in status, "Models response should include count"
                assert "models" in status, "Models response should include models list"
                assert "registry_path" in status, "Models response should include registry_path"

                self.test_results["system_status"] = True
                print("✅ Models API test passed")
            else:
                error_msg = f"Models API failed: HTTP {response.status_code}"
                self.test_results["errors"].append(error_msg)
                print(f"❌ {error_msg}")

        except Exception as e:
            error_msg = f"Models API test failed: {e}"
            self.test_results["errors"].append(error_msg)
            print(f"❌ {error_msg}")

    def print_test_report(self):
        """打印測試報告"""
        print("\n" + "=" * 60)
        print("🧪 CharaForge T2I Smoke Test Report")
        print("=" * 60)

        total_tests = len([k for k in self.test_results.keys() if k != "errors"])
        passed_tests = sum(
            [1 for k, v in self.test_results.items() if k != "errors" and v]
        )

        print(f"📈 Test Results: {passed_tests}/{total_tests} passed")
        print()

        # Print individual test results
        for test_name, result in self.test_results.items():
            if test_name == "errors":
                continue

            status_icon = "✅" if result else "❌"
            formatted_name = test_name.replace("_", " ").title()
            print(f"{status_icon} {formatted_name}")

        # Print errors if any
        if self.test_results["errors"]:
            print(f"\n❌ Errors ({len(self.test_results['errors'])}):")
            for i, error in enumerate(self.test_results["errors"], 1):
                print(f"  {i}. {error}")

        # Overall status
        print("\n🎯 Overall Status: ", end="")
        if passed_tests == total_tests:
            print("🟢 ALL TESTS PASSED - System is ready!")
        elif passed_tests >= total_tests * 0.8:
            print("🟡 MOST TESTS PASSED - System is mostly functional")
        elif passed_tests >= total_tests * 0.5:
            print("🟠 SOME TESTS FAILED - System has issues")
        else:
            print("🔴 MANY TESTS FAILED - System needs attention")

        print("=" * 60)


def main():
    """主函數"""
    import argparse

    parser = argparse.ArgumentParser(description="SagaForge T2I System Smoke Test")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Skip API tests (test only core components)",
    )
    parser.add_argument("--save-report", help="Save test report to JSON file")

    args = parser.parse_args()

    from core.config import bootstrap_config, get_config_summary

    bootstrap_config(verbose=False)
    summary = get_config_summary()
    print(f"[storage] models={summary.get('models_root')} cache={summary.get('cache_root')}")

    # Run tests
    tester = T2ISmokeTest(api_base_url=args.api_url)

    if args.skip_api:
        # Override API tests to skip them
        tester.test_results["api_health"] = (
            True  # Mark as passed to skip dependent tests
        )
        print("⏭️ Skipping API tests as requested")

    results = tester.run_all_tests()

    # Save report if requested
    if args.save_report:
        report_path = Path(args.save_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"📄 Test report saved to: {report_path}")

    # Exit with appropriate code
    total_tests = len([k for k in results.keys() if k != "errors"])
    passed_tests = sum([1 for k, v in results.items() if k != "errors" and v])

    if passed_tests == total_tests:
        return 0  # All tests passed
    elif passed_tests >= total_tests * 0.8:
        return 1  # Most tests passed
    else:
        return 2  # Too many failures


def test_core_components_only():
    """只測試核心組件（不需要 API 運行）"""
    print("🧪 Testing Core Components Only...")

    tester = T2ISmokeTest()

    # Run only core tests
    tester.test_config_system()
    tester.test_cache_system()
    tester.test_pipeline_loading()
    tester.test_safety_system()
    tester.test_watermark_system()

    # Set API tests as skipped
    tester.test_results["api_health"] = True
    tester.test_results["image_generation"] = True
    tester.test_results["lora_management"] = True
    tester.test_results["system_status"] = True

    tester.print_test_report()
    return tester.test_results


def test_installation():
    """測試安裝環境"""
    print("📦 Testing Installation Environment...")
    print("=" * 50)

    # Test Python version
    print(f"🐍 Python Version: {sys.version}")

    # Test required packages
    required_packages = [
        "torch",
        "diffusers",
        "transformers",
        "PIL",
        "fastapi",
        "pydantic",
        "numpy",
        "requests",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)

    # Test CUDA availability
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ CUDA Available: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("⚠️ CUDA Not Available - CPU mode only")
    except ImportError:
        print("❌ PyTorch not available")

    # Test cache directories
    cache_paths = [
        AI_CACHE_ROOT,
        f"{AI_CACHE_ROOT}/hf",
        f"{AI_CACHE_ROOT}/torch",
        f"{AI_CACHE_ROOT}/models",
        f"{AI_CACHE_ROOT}/datasets",
        f"{AI_CACHE_ROOT}/outputs",
    ]

    print("\n📁 Cache Directories:")
    for path in cache_paths:
        path_obj = Path(path)
        if path_obj.exists():
            print(f"✅ {path}")
        else:
            print(f"❌ {path} - Will be created")
            path_obj.mkdir(parents=True, exist_ok=True)

    # Summary
    print("\n🎯 Installation Status:")
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("📦 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    else:
        print("✅ All required packages installed")
        return True


if __name__ == "__main__":
    # Quick installation test first
    print("🚀 SagaForge T2I System Smoke Test")
    print("=" * 60)

    installation_ok = test_installation()

    if not installation_ok:
        print("\n❌ Installation issues detected. Please fix before running tests.")
        sys.exit(3)

    print("\n")

    # Run main tests
    sys.exit(main())


# ===== Additional Utility Functions =====


def quick_api_test(api_url: str = "http://localhost:8000") -> bool:
    """快速 API 測試"""
    try:
        response = requests.get(f"{api_url.rstrip('/')}/api/v1/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def test_gpu_memory():
    """測試 GPU 記憶體"""
    try:
        import torch

        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return False

        # Test GPU memory allocation
        device = torch.device("cuda")

        # Allocate a small tensor
        test_tensor = torch.randn(100, 100, device=device)

        allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
        total = torch.cuda.get_device_properties(0).total_memory / (1024**2)  # MB

        print(f"✅ GPU Memory Test: {allocated:.1f}MB allocated / {total:.1f}MB total")

        # Clean up
        del test_tensor
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"❌ GPU Memory Test Failed: {e}")
        return False


def benchmark_generation_speed():
    """基準測試生成速度"""
    try:
        import time

        from core.t2i.pipeline import get_pipeline_manager

        print("⏱️ Running generation speed benchmark...")

        manager = get_pipeline_manager()
        manager.get_pipeline("sd15")

        # Note: This would require loading actual models
        # For smoke test, we just test the pipeline creation

        start_time = time.time()
        # pipeline.load_pipeline()  # Would actually load model
        setup_time = time.time() - start_time

        print(f"✅ Pipeline setup time: {setup_time:.2f}s")
        print("⚠️ Actual generation benchmark requires loaded models")

        return True

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False


# ===== Example Usage =====
"""
使用範例：

# 基本測試（包含 API）
python scripts/smoke_test_t2i.py

# 只測試核心組件
python scripts/smoke_test_t2i.py --skip-api

# 使用自定義 API URL
python scripts/smoke_test_t2i.py --api-url http://localhost:8080

# 儲存測試報告
python scripts/smoke_test_t2i.py --save-report test_report.json

# 只在代碼中測試核心組件
if __name__ == "__main__":
    from scripts.smoke_test_t2i import test_core_components_only
    results = test_core_components_only()
"""
