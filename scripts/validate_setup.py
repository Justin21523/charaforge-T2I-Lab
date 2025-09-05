#!/usr/bin/env python3
# scripts/validate_setup.py - CharaForge T2I Lab Setup Validation
"""
CharaForge T2I Lab 環境設置驗證腳本
檢查所有必要的依賴、配置和系統需求
"""

import sys
import os
import subprocess
import importlib
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple


# 顏色代碼
class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_colored(text: str, color: str = Colors.WHITE):
    """打印彩色文字"""
    print(f"{color}{text}{Colors.END}")


def print_section(title: str):
    """打印區段標題"""
    print_colored(f"\n{'='*60}", Colors.BLUE)
    print_colored(f"🔍 {title}", Colors.BOLD + Colors.BLUE)
    print_colored("=" * 60, Colors.BLUE)


def check_python_version() -> Tuple[bool, str]:
    """檢查 Python 版本"""
    version = sys.version_info

    if version.major == 3 and version.minor >= 8:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    else:
        return (
            False,
            f"Python {version.major}.{version.minor}.{version.micro} (需要 3.8+)",
        )


def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """檢查 Python 套件"""
    if import_name is None:
        import_name = package_name

    try:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", "unknown")
        return True, f"{package_name} {version}"
    except ImportError:
        return False, f"{package_name} 未安裝"


def check_system_command(command: str) -> Tuple[bool, str]:
    """檢查系統命令"""
    try:
        result = subprocess.run(
            [command, "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            version_line = result.stdout.split("\n")[0]
            return True, version_line
        else:
            return False, f"{command} 不可用"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, f"{command} 未找到"


def check_redis_connection() -> Tuple[bool, str]:
    """檢查 Redis 連線"""
    try:
        result = subprocess.run(
            ["redis-cli", "ping"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and "PONG" in result.stdout:
            return True, "Redis 連線正常"
        else:
            return False, "Redis 無法連線"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, "redis-cli 未找到"


def check_gpu_availability() -> Tuple[bool, str]:
    """檢查 GPU 可用性"""
    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return True, f"{gpu_count} GPU(s) - {gpu_name} ({memory_gb:.1f}GB)"
        else:
            return False, "CUDA 不可用"
    except ImportError:
        return False, "PyTorch 未安裝"


def check_directory_structure() -> Tuple[bool, List[str]]:
    """檢查目錄結構"""
    required_dirs = [
        "api",
        "api/routers",
        "core",
        "core/t2i",
        "core/train",
        "workers",
        "workers/tasks",
        "configs",
        "scripts",
    ]

    missing_dirs = []

    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)

    return len(missing_dirs) == 0, missing_dirs


def check_config_files() -> Tuple[bool, List[str]]:
    """檢查配置檔案"""
    required_files = [
        ".env.example",
        "requirements.txt",
        "configs/app.yaml",
        "README.md",
    ]

    missing_files = []

    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    return len(missing_files) == 0, missing_files


def check_cache_setup() -> Tuple[bool, str]:
    """檢查共用快取設置"""
    try:
        # 嘗試導入並測試核心模組
        sys.path.insert(0, str(Path.cwd()))

        from core.config import get_cache_paths, validate_cache_setup

        validation = validate_cache_setup()
        status = validation.get("status", "unknown")

        if status == "healthy":
            cache_paths = get_cache_paths()
            return True, f"快取根目錄: {cache_paths.root}"
        else:
            return False, f"快取狀態: {status}"

    except Exception as e:
        return False, f"快取檢查失敗: {e}"


def check_core_modules() -> Tuple[bool, List[str]]:
    """檢查核心模組"""
    core_modules = [
        ("core.config", "配置管理"),
        ("core.shared_cache", "共用快取"),
        ("core.performance", "效能監控"),
        ("core.exceptions", "例外處理"),
        ("api.main", "API 主程式"),
        ("workers.celery_app", "Celery 應用"),
    ]

    failed_modules = []

    try:
        sys.path.insert(0, str(Path.cwd()))

        for module_name, description in core_modules:
            try:
                importlib.import_module(module_name)
            except ImportError as e:
                failed_modules.append(f"{description} ({module_name}): {e}")
    except Exception as e:
        failed_modules.append(f"模組檢查失敗: {e}")

    return len(failed_modules) == 0, failed_modules


def generate_recommendations(validation_results: Dict[str, Any]) -> List[str]:
    """生成改善建議"""
    recommendations = []

    # Python 版本建議
    if not validation_results["python"]["status"]:
        recommendations.append("升級 Python 到 3.8 或更高版本")

    # 必要套件建議
    failed_packages = [
        pkg
        for pkg, result in validation_results["packages"].items()
        if not result["status"]
    ]
    if failed_packages:
        recommendations.append(
            f"安裝缺失的套件: pip install {' '.join(failed_packages)}"
        )

    # Redis 建議
    if not validation_results["redis"]["status"]:
        recommendations.extend(
            [
                "安裝並啟動 Redis:",
                "  - Ubuntu/Debian: sudo apt install redis-server && sudo systemctl start redis",
                "  - macOS: brew install redis && brew services start redis",
                "  - Docker: docker run -d --name redis -p 6379:6379 redis:alpine",
            ]
        )

    # GPU 建議
    if not validation_results["gpu"]["status"]:
        recommendations.extend(
            [
                "GPU 不可用，將使用 CPU 模式 (較慢)",
                "若要啟用 GPU:",
                "  - 安裝 NVIDIA 驅動程式",
                "  - 安裝 CUDA Toolkit",
                "  - 安裝 PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121",
            ]
        )

    # 目錄結構建議
    if not validation_results["directories"]["status"]:
        recommendations.append("執行快速設置腳本: python scripts/quick_setup.py")

    # 配置檔案建議
    if not validation_results["config_files"]["status"]:
        recommendations.extend(
            ["建立缺失的配置檔案", "複製環境變數範本: cp .env.example .env"]
        )

    # 快取設置建議
    if not validation_results["cache"]["status"]:
        recommendations.extend(
            ["設定共用快取目錄", "確保 AI_CACHE_ROOT 環境變數正確設定"]
        )

    return recommendations


def main():
    """主驗證函數"""
    print_colored("🔍 CharaForge T2I Lab 環境設置驗證", Colors.BOLD + Colors.CYAN)
    print_colored("檢查所有必要的依賴、配置和系統需求\n", Colors.CYAN)

    validation_results = {}

    # 1. Python 版本檢查
    print_section("Python 環境")
    python_ok, python_info = check_python_version()
    validation_results["python"] = {"status": python_ok, "info": python_info}

    if python_ok:
        print_colored(f"✅ {python_info}", Colors.GREEN)
    else:
        print_colored(f"❌ {python_info}", Colors.RED)

    # 2. 必要套件檢查
    print_section("Python 套件依賴")
    required_packages = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "pydantic": "pydantic",
        "torch": "torch",
        "diffusers": "diffusers",
        "transformers": "transformers",
        "celery": "celery",
        "redis": "redis",
        "pandas": "pandas",
        "numpy": "numpy",
        "pillow": "PIL",
        "requests": "requests",
        "pyyaml": "yaml",
    }

    package_results = {}

    for package, import_name in required_packages.items():
        pkg_ok, pkg_info = check_package(package, import_name)
        package_results[package] = {"status": pkg_ok, "info": pkg_info}

        if pkg_ok:
            print_colored(f"✅ {pkg_info}", Colors.GREEN)
        else:
            print_colored(f"❌ {pkg_info}", Colors.RED)

    validation_results["packages"] = package_results

    # 3. 系統命令檢查
    print_section("系統工具")
    system_commands = ["git", "curl"]

    for command in system_commands:
        cmd_ok, cmd_info = check_system_command(command)

        if cmd_ok:
            print_colored(f"✅ {cmd_info}", Colors.GREEN)
        else:
            print_colored(f"❌ {cmd_info}", Colors.YELLOW)

    # 4. Redis 連線檢查
    print_section("Redis 資料庫")
    redis_ok, redis_info = check_redis_connection()
    validation_results["redis"] = {"status": redis_ok, "info": redis_info}

    if redis_ok:
        print_colored(f"✅ {redis_info}", Colors.GREEN)
    else:
        print_colored(f"❌ {redis_info}", Colors.RED)

    # 5. GPU 可用性檢查
    print_section("GPU 支援")
    gpu_ok, gpu_info = check_gpu_availability()
    validation_results["gpu"] = {"status": gpu_ok, "info": gpu_info}

    if gpu_ok:
        print_colored(f"✅ {gpu_info}", Colors.GREEN)
    else:
        print_colored(f"⚠️  {gpu_info}", Colors.YELLOW)

    # 6. 目錄結構檢查
    print_section("專案目錄結構")
    dirs_ok, missing_dirs = check_directory_structure()
    validation_results["directories"] = {"status": dirs_ok, "missing": missing_dirs}

    if dirs_ok:
        print_colored("✅ 所有必要目錄存在", Colors.GREEN)
    else:
        print_colored("❌ 缺失目錄:", Colors.RED)
        for dir_path in missing_dirs:
            print_colored(f"   - {dir_path}", Colors.RED)

    # 7. 配置檔案檢查
    print_section("配置檔案")
    config_ok, missing_files = check_config_files()
    validation_results["config_files"] = {"status": config_ok, "missing": missing_files}

    if config_ok:
        print_colored("✅ 所有配置檔案存在", Colors.GREEN)
    else:
        print_colored("❌ 缺失檔案:", Colors.RED)
        for file_path in missing_files:
            print_colored(f"   - {file_path}", Colors.RED)

    # 8. 共用快取設置檢查
    print_section("共用快取設置")
    cache_ok, cache_info = check_cache_setup()
    validation_results["cache"] = {"status": cache_ok, "info": cache_info}

    if cache_ok:
        print_colored(f"✅ {cache_info}", Colors.GREEN)
    else:
        print_colored(f"❌ {cache_info}", Colors.RED)

    # 9. 核心模組檢查
    print_section("核心模組")
    modules_ok, failed_modules = check_core_modules()
    validation_results["core_modules"] = {
        "status": modules_ok,
        "failed": failed_modules,
    }

    if modules_ok:
        print_colored("✅ 所有核心模組可正常導入", Colors.GREEN)
    else:
        print_colored("❌ 模組導入失敗:", Colors.RED)
        for module_error in failed_modules:
            print_colored(f"   - {module_error}", Colors.RED)

    # 10. 整體結果摘要
    print_section("驗證結果摘要")

    # 計算總分
    critical_checks = ["python", "redis", "directories", "config_files", "cache"]
    optional_checks = ["gpu"]

    critical_passed = sum(
        1
        for check in critical_checks
        if validation_results.get(check, {}).get("status", False)
    )
    total_critical = len(critical_checks)

    package_passed = sum(
        1
        for pkg_result in validation_results["packages"].values()
        if pkg_result["status"]
    )
    total_packages = len(validation_results["packages"])

    print_colored(
        f"關鍵檢查: {critical_passed}/{total_critical}",
        Colors.GREEN if critical_passed == total_critical else Colors.YELLOW,
    )
    print_colored(
        f"套件依賴: {package_passed}/{total_packages}",
        Colors.GREEN if package_passed >= total_packages * 0.8 else Colors.YELLOW,
    )

    # 判斷整體狀態
    if critical_passed == total_critical and package_passed >= total_packages * 0.8:
        overall_status = "READY"
        status_color = Colors.GREEN
        print_colored("🎉 環境設置完成，可以啟動 CharaForge T2I Lab！", Colors.GREEN)
    elif critical_passed >= total_critical * 0.8:
        overall_status = "PARTIAL"
        status_color = Colors.YELLOW
        print_colored("⚠️  環境基本可用，但有些問題需要解決", Colors.YELLOW)
    else:
        overall_status = "NOT_READY"
        status_color = Colors.RED
        print_colored("❌ 環境設置不完整，需要修正關鍵問題", Colors.RED)

    # 生成建議
    recommendations = generate_recommendations(validation_results)

    if recommendations:
        print_section("改善建議")
        for i, rec in enumerate(recommendations, 1):
            print_colored(f"{i}. {rec}", Colors.CYAN)

    # 快速修復建議
    print_section("快速修復")
    print_colored("執行以下命令來修復常見問題:", Colors.BLUE)
    print_colored("1. python scripts/quick_setup.py  # 建立專案結構", Colors.WHITE)
    print_colored("2. pip install -r requirements.txt  # 安裝依賴", Colors.WHITE)
    print_colored("3. cp .env.example .env  # 複製環境變數", Colors.WHITE)
    print_colored("4. redis-server  # 啟動 Redis (另一個終端)", Colors.WHITE)
    print_colored("5. python smoke_tests.py  # 執行測試", Colors.WHITE)

    # 儲存驗證結果
    results_file = Path("validation_results.json")
    try:
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": "2024-01-01T00:00:00Z",  # 實際應用中使用真實時間
                    "overall_status": overall_status,
                    "results": validation_results,
                    "recommendations": recommendations,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print_colored(f"\n📄 詳細結果已儲存至: {results_file}", Colors.BLUE)
    except Exception as e:
        print_colored(f"\n⚠️  無法儲存結果檔案: {e}", Colors.YELLOW)

    # 返回適當的退出碼
    if overall_status == "READY":
        return 0
    elif overall_status == "PARTIAL":
        return 1
    else:
        return 2


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print_colored("\n\n⚠️  驗證被用戶中斷", Colors.YELLOW)
        sys.exit(130)
    except Exception as e:
        print_colored(f"\n❌ 驗證過程發生錯誤: {e}", Colors.RED)
        sys.exit(1)
