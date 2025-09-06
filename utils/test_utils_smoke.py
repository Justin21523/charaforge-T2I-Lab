# test_utils_smoke.py - utils 模組煙霧測試
"""
煙霧測試 - 驗證 utils 模組基本功能
執行: python test_utils_smoke.py
"""

import os
import sys
import time
from pathlib import Path
import tempfile
import json

# 添加 utils 路徑
sys.path.insert(0, str(Path(__file__).parent / "utils"))


def test_shared_cache_setup():
    """測試共享快取設定"""
    print("🔧 測試共享快取設定...")

    try:
        import utils

        cache_info = utils.get_cache_info()

        print(f"  ✅ 快取根目錄: {cache_info['cache_root']}")
        print(f"  ✅ HF 快取: {cache_info['hf_cache']}")
        print(f"  ✅ Torch 快取: {cache_info['torch_cache']}")

        # 驗證目錄存在
        for key, path in cache_info.items():
            if path and Path(path).exists():
                print(f"  ✅ {key} 目錄存在")
            else:
                print(f"  ⚠️  {key} 目錄不存在: {path}")

        return True

    except Exception as e:
        print(f"  ❌ 快取設定失敗: {e}")
        return False


def test_logging_system():
    """測試日誌系統"""
    print("\n📝 測試日誌系統...")

    try:
        from utils.logging import setup_logger, PerformanceLogger, ModuleLogger

        # 基本日誌器
        logger = setup_logger("test_logger")
        logger.info("測試日誌訊息")
        print("  ✅ 基本日誌器正常")

        # 效能日誌器
        perf_logger = PerformanceLogger("test_perf")

        with perf_logger.measure("test_operation"):
            time.sleep(0.1)  # 模擬操作

        stats = perf_logger.get_stats()
        print(f"  ✅ 效能監控: {stats['total_operations']} 個操作")

        # 模組日誌器
        module_logger = ModuleLogger("test_module")
        module_logger.info("模組測試訊息", extra_data={"test": True})
        print("  ✅ 模組日誌器正常")

        return True

    except Exception as e:
        print(f"  ❌ 日誌系統失敗: {e}")
        return False


def test_file_operations():
    """測試檔案操作"""
    print("\n📁 測試檔案操作...")

    try:
        from utils.file_operations import (
            SafeFileHandler,
            safe_json_save,
            safe_json_load,
            get_file_hash,
            validate_path,
        )

        # 建立檔案處理器
        handler = SafeFileHandler()
        print("  ✅ SafeFileHandler 建立成功")

        # 測試 JSON 操作
        test_data = {"test": "data", "timestamp": time.time()}

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # 儲存 JSON
            success = safe_json_save(test_data, tmp_path)
            if success:
                print("  ✅ JSON 儲存成功")

            # 載入 JSON
            loaded_data = safe_json_load(tmp_path)
            if loaded_data and loaded_data["test"] == "data":
                print("  ✅ JSON 載入成功")

            # 計算檔案雜湊
            file_hash = get_file_hash(tmp_path)
            if file_hash:
                print(f"  ✅ 檔案雜湊: {file_hash[:8]}...")

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        # 路徑驗證
        safe_path = handler.validate_path("./test_path")
        print(f"  ✅ 路徑驗證: {safe_path}")

        return True

    except Exception as e:
        print(f"  ❌ 檔案操作失敗: {e}")
        return False


def test_security_system():
    """測試安全性系統"""
    print("\n🔒 測試安全性系統...")

    try:
        from utils.security import (
            TokenManager,
            ContentValidator,
            secure_filename,
            sanitize_input,
            SecurityAudit,
        )

        # Token 管理器
        token_manager = TokenManager()

        # 設定測試 token
        test_token = "hf_test1234567890123456789012345678"
        token_manager.set_token("TEST_TOKEN", test_token)

        # 驗證 token 格式
        validation = token_manager.validate_token_format(
            "HUGGINGFACE_TOKEN", test_token
        )
        print(
            f"  ✅ Token 驗證: {validation['valid']} ({len(validation['issues'])} 問題)"
        )

        # 內容驗證器
        validator = ContentValidator()

        # 測試安全輸入
        safe_input = validator.validate_input("正常的使用者輸入", "general")
        print(f"  ✅ 安全輸入驗證: {safe_input['safe']}")

        # 測試危險輸入
        dangerous_input = validator.validate_input(
            "<script>alert('test')</script>", "general"
        )
        print(f"  ✅ 危險輸入檢測: {not dangerous_input['safe']}")

        # 檔案名稱清理
        clean_filename = secure_filename("test<>file?.txt")
        print(f"  ✅ 檔案名稱清理: {clean_filename}")

        # 安全性稽核
        audit_result = SecurityAudit.check_environment()
        print(f"  ✅ 安全稽核: {len(audit_result['issues'])} 個問題")

        return True

    except Exception as e:
        print(f"  ❌ 安全性系統失敗: {e}")
        return False


def test_calculator_system():
    """測試計算系統"""
    print("\n🧮 測試計算系統...")

    try:
        from utils.calculator import (
            MemoryCalculator,
            PerformanceMonitor,
            get_system_info,
            estimate_vram_usage,
            optimize_batch_size,
        )

        # 系統資訊
        system_info = get_system_info()
        print(
            f"  ✅ 系統資訊: {system_info.total_ram_gb:.1f}GB RAM, {system_info.gpu_count} GPU"
        )

        # 記憶體計算器
        calculator = MemoryCalculator()

        # VRAM 估算
        vram_est = calculator.estimate_model_vram(
            "sd_1_5", precision="fp16", batch_size=1
        )
        if "error" not in vram_est:
            print(f"  ✅ VRAM 估算: SD 1.5 需要 {vram_est['total_vram_gb']:.1f}GB")

        # 批次大小最佳化
        if system_info.gpu_count > 0:
            opt_result = calculator.optimize_batch_size(
                "sd_1_5", system_info.gpu_memory_gb[0]
            )
            print(f"  ✅ 批次最佳化: 建議批次大小 {opt_result['optimal_batch_size']}")

        # 效能監控器
        monitor = PerformanceMonitor()
        monitor.log_operation("test_calc", 0.5, test_param="value")

        stats = monitor.get_performance_stats()
        print(f"  ✅ 效能監控: {stats['total_operations']} 個操作記錄")

        return True

    except Exception as e:
        print(f"  ❌ 計算系統失敗: {e}")
        return False


def test_web_search_system():
    """測試網路搜尋系統"""
    print("\n🌐 測試網路搜尋系統...")

    try:
        from utils.web_search import (
            SearchManager,
            ConnectivityChecker,
            validate_url,
            check_model_availability,
        )

        # 搜尋管理器
        manager = SearchManager()

        # URL 驗證
        test_urls = [
            "https://huggingface.co/runwayml/stable-diffusion-v1-5",
            "https://github.com/huggingface/diffusers",
            "invalid_url",
        ]

        valid_count = 0
        for url in test_urls:
            validation = manager.validate_url(url)
            if validation["valid"]:
                valid_count += 1

        print(f"  ✅ URL 驗證: {valid_count}/{len(test_urls)} 個有效")

        # 連線檢查
        checker = ConnectivityChecker()

        # 網際網路連線 (僅在有網路時測試)
        try:
            internet_ok = checker.check_internet_connection()
            print(f"  ✅ 網際網路連線: {'正常' if internet_ok else '無法連線'}")
        except:
            print("  ⚠️  網際網路連線測試跳過")

        # Hugging Face 存取 (僅在有網路時測試)
        try:
            hf_status = checker.check_huggingface_access()
            print(
                f"  ✅ HuggingFace 存取: {'正常' if hf_status['accessible'] else '無法存取'}"
            )
        except:
            print("  ⚠️  HuggingFace 存取測試跳過")

        return True

    except Exception as e:
        print(f"  ❌ 網路搜尋系統失敗: {e}")
        return False


def test_global_functions():
    """測試全域函數"""
    print("\n🌍 測試全域函數...")

    try:
        from utils import (
            get_token_manager,
            get_performance_monitor,
            get_file_handler,
            log_performance,
            validate_utils_setup,
        )

        # 全域單例測試
        token_mgr1 = get_token_manager()
        token_mgr2 = get_token_manager()

        if token_mgr1 is token_mgr2:
            print("  ✅ TokenManager 單例正常")

        perf_mon1 = get_performance_monitor()
        perf_mon2 = get_performance_monitor()

        if perf_mon1 is perf_mon2:
            print("  ✅ PerformanceMonitor 單例正常")

        file_handler1 = get_file_handler()
        file_handler2 = get_file_handler()

        if file_handler1 is file_handler2:
            print("  ✅ SafeFileHandler 單例正常")

        # 快速效能記錄
        log_performance("test_function", 0.123, param1="value1")
        print("  ✅ 快速效能記錄正常")

        # 工具驗證
        validation = validate_utils_setup()
        print(
            f"  ✅ Utils 設定驗證: {'正常' if validation['modules_loaded'] else '異常'}"
        )

        if validation.get("singletons_ready"):
            print("  ✅ 單例物件準備就緒")

        return True

    except Exception as e:
        print(f"  ❌ 全域函數失敗: {e}")
        return False


def main():
    """主測試函數"""
    print("🚀 SagaForge Utils 模組煙霧測試")
    print("=" * 50)

    tests = [
        ("共享快取設定", test_shared_cache_setup),
        ("日誌系統", test_logging_system),
        ("檔案操作", test_file_operations),
        ("安全性系統", test_security_system),
        ("計算系統", test_calculator_system),
        ("網路搜尋", test_web_search_system),
        ("全域函數", test_global_functions),
    ]

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {name} 測試失敗")
        except Exception as e:
            print(f"❌ {name} 測試異常: {e}")

    print("\n" + "=" * 50)
    print(f"📊 測試結果: {passed}/{total} 通過")

    if passed == total:
        print("🎉 所有測試通過！Utils 模組準備就緒")
        return True
    else:
        print("⚠️  部分測試失敗，請檢查上述錯誤")
        return False


if __name__ == "__main__":
    main()
