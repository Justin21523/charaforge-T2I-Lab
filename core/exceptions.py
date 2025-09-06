# core/exceptions.py - Unified Exception Handling
"""
統一例外處理系統
為 CharaForge T2I Lab 提供標準化的錯誤處理和例外類別
"""

import logging
import traceback
from typing import Any, Optional, Dict, List
from collections import Counter
from functools import wraps

logger = logging.getLogger(__name__)


class CharaForgeError(Exception):
    """CharaForge T2I Lab 基礎例外類別"""

    def __init__(
        self,
        message: str,
        error_code: str = "CHARAFORGE_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式（用於 API 回應）"""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
        }

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"


# ===== 配置相關例外 =====


class ConfigError(CharaForgeError):
    """配置相關錯誤"""

    def __init__(
        self, message: str, config_key: str = "", error_code: str = "CONFIG_ERROR"
    ):
        super().__init__(message, error_code, {"config_key": config_key})


class CacheError(CharaForgeError):
    """快取相關錯誤"""

    def __init__(
        self, message: str, cache_type: str = "", error_code: str = "CACHE_ERROR"
    ):
        super().__init__(message, error_code, {"cache_type": cache_type})


# ===== 模型相關例外 =====


class ModelError(CharaForgeError):
    """模型相關錯誤"""

    def __init__(
        self, message: str, model_name: str = "", error_code: str = "MODEL_ERROR"
    ):
        super().__init__(message, error_code, {"model_name": model_name})


class ModelNotFoundError(ModelError):
    """模型未找到錯誤"""

    def __init__(self, model_name: str):
        super().__init__(
            f"Model not found: {model_name}", model_name, "MODEL_NOT_FOUND"
        )


class ModelLoadError(ModelError):
    """模型載入失敗"""

    def __init__(self, model_name: str, reason: str = ""):
        message = f"Failed to load model: {model_name}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, model_name, "MODEL_LOAD_ERROR")


class CUDAOutOfMemoryError(ModelError):
    """CUDA 記憶體不足錯誤"""

    def __init__(self, model_name: str = "", allocated_gb: float = 0):
        message = f"CUDA out of memory"
        if model_name:
            message += f" when loading {model_name}"
        if allocated_gb > 0:
            message += f" (allocated: {allocated_gb:.1f}GB)"

        super().__init__(message, model_name, "CUDA_OOM")
        self.details["allocated_gb"] = allocated_gb


# ===== T2I 相關例外 =====


class T2IError(CharaForgeError):
    """Text-to-Image 錯誤"""

    def __init__(self, message: str, error_code: str = "T2I_ERROR"):
        super().__init__(message, error_code)


class PipelineError(T2IError):
    """管線錯誤"""

    def __init__(
        self, message: str, pipeline_type: str = "", error_code: str = "PIPELINE_ERROR"
    ):
        super().__init__(message, error_code)
        self.details["pipeline_type"] = pipeline_type


class GenerationError(T2IError):
    """圖像生成錯誤"""

    def __init__(
        self, message: str, prompt: str = "", error_code: str = "GENERATION_ERROR"
    ):
        super().__init__(message, error_code)
        self.details["prompt"] = prompt[:100] if prompt else ""  # 限制長度


class LoRAError(T2IError):
    """LoRA 相關錯誤"""

    def __init__(self, message: str, lora_id: str = "", error_code: str = "LORA_ERROR"):
        super().__init__(message, error_code)
        self.details["lora_id"] = lora_id


class LoRANotFoundError(LoRAError):
    """LoRA 模型未找到"""

    def __init__(self, lora_id: str):
        super().__init__(f"LoRA not found: {lora_id}", lora_id, "LORA_NOT_FOUND")


class LoRALoadError(LoRAError):
    """LoRA 載入失敗"""

    def __init__(self, lora_id: str, reason: str = ""):
        message = f"Failed to load LoRA: {lora_id}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, lora_id, "LORA_LOAD_ERROR")


class ControlNetError(T2IError):
    """ControlNet 錯誤"""

    def __init__(
        self,
        message: str,
        controlnet_type: str = "",
        error_code: str = "CONTROLNET_ERROR",
    ):
        super().__init__(message, error_code)
        self.details["controlnet_type"] = controlnet_type


# ===== 訓練相關例外 =====


class TrainingError(CharaForgeError):
    """訓練相關錯誤"""

    def __init__(
        self, message: str, run_id: str = "", error_code: str = "TRAINING_ERROR"
    ):
        super().__init__(message, error_code, {"run_id": run_id})


class DatasetError(TrainingError):
    """資料集錯誤"""

    def __init__(
        self, message: str, dataset_name: str = "", error_code: str = "DATASET_ERROR"
    ):
        super().__init__(message, "", error_code)
        self.details["dataset_name"] = dataset_name


class DatasetNotFoundError(DatasetError):
    """資料集未找到"""

    def __init__(self, dataset_name: str):
        super().__init__(
            f"Dataset not found: {dataset_name}", dataset_name, "DATASET_NOT_FOUND"
        )


class TrainingJobError(TrainingError):
    """訓練任務錯誤"""

    def __init__(
        self, message: str, job_id: str = "", error_code: str = "TRAINING_JOB_ERROR"
    ):
        super().__init__(message, "", error_code)
        self.details["job_id"] = job_id


# ===== 安全與內容相關例外 =====


class SafetyError(CharaForgeError):
    """安全檢查錯誤"""

    def __init__(
        self, message: str, content_type: str = "", error_code: str = "SAFETY_ERROR"
    ):
        super().__init__(message, error_code, {"content_type": content_type})


class ContentBlockedError(SafetyError):
    """內容被安全過濾器阻擋"""

    def __init__(self, reason: str, content_type: str = ""):
        super().__init__(f"Content blocked: {reason}", content_type, "CONTENT_BLOCKED")


class NSFWContentError(SafetyError):
    """NSFW 內容錯誤"""

    def __init__(self, confidence: float = 0.0):
        message = f"NSFW content detected"
        if confidence > 0:
            message += f" (confidence: {confidence:.2f})"
        super().__init__(message, "image", "NSFW_CONTENT")
        self.details["confidence"] = confidence


# ===== 輸入驗證例外 =====


class ValidationError(CharaForgeError):
    """輸入驗證錯誤"""

    def __init__(self, field: str, value: Any, reason: str = ""):
        message = f"Validation failed for field '{field}'"
        if reason:
            message += f": {reason}"

        super().__init__(
            message,
            "VALIDATION_ERROR",
            {"field": field, "value": str(value)[:100], "reason": reason},  # 限制長度
        )


class InvalidParameterError(ValidationError):
    """無效參數錯誤"""

    def __init__(self, parameter: str, value: Any, expected: str = ""):
        reason = f"Expected {expected}" if expected else "Invalid value"
        super().__init__(parameter, value, reason)
        self.error_code = "INVALID_PARAMETER"


# ===== 資源相關例外 =====


class ResourceError(CharaForgeError):
    """資源管理錯誤"""

    def __init__(
        self, resource_type: str, message: str, error_code: str = "RESOURCE_ERROR"
    ):
        super().__init__(
            f"{resource_type}: {message}", error_code, {"resource_type": resource_type}
        )


class StorageError(ResourceError):
    """儲存相關錯誤"""

    def __init__(self, message: str, path: str = ""):
        super().__init__("Storage", message, "STORAGE_ERROR")
        if path:
            self.details["path"] = path


class DiskSpaceError(StorageError):
    """磁碟空間不足錯誤"""

    def __init__(self, required_gb: float = 0, available_gb: float = 0):
        message = "Insufficient disk space"
        if required_gb > 0 and available_gb > 0:
            message += (
                f" (required: {required_gb:.1f}GB, available: {available_gb:.1f}GB)"
            )

        super().__init__(message)
        self.error_code = "DISK_SPACE_ERROR"
        self.details.update({"required_gb": required_gb, "available_gb": available_gb})


# ===== API 相關例外 =====


class APIError(CharaForgeError):
    """API 相關錯誤"""

    def __init__(
        self, message: str, status_code: int = 500, error_code: str = "API_ERROR"
    ):
        super().__init__(message, error_code, {"status_code": status_code})
        self.status_code = status_code


class RateLimitError(APIError):
    """請求頻率限制錯誤"""

    def __init__(self, retry_after: int = 60):
        super().__init__(
            f"Rate limit exceeded. Try again in {retry_after} seconds.",
            429,
            "RATE_LIMIT_ERROR",
        )
        self.details["retry_after"] = retry_after


class JobNotFoundError(APIError):
    """任務未找到錯誤"""

    def __init__(self, job_id: str):
        super().__init__(f"Job not found: {job_id}", 404, "JOB_NOT_FOUND")
        self.details["job_id"] = job_id


# ===== 錯誤處理裝飾器 =====


def handle_errors(log_errors: bool = True, reraise: bool = True):
    """錯誤處理裝飾器"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except CharaForgeError:
                # 已知的 CharaForge 錯誤，直接重新拋出
                if reraise:
                    raise
            except Exception as e:
                # 未知錯誤，包裝為 CharaForgeError
                if log_errors:
                    logger.error(f"Unexpected error in {func.__name__}: {e}")
                    logger.error(traceback.format_exc())

                if reraise:
                    # 包裝為 CharaForgeError
                    wrapped_error = CharaForgeError(
                        f"Unexpected error in {func.__name__}: {str(e)}",
                        "UNEXPECTED_ERROR",
                        {"original_error": str(e), "function": func.__name__},
                    )
                    raise wrapped_error from e

        return wrapper

    return decorator


def handle_cuda_oom(func):
    """CUDA OOM 專用處理裝飾器"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                # 嘗試取得已分配記憶體資訊
                allocated_gb = 0
                try:
                    import torch

                    if torch.cuda.is_available():
                        allocated_gb = torch.cuda.memory_allocated() / (1024**3)
                except ImportError:
                    pass

                raise CUDAOutOfMemoryError("", allocated_gb) from e
            else:
                raise

    return wrapper


def handle_model_loading(func):
    """模型載入專用處理裝飾器"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            raise ModelNotFoundError(str(e)) from e
        except (ImportError, AttributeError) as e:
            raise ModelLoadError("", str(e)) from e
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise CUDAOutOfMemoryError("") from e
            else:
                raise ModelError(str(e)) from e

    return wrapper


# ===== 錯誤報告與記錄 =====


class ErrorReporter:
    """錯誤報告器"""

    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.recent_errors: List[Dict[str, Any]] = []
        self.max_recent_errors = 100

    def report_error(self, error: CharaForgeError):
        """報告錯誤"""
        error_key = error.error_code

        # 更新計數
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

        # 記錄最近錯誤
        error_record = {
            "timestamp": logger.info.__module__,  # 使用當前時間
            "error_code": error.error_code,
            "message": error.message,
            "details": error.details,
        }

        self.recent_errors.append(error_record)

        # 限制記錄數量
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors.pop(0)

        # 記錄到日誌
        logger.error(f"Error reported: {error}")

    def get_error_summary(self) -> Dict[str, Any]:
        """取得錯誤摘要"""
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_types": len(self.error_counts),
            "error_counts": self.error_counts.copy(),
            "recent_errors_count": len(self.recent_errors),
            "most_common_errors": sorted(
                self.error_counts.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }

    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """取得最近的錯誤"""
        return self.recent_errors[-limit:] if self.recent_errors else []


# ===== 全域錯誤報告器 =====
global_error_reporter = ErrorReporter()


def report_error(error: CharaForgeError):
    """報告錯誤到全域報告器"""
    global_error_reporter.report_error(error)


def get_error_summary() -> Dict[str, Any]:
    """取得全域錯誤摘要"""
    return global_error_reporter.get_error_summary()


# ===== 便利函數 =====


def safe_call(func, *args, default=None, log_errors: bool = True, **kwargs):
    """安全呼叫函數（不拋出例外）"""
    try:
        return func(*args, **kwargs)
    except CharaForgeError as e:
        if log_errors:
            logger.warning(f"CharaForge error in {func.__name__}: {e}")
        report_error(e)
        return default
    except Exception as e:
        if log_errors:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
        # 包裝為 CharaForge 錯誤並報告
        wrapped_error = CharaForgeError(
            f"Unexpected error: {str(e)}",
            "UNEXPECTED_ERROR",
            {"function": func.__name__, "original_error": str(e)},
        )
        report_error(wrapped_error)
        return default


def validate_required(value: Any, field_name: str):
    """驗證必填欄位"""
    if value is None or (isinstance(value, str) and value.strip() == ""):
        raise ValidationError(field_name, value, "Field is required")


def validate_range(
    value: float, field_name: str, min_val: float = None, max_val: float = None  # type: ignore
):
    """驗證數值範圍"""
    if min_val is not None and value < min_val:
        raise ValidationError(field_name, value, f"Must be >= {min_val}")
    if max_val is not None and value > max_val:
        raise ValidationError(field_name, value, f"Must be <= {max_val}")


def validate_choices(value: Any, field_name: str, choices: List[Any]):
    """驗證選項"""
    if value not in choices:
        raise ValidationError(field_name, value, f"Must be one of: {choices}")


# ===== 系統健康檢查相關例外 =====


class HealthCheckError(CharaForgeError):
    """健康檢查錯誤"""

    def __init__(
        self, component: str, message: str, error_code: str = "HEALTH_CHECK_ERROR"
    ):
        super().__init__(f"Health check failed for {component}: {message}", error_code)
        self.details["component"] = component


class ServiceUnavailableError(HealthCheckError):
    """服務不可用錯誤"""

    def __init__(self, service_name: str, reason: str = ""):
        message = f"Service unavailable: {service_name}"
        if reason:
            message += f" - {reason}"
        super().__init__(service_name, reason, "SERVICE_UNAVAILABLE")


class DatabaseConnectionError(HealthCheckError):
    """資料庫連線錯誤"""

    def __init__(self, database_type: str = "redis", reason: str = ""):
        message = f"Database connection failed"
        if reason:
            message += f": {reason}"
        super().__init__(database_type, message, "DATABASE_CONNECTION_ERROR")


# ===== 批次處理相關例外 =====


class BatchProcessingError(CharaForgeError):
    """批次處理錯誤"""

    def __init__(
        self, message: str, batch_id: str = "", error_code: str = "BATCH_ERROR"
    ):
        super().__init__(message, error_code, {"batch_id": batch_id})


class BatchJobError(BatchProcessingError):
    """批次任務錯誤"""

    def __init__(self, message: str, job_id: str = "", batch_id: str = ""):
        super().__init__(message, batch_id, "BATCH_JOB_ERROR")
        self.details["job_id"] = job_id


class BatchTimeoutError(BatchProcessingError):
    """批次處理超時"""

    def __init__(self, batch_id: str, timeout_seconds: int):
        message = f"Batch processing timeout after {timeout_seconds} seconds"
        super().__init__(message, batch_id, "BATCH_TIMEOUT")
        self.details["timeout_seconds"] = timeout_seconds


# ===== 匯出相關例外 =====


class ExportError(CharaForgeError):
    """匯出錯誤"""

    def __init__(
        self, message: str, export_type: str = "", error_code: str = "EXPORT_ERROR"
    ):
        super().__init__(message, error_code, {"export_type": export_type})


class ExportNotFoundError(ExportError):
    """匯出檔案未找到"""

    def __init__(self, export_id: str):
        super().__init__(f"Export not found: {export_id}", "", "EXPORT_NOT_FOUND")
        self.details["export_id"] = export_id


# ===== 錯誤恢復策略 =====


class ErrorRecoveryStrategy:
    """錯誤恢復策略"""

    @staticmethod
    def handle_cuda_oom_recovery():
        """CUDA OOM 恢復策略"""
        try:
            import torch

            if torch.cuda.is_available():
                # 清理 GPU 記憶體
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # 強制垃圾回收
                import gc

                gc.collect()

                logger.info("CUDA OOM recovery: GPU memory cleared")
                return True
        except Exception as e:
            logger.error(f"CUDA recovery failed: {e}")

        return False

    @staticmethod
    def handle_model_load_recovery(model_name: str) -> Dict[str, Any]:
        """模型載入恢復策略"""
        recovery_suggestions = {
            "model_name": model_name,
            "suggestions": [
                "Check model name and path",
                "Verify model is downloaded and accessible",
                "Try CPU device if CUDA fails",
                "Enable low-memory mode",
                "Clear GPU memory and retry",
            ],
            "fallback_options": [
                "Use smaller model variant",
                "Enable model quantization (4-bit/8-bit)",
                "Use CPU offloading",
            ],
        }

        return recovery_suggestions

    @staticmethod
    def handle_generation_failure_recovery(prompt: str) -> Dict[str, Any]:
        """生成失敗恢復策略"""
        return {
            "original_prompt": prompt[:100],
            "suggestions": [
                "Simplify the prompt",
                "Reduce image dimensions",
                "Lower the number of inference steps",
                "Disable LoRA models temporarily",
                "Clear GPU memory and retry",
            ],
            "modified_prompt_suggestions": [
                "Remove complex modifiers",
                "Use simpler descriptive words",
                "Reduce negative prompt complexity",
            ],
        }


# ===== 錯誤分析工具 =====


class ErrorAnalyzer:
    """錯誤分析工具"""

    @staticmethod
    def analyze_error_pattern(errors: List[CharaForgeError]) -> Dict[str, Any]:
        """分析錯誤模式"""
        if not errors:
            return {"analysis": "No errors to analyze"}

        error_codes = [e.error_code for e in errors]
        error_messages = [e.message for e in errors]

        # 統計分析
        from collections import Counter

        code_counts = Counter(error_codes)

        # 尋找關鍵字
        common_keywords = []
        for message in error_messages:
            words = message.lower().split()
            common_keywords.extend([w for w in words if len(w) > 3])

        keyword_counts = Counter(common_keywords)

        return {
            "total_errors": len(errors),
            "unique_error_codes": len(set(error_codes)),
            "most_common_errors": code_counts.most_common(5),
            "common_keywords": keyword_counts.most_common(10),
            "severity_distribution": ErrorAnalyzer._analyze_severity(errors),
            "recommendations": ErrorAnalyzer._generate_recommendations(code_counts),
        }

    @staticmethod
    def _analyze_severity(errors: List[CharaForgeError]) -> Dict[str, int]:
        """分析錯誤嚴重程度"""
        severity_map = {
            "CUDA_OOM": "critical",
            "MODEL_NOT_FOUND": "high",
            "VALIDATION_ERROR": "medium",
            "CONTENT_BLOCKED": "low",
        }

        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}

        for error in errors:
            severity = severity_map.get(error.error_code, "medium")
            severity_counts[severity] += 1

        return severity_counts

    @staticmethod
    def _generate_recommendations(error_counts: Counter) -> List[str]:
        """基於錯誤模式生成建議"""
        recommendations = []

        for error_code, count in error_counts.most_common(3):
            if error_code == "CUDA_OOM":
                recommendations.append(
                    f"High CUDA OOM frequency ({count}x): Enable low-memory mode, "
                    "reduce batch size, or use CPU offloading"
                )
            elif error_code == "MODEL_NOT_FOUND":
                recommendations.append(
                    f"Model loading issues ({count}x): Verify model downloads "
                    "and check cache paths"
                )
            elif error_code == "VALIDATION_ERROR":
                recommendations.append(
                    f"Input validation failures ({count}x): Review API parameters "
                    "and add client-side validation"
                )
            elif error_code == "GENERATION_ERROR":
                recommendations.append(
                    f"Generation failures ({count}x): Check prompts and model settings"
                )

        if not recommendations:
            recommendations.append("No specific patterns detected. Monitor for trends.")

        return recommendations


# ===== 測試與示例 =====

if __name__ == "__main__":
    print("=== CharaForge Exception System Test ===")

    # 測試基本例外
    try:
        raise ModelNotFoundError("test-model")
    except ModelNotFoundError as e:
        print(f"✅ Caught ModelNotFoundError: {e}")
        print(f"   Error dict: {e.to_dict()}")

    # 測試錯誤報告器
    test_errors = [
        ModelNotFoundError("model-1"),
        CUDAOutOfMemoryError("model-2", 7.5),
        ValidationError("prompt", "", "Empty prompt"),
        LoRANotFoundError("lora-1"),
    ]

    for error in test_errors:
        report_error(error)

    summary = get_error_summary()
    print(f"\n📊 Error Summary:")
    print(f"   Total errors: {summary['total_errors']}")
    print(f"   Error types: {summary['error_types']}")
    print(f"   Most common: {summary['most_common_errors']}")

    # 測試錯誤分析
    analysis = ErrorAnalyzer.analyze_error_pattern(test_errors)
    print(f"\n🔍 Error Analysis:")
    print(f"   Unique codes: {analysis['unique_error_codes']}")
    print(f"   Recommendations: {analysis['recommendations']}")

    # 測試安全呼叫
    def failing_function():
        raise ValueError("Test error")

    result = safe_call(failing_function, default="fallback_value")
    print(f"\n🛡️  Safe call result: {result}")

    # 測試驗證函數
    try:
        validate_required("", "test_field")
    except ValidationError as e:
        print(f"✅ Validation error caught: {e}")

    print("\n✅ Exception system test completed")
