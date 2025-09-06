# utils/security.py
"""
安全性工具 - Token 管理、內容驗證、輸入清理
"""

import os
import re
import hashlib
import secrets
import base64
from typing import Dict, Any, Optional, List, Set, Union
from datetime import datetime, timedelta
from pathlib import Path
import threading
from functools import wraps

from .logging import get_logger

logger = get_logger(__name__)


class TokenManager:
    """Token 安全管理器"""

    def __init__(self):
        self._tokens: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        # 敏感環境變數名稱
        self.sensitive_env_vars = {
            "HUGGINGFACE_TOKEN",
            "HF_TOKEN",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "WANDB_API_KEY",
            "COMET_API_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "AZURE_CLIENT_SECRET",
        }

        # 載入環境變數中的 tokens
        self._load_env_tokens()

    def _load_env_tokens(self):
        """從環境變數載入 tokens"""
        for var_name in self.sensitive_env_vars:
            token_value = os.getenv(var_name)
            if token_value:
                self._tokens[var_name] = {
                    "value": token_value,
                    "source": "environment",
                    "masked": self._mask_token(token_value),
                    "loaded_at": datetime.now().isoformat(),
                }

    def _mask_token(self, token: str) -> str:
        """遮罩 token 顯示"""
        if len(token) <= 8:
            return "*" * len(token)
        return token[:4] + "*" * (len(token) - 8) + token[-4:]

    def get_token(self, name: str) -> Optional[str]:
        """安全取得 token"""
        with self._lock:
            token_info = self._tokens.get(name)
            if token_info:
                return token_info["value"]
        return None

    def set_token(self, name: str, value: str, source: str = "manual") -> bool:
        """設定 token"""
        try:
            with self._lock:
                self._tokens[name] = {
                    "value": value,
                    "source": source,
                    "masked": self._mask_token(value),
                    "set_at": datetime.now().isoformat(),
                }

            logger.info(f"Token set: {name} (source: {source})")
            return True

        except Exception as e:
            logger.error(f"Failed to set token {name}: {e}")
            return False

    def list_tokens(self, show_masked: bool = True) -> Dict[str, Any]:
        """列出已載入的 tokens (安全模式)"""
        with self._lock:
            if show_masked:
                return {
                    name: {
                        "masked": info["masked"],
                        "source": info["source"],
                        "loaded_at": info.get("loaded_at", info.get("set_at")),
                    }
                    for name, info in self._tokens.items()
                }
            else:
                return {
                    name: {
                        "source": info["source"],
                        "has_value": bool(info["value"]),
                        "loaded_at": info.get("loaded_at", info.get("set_at")),
                    }
                    for name, info in self._tokens.items()
                }

    def validate_token_format(
        self, token_name: str, token_value: str
    ) -> Dict[str, Any]:
        """驗證 token 格式"""
        result = {"valid": False, "issues": []}

        # 基本檢查
        if not token_value or len(token_value.strip()) == 0:
            result["issues"].append("Token is empty")
            return result

        token_value = token_value.strip()

        # 格式驗證規則
        validation_rules = {
            "HUGGINGFACE_TOKEN": {
                "pattern": r"^hf_[a-zA-Z0-9]{30,}$",
                "min_length": 33,
                "description": 'Should start with "hf_" followed by alphanumeric characters',
            },
            "OPENAI_API_KEY": {
                "pattern": r"^sk-[a-zA-Z0-9]{20,}$",
                "min_length": 23,
                "description": 'Should start with "sk-" followed by alphanumeric characters',
            },
            "WANDB_API_KEY": {
                "pattern": r"^[a-f0-9]{40}$",
                "min_length": 40,
                "max_length": 40,
                "description": "Should be 40 hexadecimal characters",
            },
        }

        # 執行特定驗證
        if token_name in validation_rules:
            rules = validation_rules[token_name]

            # 長度檢查
            if "min_length" in rules and len(token_value) < rules["min_length"]:
                result["issues"].append(f"Token too short (min: {rules['min_length']})")

            if "max_length" in rules and len(token_value) > rules["max_length"]:
                result["issues"].append(f"Token too long (max: {rules['max_length']})")

            # 格式檢查
            if "pattern" in rules and not re.match(rules["pattern"], token_value):
                result["issues"].append(f"Invalid format: {rules['description']}")

        # 通用安全檢查
        if len(token_value) < 8:
            result["issues"].append("Token seems too short for security")

        if token_value.lower() in ["test", "demo", "example", "placeholder"]:
            result["issues"].append("Token appears to be a placeholder")

        result["valid"] = len(result["issues"]) == 0
        return result

    def generate_api_key(self, length: int = 32) -> str:
        """產生安全的 API key"""
        return secrets.token_urlsafe(length)


class ContentValidator:
    """內容驗證器"""

    def __init__(self):
        # 危險關鍵字模式
        self.dangerous_patterns = {
            "sql_injection": [
                r"(\bUNION\b.*\bSELECT\b)",
                r"(\bDROP\b.*\bTABLE\b)",
                r"(\bINSERT\b.*\bINTO\b)",
                r"(\bDELETE\b.*\bFROM\b)",
                r"(\'.*OR.*\'.*=.*\')",
            ],
            "script_injection": [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"eval\s*\(",
                r"setTimeout\s*\(",
            ],
            "path_traversal": [
                r"\.\./",
                r"\.\.\x5c",
                r"%2e%2e%2f",
                r"%2e%2e%5c",
            ],
            "command_injection": [
                r"[;&|`]",
                r"\$\(",
                r"`.*`",
                r"\|\s*\w+",
            ],
        }

        # 編譯正則表達式
        self.compiled_patterns = {}
        for category, patterns in self.dangerous_patterns.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in patterns
            ]

    def validate_input(
        self, content: str, content_type: str = "general"
    ) -> Dict[str, Any]:
        """驗證輸入內容"""
        result = {
            "safe": True,
            "issues": [],
            "content_type": content_type,
            "length": len(content),
            "detected_threats": [],
        }

        # 基本檢查
        if not isinstance(content, str):
            result["safe"] = False
            result["issues"].append("Content must be string")
            return result

        # 長度檢查
        max_lengths = {"prompt": 2000, "filename": 255, "path": 4096, "general": 10000}

        max_length = max_lengths.get(content_type, max_lengths["general"])
        if len(content) > max_length:
            result["safe"] = False
            result["issues"].append(f"Content too long (max: {max_length})")

        # 威脅檢測
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(content):
                    result["safe"] = False
                    result["detected_threats"].append(category)
                    result["issues"].append(f"Detected {category} pattern")
                    break

        # 特殊字元檢查
        suspicious_chars = ["<", ">", '"', "'", "&", ";", "|", "`"]
        if content_type in ["filename", "path"]:
            for char in suspicious_chars:
                if char in content:
                    result["safe"] = False
                    result["issues"].append(f"Suspicious character: {char}")

        return result

    def sanitize_filename(self, filename: str) -> str:
        """清理檔案名稱"""
        # 移除危險字元
        sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)

        # 移除控制字元
        sanitized = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", sanitized)

        # 限制長度
        if len(sanitized) > 255:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[: 255 - len(ext)] + ext

        # 避免保留名稱
        reserved_names = {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            "COM1",
            "COM2",
            "COM3",
            "COM4",
            "COM5",
            "COM6",
            "COM7",
            "COM8",
            "COM9",
            "LPT1",
            "LPT2",
            "LPT3",
            "LPT4",
            "LPT5",
            "LPT6",
            "LPT7",
            "LPT8",
            "LPT9",
        }

        name_only = os.path.splitext(sanitized)[0].upper()
        if name_only in reserved_names:
            sanitized = f"_{sanitized}"

        return sanitized


class InputSanitizer:
    """輸入清理器"""

    @staticmethod
    def sanitize_prompt(prompt: str) -> str:
        """清理生成提示詞"""
        # 移除潛在的指令注入
        dangerous_prefixes = [
            "ignore previous instructions",
            "forget everything",
            "new instruction:",
            "system:",
            "assistant:",
            "admin:",
            "root:",
        ]

        sanitized = prompt.lower()
        for prefix in dangerous_prefixes:
            if sanitized.startswith(prefix):
                logger.warning(f"Dangerous prompt prefix detected: {prefix}")
                prompt = prompt[len(prefix) :].strip()
                break

        # 限制長度
        max_length = 2000
        if len(prompt) > max_length:
            prompt = prompt[:max_length]
            logger.warning(f"Prompt truncated to {max_length} characters")

        return prompt.strip()

    @staticmethod
    def sanitize_path_component(component: str) -> str:
        """清理路徑組件"""
        # 移除路徑遍歷
        component = component.replace("..", "")
        component = component.replace("/", "_")
        component = component.replace("\\", "_")

        # 移除特殊字元
        component = re.sub(r'[<>:"|?*]', "_", component)

        # 移除前後空白和點
        component = component.strip(" .")

        return component


def secure_filename(filename: str) -> str:
    """安全檔案名稱 (獨立函數)"""
    validator = ContentValidator()
    return validator.sanitize_filename(filename)


def sanitize_input(content: str, content_type: str = "general") -> str:
    """清理輸入 (獨立函數)"""
    validator = ContentValidator()

    if content_type == "prompt":
        return InputSanitizer.sanitize_prompt(content)
    elif content_type == "filename":
        return secure_filename(content)
    elif content_type == "path":
        return InputSanitizer.sanitize_path_component(content)

    # 通用清理
    validation = validator.validate_input(content, content_type)
    if not validation["safe"]:
        logger.warning(f"Input validation failed: {validation['issues']}")
        # 基本清理
        content = re.sub(r'[<>&"\'`;|]', "", content)

    return content


def require_token(token_name: str):
    """裝飾器：要求特定 token"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            token_manager = TokenManager()
            token_value = token_manager.get_token(token_name)

            if not token_value:
                raise ValueError(f"Required token not found: {token_name}")

            return func(*args, **kwargs)

        return wrapper

    return decorator


class SecurityAudit:
    """安全性稽核"""

    @staticmethod
    def check_environment() -> Dict[str, Any]:
        """檢查環境安全性"""
        audit_result = {
            "secure": True,
            "issues": [],
            "recommendations": [],
            "token_status": {},
        }

        token_manager = TokenManager()

        # 檢查必要 tokens
        required_tokens = ["HUGGINGFACE_TOKEN"]
        for token_name in required_tokens:
            token_value = token_manager.get_token(token_name)
            audit_result["token_status"][token_name] = {
                "present": token_value is not None,
                "valid_format": False,
            }

            if token_value:
                validation = token_manager.validate_token_format(
                    token_name, token_value
                )
                audit_result["token_status"][token_name]["valid_format"] = validation[
                    "valid"
                ]

                if not validation["valid"]:
                    audit_result["secure"] = False
                    audit_result["issues"].extend(
                        [f"{token_name}: {issue}" for issue in validation["issues"]]
                    )
            else:
                audit_result["recommendations"].append(
                    f"Consider setting {token_name} for full functionality"
                )

        # 檢查權限設定
        cache_root = os.getenv("AI_CACHE_ROOT", "../ai_warehouse/cache")
        cache_path = Path(cache_root)

        if cache_path.exists():
            try:
                # 測試寫入權限
                test_file = cache_path / ".security_test"
                test_file.touch()
                test_file.unlink()
            except Exception:
                audit_result["secure"] = False
                audit_result["issues"].append("Cache directory not writable")
        else:
            audit_result["recommendations"].append(
                "Cache directory will be created on first use"
            )

        return audit_result
