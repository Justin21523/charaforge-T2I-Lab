# utils/web_search.py
"""
網路搜尋與驗證工具 - 模型下載、URL 驗證、連線檢查
"""

import re
import urllib.parse
import urllib.request
import socket
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import time
from datetime import datetime

from .logging import get_logger
from .security import sanitize_input

logger = get_logger(__name__)


class SearchManager:
    """搜尋與下載管理器"""

    def __init__(self, timeout: int = 30, user_agent: str = None):  # type: ignore
        self.timeout = timeout
        self.user_agent = user_agent or "SagaForge-T2I-Lab/1.0"

        # 信任的模型庫域名
        self.trusted_domains = {
            "huggingface.co",
            "github.com",
            "civitai.com",
            "kaggle.com",
            "drive.google.com",
        }

        # 支援的檔案類型
        self.supported_extensions = {
            ".safetensors",
            ".bin",
            ".ckpt",
            ".pt",
            ".pth",
            ".json",
            ".txt",
            ".yaml",
            ".yml",
        }

    def validate_url(self, url: str) -> Dict[str, Any]:
        """
        驗證 URL 安全性和可達性

        Args:
            url: 要驗證的 URL

        Returns:
            Dict: 驗證結果
        """
        result = {
            "valid": False,
            "reachable": False,
            "trusted": False,
            "file_info": {},
            "issues": [],
        }

        try:
            # 清理輸入
            url = sanitize_input(url, "general").strip()

            # URL 格式驗證
            parsed = urllib.parse.urlparse(url)

            if not parsed.scheme in ("http", "https"):
                result["issues"].append("Only HTTP/HTTPS URLs are allowed")
                return result

            if not parsed.netloc:
                result["issues"].append("Invalid URL format")
                return result

            # 域名檢查
            domain = parsed.netloc.lower()
            if any(trusted in domain for trusted in self.trusted_domains):
                result["trusted"] = True
            else:
                result["issues"].append(f"Untrusted domain: {domain}")

            # 基本格式驗證通過
            result["valid"] = True
            result["parsed_url"] = {
                "scheme": parsed.scheme,
                "domain": domain,
                "path": parsed.path,
                "filename": Path(parsed.path).name,
            }

            # 連線測試
            reachability = self.check_url_reachable(url)
            result.update(reachability)

        except Exception as e:
            result["issues"].append(f"URL validation error: {str(e)}")
            logger.error(f"URL validation failed for {url}: {e}")

        return result

    def check_url_reachable(self, url: str) -> Dict[str, Any]:
        """檢查 URL 是否可達"""
        result = {
            "reachable": False,
            "status_code": None,
            "content_type": None,
            "content_length": None,
            "response_time_ms": None,
        }

        try:
            start_time = time.time()

            # 建立請求
            request = urllib.request.Request(url)
            request.add_header("User-Agent", self.user_agent)

            # 只發送 HEAD 請求檢查可達性
            request.get_method = lambda: "HEAD"

            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result["reachable"] = True
                result["status_code"] = response.getcode()
                result["content_type"] = response.headers.get("Content-Type")
                result["content_length"] = response.headers.get("Content-Length")

                # 嘗試轉換內容長度
                if result["content_length"]:
                    try:
                        content_length_int = int(result["content_length"])
                        result["content_length_mb"] = round(
                            content_length_int / (1024 * 1024), 2
                        )
                    except ValueError:
                        pass

            result["response_time_ms"] = round((time.time() - start_time) * 1000, 2)

        except urllib.error.HTTPError as e: # type: ignore
            result["status_code"] = e.code
            result["error"] = f"HTTP Error {e.code}: {e.reason}"
        except urllib.error.URLError as e:  # type: ignore
            result["error"] = f"URL Error: {e.reason}"
        except socket.timeout:
            result["error"] = f"Timeout after {self.timeout} seconds"
        except Exception as e:
            result["error"] = f"Connection error: {str(e)}"

        return result

    def extract_model_info(self, url: str) -> Dict[str, Any]:
        """從 URL 提取模型資訊"""
        info = {
            "model_name": None,
            "repository": None,
            "filename": None,
            "estimated_type": None,
            "huggingface_repo": None,
        }

        try:
            parsed = urllib.parse.urlparse(url)
            path_parts = [p for p in parsed.path.split("/") if p]

            # Hugging Face 模型
            if "huggingface.co" in parsed.netloc:
                if len(path_parts) >= 2:
                    info["repository"] = f"{path_parts[0]}/{path_parts[1]}"  # type: ignore
                    info["huggingface_repo"] = info["repository"]  # type: ignore

                if "resolve" in path_parts and len(path_parts) >= 4:
                    filename_idx = path_parts.index("resolve") + 2
                    if filename_idx < len(path_parts):
                        info["filename"] = path_parts[filename_idx]  # type: ignore

            # GitHub 發布
            elif "github.com" in parsed.netloc:
                if len(path_parts) >= 2:
                    info["repository"] = f"{path_parts[0]}/{path_parts[1]}"  # type: ignore

                if "releases" in path_parts:
                    info["filename"] = path_parts[-1]  # type: ignore

            # CivitAI
            elif "civitai.com" in parsed.netloc:
                if "models" in path_parts:
                    model_idx = path_parts.index("models") + 1
                    if model_idx < len(path_parts):
                        info["model_name"] = path_parts[model_idx]  # type: ignore

            # 檔案名稱後備方案
            if not info["filename"]:
                info["filename"] = Path(parsed.path).name  # type: ignore

            # 推測模型類型
            if info["filename"]:
                filename_lower = info["filename"].lower()

                if any(x in filename_lower for x in ["sdxl", "xl"]):
                    info["estimated_type"] = "sdxl"  # type: ignore
                elif any(x in filename_lower for x in ["sd_", "sd1", "sd2", "stable"]):
                    info["estimated_type"] = "stable_diffusion"  # type: ignore
                elif any(x in filename_lower for x in ["lora", "dreambooth"]):
                    info["estimated_type"] = "lora"  # type: ignore
                elif "vae" in filename_lower:
                    info["estimated_type"] = "vae"  # type: ignore
                elif any(x in filename_lower for x in ["clip", "text_encoder"]):
                    info["estimated_type"] = "text_encoder"  # type: ignore

        except Exception as e:
            logger.error(f"Failed to extract model info from {url}: {e}")

        return info

    def validate_model_url(self, url: str) -> Dict[str, Any]:
        """專門驗證模型下載 URL"""
        result = self.validate_url(url)

        if result["valid"]:
            # 提取模型資訊
            model_info = self.extract_model_info(url)
            result["model_info"] = model_info

            # 檔案類型檢查
            if model_info["filename"]:
                file_ext = Path(model_info["filename"]).suffix.lower()
                if file_ext in self.supported_extensions:
                    result["supported_file_type"] = True
                else:
                    result["supported_file_type"] = False
                    result["issues"].append(f"Unsupported file type: {file_ext}")

            # 大小警告
            if "content_length_mb" in result and result["content_length_mb"]:
                size_mb = result["content_length_mb"]
                if size_mb > 10000:  # 10GB
                    result["issues"].append(f"Very large file: {size_mb:.1f}MB")
                elif size_mb > 5000:  # 5GB
                    result["issues"].append(f"Large file: {size_mb:.1f}MB")

        return result


class ConnectivityChecker:
    """連線檢查器"""

    @staticmethod
    def check_internet_connection() -> bool:
        """檢查網際網路連線"""
        test_urls = [
            "https://www.google.com",
            "https://huggingface.co",
            "https://github.com",
        ]

        for url in test_urls:
            try:
                request = urllib.request.Request(url)
                request.get_method = lambda: "HEAD"

                with urllib.request.urlopen(request, timeout=10):
                    return True
            except:
                continue

        return False

    @staticmethod
    def check_huggingface_access() -> Dict[str, Any]:
        """檢查 Hugging Face 存取"""
        result = {
            "accessible": False,
            "authenticated": False,
            "api_functional": False,
            "rate_limit_ok": True,
        }

        try:
            # 基本連線測試
            test_url = "https://huggingface.co"
            request = urllib.request.Request(test_url)
            request.get_method = lambda: "HEAD"

            with urllib.request.urlopen(request, timeout=15):
                result["accessible"] = True

            # API 測試
            api_url = "https://huggingface.co/api/models/gpt2"
            with urllib.request.urlopen(api_url, timeout=15):
                result["api_functional"] = True

            # Token 測試
            try:
                from utils import get_token_manager

                token_manager = get_token_manager()
                hf_token = token_manager.get_token("HUGGINGFACE_TOKEN")

                if hf_token:
                    auth_test_url = "https://huggingface.co/api/whoami"
                    request = urllib.request.Request(auth_test_url)
                    request.add_header("Authorization", f"Bearer {hf_token}")

                    with urllib.request.urlopen(request, timeout=15):
                        result["authenticated"] = True
            except:
                pass  # Token 測試失敗不影響基本功能

        except Exception as e:
            result["error"] = str(e)  # type: ignore

        return result


def validate_url(url: str) -> Dict[str, Any]:
    """驗證 URL (獨立函數)"""
    manager = SearchManager()
    return manager.validate_url(url)


def check_model_availability(urls: List[str]) -> Dict[str, Any]:
    """批次檢查模型可用性"""
    manager = SearchManager()
    results = {
        "total_urls": len(urls),
        "available": 0,
        "trusted": 0,
        "unreachable": 0,
        "details": [],
    }

    for i, url in enumerate(urls):
        try:
            validation = manager.validate_model_url(url)

            result_summary = {
                "index": i,
                "url": url,
                "valid": validation.get("valid", False),
                "reachable": validation.get("reachable", False),
                "trusted": validation.get("trusted", False),
                "model_info": validation.get("model_info", {}),
                "issues": validation.get("issues", []),
            }

            results["details"].append(result_summary)

            if result_summary["reachable"]:
                results["available"] += 1
            else:
                results["unreachable"] += 1

            if result_summary["trusted"]:
                results["trusted"] += 1

        except Exception as e:
            results["details"].append({"index": i, "url": url, "error": str(e)})
            results["unreachable"] += 1

    results["success_rate"] = (
        results["available"] / results["total_urls"] if results["total_urls"] > 0 else 0
    )

    return results


def generate_download_command(url: str, output_path: str = None) # type: ignore -> Dict[str, str]:
    """產生下載指令"""
    manager = SearchManager()
    validation = manager.validate_model_url(url)

    if not validation.get("valid"):
        return {"error": "Invalid URL"}

    model_info = validation.get("model_info", {})
    filename = model_info.get("filename", "downloaded_file")

    if not output_path:
        output_path = f"./downloads/{filename}"

    commands = {}

    # wget 指令
    commands["wget"] = f'wget -O "{output_path}" "{url}"'

    # curl 指令
    commands["curl"] = f'curl -L -o "{output_path}" "{url}"'

    # Python 指令
    commands[
        "python"
    ] = f"""
import urllib.request
urllib.request.urlretrieve("{url}", "{output_path}")
"""

    # Hugging Face Hub 指令 (如果適用)
    if validation.get("model_info", {}).get("huggingface_repo"):
        repo = model_info["huggingface_repo"]
        commands[
            "huggingface_hub"
        ] = f"""
from huggingface_hub import snapshot_download
snapshot_download(repo_id="{repo}", local_dir="./models/{repo.replace('/', '_')}")
"""

    return commands
