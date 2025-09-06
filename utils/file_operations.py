# utils/file_operations.py
"""
安全檔案操作工具 - 支援驗證、快取管理、批次處理
"""

import os
import shutil
import hashlib
import json
import tempfile
import mimetypes
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Generator
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta

from .logging import get_logger

logger = get_logger(__name__)


class SafeFileHandler:
    """安全檔案處理器"""

    def __init__(self, base_path: str = None, max_file_size: int = 500 * 1024 * 1024):
        """
        初始化檔案處理器

        Args:
            base_path: 基礎路徑，預設使用快取根目錄
            max_file_size: 最大檔案大小 (bytes)，預設 500MB
        """
        self.base_path = Path(
            base_path or os.getenv("AI_CACHE_ROOT", "../ai_warehouse/cache")
        )
        self.max_file_size = max_file_size
        self._lock = threading.Lock()

        # 允許的檔案類型
        self.allowed_extensions = {
            "images": {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"},
            "models": {".safetensors", ".bin", ".ckpt", ".pt", ".pth"},
            "configs": {".json", ".yaml", ".yml", ".toml"},
            "data": {".csv", ".tsv", ".txt", ".jsonl"},
            "archives": {".zip", ".tar", ".gz", ".7z"},
        }

        # 建立必要目錄
        self._ensure_directories()

    def _ensure_directories(self):
        """確保必要目錄存在"""
        directories = ["temp", "uploads", "outputs", "logs", "backups"]
        for dirname in directories:
            (self.base_path / dirname).mkdir(parents=True, exist_ok=True)

    def validate_path(self, path: Union[str, Path], must_exist: bool = False) -> bool:
        """
        驗證路徑安全性

        Args:
            path: 檔案路徑
            must_exist: 是否必須存在

        Returns:
            bool: 路徑是否安全
        """
        try:
            path = Path(path).resolve()

            # 檢查是否在允許的基礎路徑下
            if not str(path).startswith(str(self.base_path.resolve())):
                logger.warning(f"Path outside base directory: {path}")
                return False

            # 檢查路徑遍歷攻擊
            if ".." in str(path) or path.name.startswith("."):
                logger.warning(f"Suspicious path detected: {path}")
                return False

            # 檢查是否存在 (如果需要)
            if must_exist and not path.exists():
                logger.warning(f"Required path does not exist: {path}")
                return False

            return True

        except Exception as e:
            logger.error(f"Path validation error: {e}")
            return False

    def validate_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        驗證檔案

        Returns:
            Dict: 驗證結果
        """
        file_path = Path(file_path)
        result = {
            "valid": False,
            "path_safe": False,
            "size_ok": False,
            "type_allowed": False,
            "mime_type": None,
            "file_size": 0,
            "errors": [],
        }

        try:
            # 路徑安全性
            result["path_safe"] = self.validate_path(file_path, must_exist=True)
            if not result["path_safe"]:
                result["errors"].append("Unsafe file path")
                return result

            # 檔案大小
            result["file_size"] = file_path.stat().st_size
            result["size_ok"] = result["file_size"] <= self.max_file_size
            if not result["size_ok"]:
                result["errors"].append(
                    f"File too large: {result['file_size']} > {self.max_file_size}"
                )

            # MIME 類型
            result["mime_type"] = mimetypes.guess_type(str(file_path))[0]

            # 副檔名檢查
            file_ext = file_path.suffix.lower()
            type_allowed = False
            for category, extensions in self.allowed_extensions.items():
                if file_ext in extensions:
                    type_allowed = True
                    result["file_category"] = category
                    break

            result["type_allowed"] = type_allowed
            if not type_allowed:
                result["errors"].append(f"File type not allowed: {file_ext}")

            # 整體驗證結果
            result["valid"] = (
                result["path_safe"] and result["size_ok"] and result["type_allowed"]
            )

        except Exception as e:
            result["errors"].append(f"Validation error: {str(e)}")
            logger.error(f"File validation failed: {e}")

        return result

    def safe_copy(self, src: Union[str, Path], dst: Union[str, Path]) -> bool:
        """安全複製檔案"""
        try:
            src_path = Path(src)
            dst_path = Path(dst)

            # 驗證來源檔案
            validation = self.validate_file(src_path)
            if not validation["valid"]:
                logger.error(f"Source file validation failed: {validation['errors']}")
                return False

            # 驗證目標路徑
            if not self.validate_path(dst_path.parent):
                logger.error(f"Destination path unsafe: {dst_path.parent}")
                return False

            # 確保目標目錄存在
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            # 執行複製
            with self._lock:
                shutil.copy2(src_path, dst_path)

            logger.info(f"File copied: {src_path} -> {dst_path}")
            return True

        except Exception as e:
            logger.error(f"File copy failed: {e}")
            return False

    def safe_move(self, src: Union[str, Path], dst: Union[str, Path]) -> bool:
        """安全移動檔案"""
        try:
            if self.safe_copy(src, dst):
                Path(src).unlink()
                logger.info(f"File moved: {src} -> {dst}")
                return True
            return False
        except Exception as e:
            logger.error(f"File move failed: {e}")
            return False

    def get_file_hash(
        self, file_path: Union[str, Path], algorithm: str = "md5"
    ) -> Optional[str]:
        """計算檔案雜湊值"""
        try:
            file_path = Path(file_path)

            if not self.validate_path(file_path, must_exist=True):
                return None

            hash_obj = hashlib.new(algorithm)

            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_obj.update(chunk)

            return hash_obj.hexdigest()

        except Exception as e:
            logger.error(f"Hash calculation failed: {e}")
            return None

    def cleanup_temp_files(self, max_age_hours: int = 24) -> Dict[str, int]:
        """清理暫存檔案"""
        temp_dir = self.base_path / "temp"
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        cleanup_stats = {"files_removed": 0, "space_freed_mb": 0}

        try:
            for file_path in temp_dir.rglob("*"):
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)

                    if file_time < cutoff_time:
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        file_path.unlink()
                        cleanup_stats["files_removed"] += 1
                        cleanup_stats["space_freed_mb"] += size_mb

            logger.info(f"Temp cleanup completed: {cleanup_stats}")

        except Exception as e:
            logger.error(f"Temp cleanup failed: {e}")

        return cleanup_stats

    @contextmanager
    def temp_file(self, suffix: str = "", prefix: str = "sagaforge_"):
        """臨時檔案 context manager"""
        temp_dir = self.base_path / "temp"

        try:
            with tempfile.NamedTemporaryFile(
                suffix=suffix, prefix=prefix, dir=temp_dir, delete=False
            ) as tmp_file:
                temp_path = Path(tmp_file.name)

            yield temp_path

        finally:
            # 清理臨時檔案
            if temp_path.exists():
                temp_path.unlink()


def ensure_directory(path: Union[str, Path]) -> Path:
    """確保目錄存在"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_path(path: Union[str, Path], base_path: str = None) -> bool:
    """驗證路徑 (獨立函數)"""
    handler = SafeFileHandler(base_path)
    return handler.validate_path(path)


def get_file_hash(file_path: Union[str, Path], algorithm: str = "md5") -> Optional[str]:
    """計算檔案雜湊值 (獨立函數)"""
    handler = SafeFileHandler()
    return handler.get_file_hash(file_path, algorithm)


def safe_json_load(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """安全載入 JSON 檔案"""
    try:
        file_path = Path(file_path)

        # 基本驗證
        if not file_path.exists():
            logger.warning(f"JSON file not found: {file_path}")
            return None

        if file_path.suffix.lower() != ".json":
            logger.warning(f"Not a JSON file: {file_path}")
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.debug(f"JSON loaded: {file_path}")
        return data

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load JSON {file_path}: {e}")
        return None


def safe_json_save(data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
    """安全儲存 JSON 檔案"""
    try:
        file_path = Path(file_path)

        # 確保目錄存在
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # 寫入臨時檔案，然後原子性移動
        temp_path = file_path.with_suffix(".tmp")

        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # 原子性移動
        temp_path.replace(file_path)

        logger.debug(f"JSON saved: {file_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save JSON {file_path}: {e}")
        return False


def get_directory_size(path: Union[str, Path]) -> Tuple[int, int]:
    """
    計算目錄大小

    Returns:
        Tuple[int, int]: (總大小 bytes, 檔案數量)
    """
    try:
        path = Path(path)
        total_size = 0
        file_count = 0

        for file_path in path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1

        return total_size, file_count

    except Exception as e:
        logger.error(f"Failed to calculate directory size: {e}")
        return 0, 0


def batch_file_processor(
    file_paths: List[Union[str, Path]], operation: str, **kwargs
) -> Generator[Dict[str, Any], None, None]:
    """
    批次檔案處理器

    Args:
        file_paths: 檔案路徑列表
        operation: 操作類型 ('validate', 'hash', 'copy', 'move')
        **kwargs: 額外參數

    Yields:
        Dict: 每個檔案的處理結果
    """
    handler = SafeFileHandler()

    for i, file_path in enumerate(file_paths):
        result = {
            "index": i,
            "file_path": str(file_path),
            "success": False,
            "operation": operation,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            if operation == "validate":
                validation = handler.validate_file(file_path)
                result.update(validation)
                result["success"] = validation["valid"]

            elif operation == "hash":
                algorithm = kwargs.get("algorithm", "md5")
                file_hash = handler.get_file_hash(file_path, algorithm)
                result["hash"] = file_hash
                result["algorithm"] = algorithm
                result["success"] = file_hash is not None

            elif operation == "copy":
                dst = kwargs.get("destination")
                if dst:
                    dst_path = Path(dst) / Path(file_path).name
                    result["destination"] = str(dst_path)
                    result["success"] = handler.safe_copy(file_path, dst_path)
                else:
                    result["error"] = "No destination specified"

            elif operation == "move":
                dst = kwargs.get("destination")
                if dst:
                    dst_path = Path(dst) / Path(file_path).name
                    result["destination"] = str(dst_path)
                    result["success"] = handler.safe_move(file_path, dst_path)
                else:
                    result["error"] = "No destination specified"

            else:
                result["error"] = f"Unknown operation: {operation}"

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Batch operation failed for {file_path}: {e}")

        yield result


class BackupManager:
    """備份管理器"""

    def __init__(self, base_path: str = None):
        self.base_path = Path(
            base_path or os.getenv("AI_CACHE_ROOT", "../ai_warehouse/cache")
        )
        self.backup_dir = self.base_path / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def create_backup(
        self, source_path: Union[str, Path], backup_name: str = None
    ) -> Optional[Path]:
        """建立備份"""
        try:
            source_path = Path(source_path)

            if not source_path.exists():
                logger.error(f"Source path not found: {source_path}")
                return None

            # 產生備份名稱
            if backup_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"{source_path.name}_{timestamp}"

            backup_path = self.backup_dir / backup_name

            # 執行備份
            if source_path.is_file():
                shutil.copy2(source_path, backup_path)
            else:
                shutil.copytree(source_path, backup_path)

            logger.info(f"Backup created: {backup_path}")
            return backup_path

        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return None

    def restore_backup(self, backup_name: str, restore_path: Union[str, Path]) -> bool:
        """還原備份"""
        try:
            backup_path = self.backup_dir / backup_name

            if not backup_path.exists():
                logger.error(f"Backup not found: {backup_path}")
                return False

            restore_path = Path(restore_path)

            # 執行還原
            if backup_path.is_file():
                shutil.copy2(backup_path, restore_path)
            else:
                if restore_path.exists():
                    shutil.rmtree(restore_path)
                shutil.copytree(backup_path, restore_path)

            logger.info(f"Backup restored: {backup_path} -> {restore_path}")
            return True

        except Exception as e:
            logger.error(f"Backup restoration failed: {e}")
            return False

    def list_backups(self) -> List[Dict[str, Any]]:
        """列出備份"""
        backups = []

        try:
            for backup_path in self.backup_dir.iterdir():
                if backup_path.name.startswith("."):
                    continue

                stat_info = backup_path.stat()
                backup_info = {
                    "name": backup_path.name,
                    "path": str(backup_path),
                    "size_mb": round(stat_info.st_size / (1024 * 1024), 2),
                    "created": datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                    "is_directory": backup_path.is_dir(),
                }
                backups.append(backup_info)

            # 按建立時間排序
            backups.sort(key=lambda x: x["created"], reverse=True)

        except Exception as e:
            logger.error(f"Failed to list backups: {e}")

        return backups

    def cleanup_old_backups(
        self, max_age_days: int = 30, max_count: int = 10
    ) -> Dict[str, int]:
        """清理舊備份"""
        cleanup_stats = {"removed_count": 0, "space_freed_mb": 0}

        try:
            backups = self.list_backups()

            # 依日期和數量限制刪除
            cutoff_date = datetime.now() - timedelta(days=max_age_days)

            for i, backup in enumerate(backups):
                backup_path = Path(backup["path"])
                backup_date = datetime.fromisoformat(
                    backup["created"].replace("Z", "+00:00").replace("+00:00", "")
                )

                should_remove = (
                    backup_date < cutoff_date or i >= max_count  # 太舊  # 超過數量限制
                )

                if should_remove and backup_path.exists():
                    size_mb = backup["size_mb"]

                    if backup_path.is_file():
                        backup_path.unlink()
                    else:
                        shutil.rmtree(backup_path)

                    cleanup_stats["removed_count"] += 1
                    cleanup_stats["space_freed_mb"] += size_mb

            logger.info(f"Backup cleanup completed: {cleanup_stats}")

        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")

        return cleanup_stats
