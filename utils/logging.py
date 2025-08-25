# backend/utils/logging.py - Structured Logging
import logging
import json
import sys
from datetime import datetime
from typing import Dict, Any
import os


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, "extra"):
            log_entry.update(record.extra)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging():
    """Setup application logging configuration"""

    # Create logger
    logger = logging.getLogger("multi_modal_lab")
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler with JSON formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)

    # File handler for persistent logs
    log_dir = f"{os.getenv('AI_CACHE_ROOT', '/tmp')}/logs"
    os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(f"{log_dir}/multi-modal-lab.log")
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    # Celery logger
    celery_logger = logging.getLogger("celery")
    celery_logger.setLevel(logging.INFO)
    celery_logger.addHandler(console_handler)
    celery_logger.addHandler(file_handler)

    return logger


def get_logger(name: str = None):
    """Get logger instance"""
    return logging.getLogger(name or "multi_modal_lab")
