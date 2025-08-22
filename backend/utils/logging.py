# ===== backend/utils/logging.py =====
import logging
import sys
from datetime import datetime


def setup_logging(level=logging.INFO):
    """Setup structured logging"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"/tmp/multimodal-lab.log"),
        ],
    )

    # Reduce noise from transformers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
