import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure tests never write to the real AI_WAREHOUSE paths.
_tmp_root = tempfile.mkdtemp(prefix="charaforge_test_")
os.environ["PROJECT_SLUG"] = "charaforge-test"
os.environ["AI_CACHE_ROOT"] = os.path.join(_tmp_root, "ai_cache")
os.environ["AI_MODELS_ROOT"] = os.path.join(_tmp_root, "ai_models")
os.environ["AI_DATASETS_ROOT"] = os.path.join(_tmp_root, "datasets")
os.environ["AI_TRAINING_ROOT"] = os.path.join(_tmp_root, "training")

# Ensure framework caches follow the same temp root (core.config respects these).
os.environ["XDG_CACHE_HOME"] = os.environ["AI_CACHE_ROOT"]
os.environ["HF_HOME"] = os.path.join(os.environ["AI_CACHE_ROOT"], "huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]
os.environ["TORCH_HOME"] = os.path.join(os.environ["AI_CACHE_ROOT"], "torch")
os.environ["HF_HUB_CACHE"] = os.path.join(os.environ["HF_HOME"], "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ["HF_HUB_CACHE"]
os.environ["PIP_CACHE_DIR"] = os.path.join(os.environ["AI_CACHE_ROOT"], "pip")

# Ensure tests are not impacted by a developer's local auth/rate-limit env vars.
for _var in (
    "API_KEY",
    "KEY",
    "API_KEYS",
    "API_ADMIN_KEYS",
    "API_KEY_HEADER",
    "API_RATE_LIMIT",
    "API_SCAN_RATE_LIMIT",
):
    os.environ.pop(_var, None)
os.environ["API_RATE_LIMIT"] = "0"
os.environ["API_SCAN_RATE_LIMIT"] = "0"

# Ensure the repository root is importable (so `import api`, `import core`, etc).
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


@pytest.fixture
def anyio_backend():
    return "asyncio"
