import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure tests never write to the real AI_WAREHOUSE paths.
_tmp_root = tempfile.mkdtemp(prefix="charaforge_test_")
os.environ.setdefault("PROJECT_SLUG", "charaforge-test")
os.environ.setdefault("AI_CACHE_ROOT", os.path.join(_tmp_root, "ai_cache"))
os.environ.setdefault("AI_MODELS_ROOT", os.path.join(_tmp_root, "ai_models"))
os.environ.setdefault("AI_DATASETS_ROOT", os.path.join(_tmp_root, "datasets"))
os.environ.setdefault("AI_TRAINING_ROOT", os.path.join(_tmp_root, "training"))

# Ensure framework caches follow the same temp root (core.config respects these).
os.environ.setdefault("XDG_CACHE_HOME", os.environ["AI_CACHE_ROOT"])
os.environ.setdefault("HF_HOME", os.path.join(os.environ["AI_CACHE_ROOT"], "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])
os.environ.setdefault("TORCH_HOME", os.path.join(os.environ["AI_CACHE_ROOT"], "torch"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ["HF_HUB_CACHE"])
os.environ.setdefault("PIP_CACHE_DIR", os.path.join(os.environ["AI_CACHE_ROOT"], "pip"))

# Ensure the repository root is importable (so `import api`, `import core`, etc).
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


@pytest.fixture
def anyio_backend():
    return "asyncio"
