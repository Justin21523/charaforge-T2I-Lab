#!/usr/bin/env python3
"""Verify the runtime dependencies expected by the full API.

Usage:
  python scripts/check_ai_env.py
"""

from __future__ import annotations

import importlib

REQUIRED = [
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "peft",
    "celery",
    "redis",
]


def main() -> int:
    missing = []
    versions = {}

    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            missing.append(name)

    if missing:
        print("Missing packages:", ", ".join(missing))
        print("Fix (recommended):")
        print("  conda env create -f environment.yml")
        print("  conda activate ai_env")
        print("  pip install -r requirements.txt")
        print("")
        print("Verification command:")
        print('  python -c "import peft,redis,celery"')
        return 1

    print("OK. Versions:")
    for name in REQUIRED:
        print(f"  - {name}=={versions.get(name)}")

    print("")
    print('Quick verify: python -c "import peft,redis,celery"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
