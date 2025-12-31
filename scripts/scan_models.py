#!/usr/bin/env python3
"""Scan `/mnt/c/ai_models` and write `/mnt/c/ai_models/registry.json`.

Usage:
  python scripts/scan_models.py --replace
"""

from __future__ import annotations

import argparse
import json

from core.config import bootstrap_config
from core.train.registry import get_model_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan local model folders into registry.json")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Rebuild registry from disk (drop existing model entries)",
    )
    args = parser.parse_args()

    bootstrap_config(verbose=False)
    registry = get_model_registry()
    result = registry.scan_filesystem(replace=bool(args.replace))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

