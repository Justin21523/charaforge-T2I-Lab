"""Utility helpers for estimating T2I cost units."""

from __future__ import annotations

import math


def estimate_t2i_cost(*, width: int, height: int, steps: int, batch_size: int) -> int:
    width = int(width or 0)
    height = int(height or 0)
    steps = int(steps or 0)
    batch_size = int(batch_size or 0)

    if width <= 0 or height <= 0 or steps <= 0 or batch_size <= 0:
        return 1

    base_pixels = 512 * 512
    pixels = width * height
    area = pixels / base_pixels
    cost = math.ceil(max(1.0, float(steps) * float(batch_size) * area))
    return int(max(cost, 1))

