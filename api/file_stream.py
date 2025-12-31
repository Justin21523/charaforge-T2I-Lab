"""Small helpers for serving local files without FileResponse."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi.responses import StreamingResponse


async def _iter_file(path: Path, *, chunk_size: int = 1024 * 1024) -> AsyncIterator[bytes]:
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            yield chunk
            await asyncio.sleep(0)


def stream_file(
    path: Path,
    *,
    media_type: str,
    filename: Optional[str] = None,
    disposition: str = "inline",
) -> StreamingResponse:
    headers: dict[str, str] = {}
    if filename:
        headers["Content-Disposition"] = f'{disposition}; filename="{filename}"'
    try:
        headers["Content-Length"] = str(path.stat().st_size)
    except Exception:
        pass
    return StreamingResponse(_iter_file(path), media_type=media_type, headers=headers)

