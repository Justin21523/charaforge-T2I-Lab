"""Cross-process file lock helpers.

Uses `fcntl.flock` when available (recommended), with a fallback lock-file
implementation for environments without `fcntl` (e.g., Windows).
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore


@contextmanager
def file_lock(path: Path, *, timeout_s: float = 5.0, poll_s: float = 0.05) -> Iterator[None]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    timeout_s = float(timeout_s or 0.0)
    poll_s = float(poll_s or 0.05)
    if poll_s <= 0:
        poll_s = 0.01

    if fcntl is not None:
        handle = path.open("a+", encoding="utf-8")
        try:
            deadline = time.monotonic() + max(0.0, timeout_s)
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if timeout_s <= 0 or time.monotonic() >= deadline:
                        raise TimeoutError(f"Timed out acquiring lock: {path}")
                    time.sleep(poll_s)

            yield
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                handle.close()
            except Exception:
                pass
        return

    # Fallback: lock file via O_EXCL creation. Note: if the process crashes, the
    # lock file may remain and must be removed manually.
    deadline = time.monotonic() + max(0.0, timeout_s)
    acquired = False
    fd = None
    while True:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            if timeout_s <= 0 or time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out acquiring lock: {path}")
            time.sleep(poll_s)
            continue
        except Exception as exc:
            raise RuntimeError(f"Failed to acquire lock: {exc}") from exc
        else:
            acquired = True
            break

    try:
        try:
            os.write(fd, f"pid={os.getpid()}\n".encode("utf-8"))
        except Exception:
            pass
        yield
    finally:
        try:
            if fd is not None:
                os.close(fd)
        except Exception:
            pass
        if acquired:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass

