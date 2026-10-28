"""Crash-safe filesystem writes used by the job cache.

All temporary files are created next to their destination so ``os.replace`` is
atomic on the supported filesystems (including Windows NTFS).  The helpers do
not delete or overwrite a destination until the replacement is ready.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> Path:
    """Write *data* and atomically replace *path* after it is flushed."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise
    return destination


def atomic_write_text(
    path: str | os.PathLike[str], text: str, *, encoding: str = "utf-8"
) -> Path:
    return atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(
    path: str | os.PathLike[str], value: Any, *, indent: int = 2
) -> Path:
    text = json.dumps(value, ensure_ascii=False, indent=indent, sort_keys=True) + "\n"
    return atomic_write_text(path, text)


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory durability; Windows does not expose this portably."""

    if os.name == "nt":
        return
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
