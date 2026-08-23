"""Low-level durability helpers shared by NFS-safe publication paths."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

__all__ = ["atomic_write_bytes", "fsync_dir", "fsync_file"]


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Durably publish ``data`` via mkstemp, fsync, replace, and directory fsync."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # noqa: PTH105  -- atomic overwrite; Path has no equivalent
    finally:
        Path(tmp).unlink(missing_ok=True)
    fsync_dir(path.parent)


def fsync_dir(path: Path) -> None:
    """Fsync a directory so a contained rename or unlink is durable."""
    _fsync_path(path, os.O_DIRECTORY)


def fsync_file(path: Path) -> None:
    """Fsync a regular file's contents."""
    _fsync_path(path, os.O_RDONLY)


def _fsync_path(path: Path, flags: int) -> None:
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
