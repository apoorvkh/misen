"""Low-level durability helpers shared by NFS-safe publication paths."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["atomic_write_bytes", "fsync_dir", "fsync_file"]


def atomic_write_bytes(
    path: Path,
    data: bytes,
    *,
    before_commit: Callable[[], None] | None = None,
    overwrite: bool = True,
) -> bool:
    """Durably publish bytes, optionally requiring an absent destination."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    published = False
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if before_commit is not None:
            before_commit()
        if overwrite:
            os.replace(tmp, path)  # noqa: PTH105  -- atomic overwrite; Path has no equivalent
            published = True
        else:
            try:
                os.link(tmp, path)
                published = True
            except FileExistsError:
                pass
    finally:
        Path(tmp).unlink(missing_ok=True)
    if published:
        fsync_dir(path.parent)
    return published


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
