"""Low-level fsync helpers shared by NFS-safe publication code paths.

These live in ``misen.utils`` (rather than ``misen.workspaces.disk``, where
they originated) so utility modules like ``misen.utils.snapshot`` can reuse
them without importing a workspace backend.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

__all__ = ["atomic_write_bytes", "fsync_dir", "fsync_file"]


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Durably publish ``data`` at ``path`` (mkstemp → fsync → rename → fsync-dir).

    The atomic-overwrite-plus-fsync sequence shared by every
    payload-before-pointer commit point (hash-index writes, store and
    snapshot markers): a crash leaves either the old file or the new one,
    never a partial write, and the rename itself is durable.

    Args:
        path: Final file path (parent directory must exist).
        data: File contents.
    """
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
    """Fsync a directory entry so a contained rename or unlink is durable.

    Args:
        path: Directory to fsync.
    """
    fd = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_file(path: Path) -> None:
    """Fsync a regular file's contents so they reach durable storage.

    Args:
        path: File whose contents should be flushed.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
