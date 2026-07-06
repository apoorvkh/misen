"""Low-level fsync helpers shared by NFS-safe publication code paths.

These live in ``misen.utils`` (rather than ``misen.workspaces.disk``, where
they originated) so utility modules like ``misen.utils.snapshot`` can reuse
them without importing a workspace backend.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["fsync_dir", "fsync_file"]


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
