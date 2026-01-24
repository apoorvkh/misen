"""Sentinel values used across the codebase."""

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import pathlib

WORK_DIR = cast("pathlib.Path", object())
