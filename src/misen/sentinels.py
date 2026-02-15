"""Public sentinel values used in task argument binding."""

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pathlib import Path

    from misen.utils.assigned_resources import AssignedResources

__all__ = ["ASSIGNED_RESOURCES", "WORK_DIR"]

WORK_DIR = cast("Path", object())
ASSIGNED_RESOURCES = cast("AssignedResources | None", object())
