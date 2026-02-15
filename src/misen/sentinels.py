"""Sentinel objects used for runtime argument injection.

These markers can be used as task arguments and are resolved at execution time
inside :func:`misen.utils.task_utils.execute_task`:

- ``WORK_DIR`` -> per-task working directory path
- ``ASSIGNED_RESOURCES`` -> scheduler-assigned CPU/GPU metadata
"""

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pathlib import Path

    from misen.utils.assigned_resources import AssignedResources

__all__ = ["ASSIGNED_RESOURCES", "WORK_DIR"]

WORK_DIR = cast("Path", object())
"""Sentinel indicating "inject this task's runtime work directory"."""

ASSIGNED_RESOURCES = cast("AssignedResources | None", object())
"""Sentinel indicating "inject scheduler-assigned runtime resources"."""
