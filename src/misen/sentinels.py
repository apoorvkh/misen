"""Sentinel objects used for runtime argument injection.

These markers can be used as task arguments and are resolved at execution time
inside :func:`misen.utils.task_utils.execute_task`:

- ``WORK_DIR`` -> per-task working directory path
- ``ASSIGNED_RESOURCES`` -> scheduler-assigned CPU/GPU metadata (single-node)
- ``ASSIGNED_RESOURCES_PER_NODE`` -> scheduler-assigned CPU/GPU metadata keyed by hostname
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Self, cast

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from misen.utils.assigned_resources import AssignedResources, AssignedResourcesPerNode

__all__ = ["ASSIGNED_RESOURCES", "ASSIGNED_RESOURCES_PER_NODE", "WORK_DIR"]


class _RuntimeSentinel:
    """Pickle-stable singleton sentinel used for runtime argument injection."""

    __slots__ = ("_name",)
    _name: str
    _instances: ClassVar[dict[str, _RuntimeSentinel]] = {}

    def __new__(cls, name: str) -> Self:
        if name in cls._instances:
            return cast("Self", cls._instances[name])
        self = cast("Self", super().__new__(cls))
        self._name = name
        cls._instances[name] = self
        return self

    def __repr__(self) -> str:
        return self._name

    def __reduce__(self) -> tuple[Callable[[str], _RuntimeSentinel], tuple[str]]:
        # Preserve singleton identity when payloads are pickled across processes.
        return (_runtime_sentinel, (self._name,))


def _runtime_sentinel(name: str) -> _RuntimeSentinel:
    return _RuntimeSentinel(name)


WORK_DIR = cast("Path", _RuntimeSentinel("WORK_DIR"))
"""Sentinel indicating "inject this task's runtime work directory"."""

ASSIGNED_RESOURCES = cast("AssignedResources | None", _RuntimeSentinel("ASSIGNED_RESOURCES"))
"""Sentinel indicating "inject scheduler-assigned runtime resources for one node"."""

ASSIGNED_RESOURCES_PER_NODE = cast("AssignedResourcesPerNode | None", _RuntimeSentinel("ASSIGNED_RESOURCES_PER_NODE"))
"""Sentinel indicating "inject scheduler-assigned runtime resources for all nodes"."""
