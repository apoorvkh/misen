"""Sentinel objects used for runtime argument injection.

These markers are bound as top-level ``Task(...)`` arguments and resolved at
execution time inside :func:`misen.utils.task_utils.execute_task`:

- ``SCRATCH_DIR`` -> per-task scratch directory path
- ``DASK_CLIENT`` -> multi-node allocation-scoped :class:`distributed.Client`

Sentinel-valued arguments are excluded from task identity automatically (the
injected value varies per workspace/machine). Misuse is rejected when the
``Task`` is constructed: a sentinel left as an unbound function-signature
default would bypass the argument resolver entirely (Python applies signature
defaults at call time), and a sentinel nested inside a container argument
cannot be resolved — both raise ``TypeError`` at graph-build time instead of
leaking the raw sentinel object into the function body mid-run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Self, cast

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from distributed import Client

__all__ = ["DASK_CLIENT", "SCRATCH_DIR", "is_runtime_sentinel"]


class _RuntimeSentinel:
    """Pickle-stable singleton sentinel used for runtime argument injection."""

    __slots__ = ("_name",)
    _name: str
    _instances: ClassVar[dict[str, _RuntimeSentinel]] = {}

    def __new__(cls, name: str) -> Self:
        if name in cls._instances:
            return cast("Self", cls._instances[name])
        self = super().__new__(cls)
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


def is_runtime_sentinel(value: object) -> bool:
    """Return whether ``value`` is a runtime-injection sentinel (e.g. ``SCRATCH_DIR``)."""
    return isinstance(value, _RuntimeSentinel)


SCRATCH_DIR = cast("Path", _RuntimeSentinel("SCRATCH_DIR"))
"""Sentinel indicating "inject this task's runtime scratch directory"."""

DASK_CLIENT = cast("Client", _RuntimeSentinel("DASK_CLIENT"))
"""Sentinel indicating "inject this multi-node work unit's Dask client"."""
