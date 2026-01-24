"""Helpers for resolving auto-configured components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

if TYPE_CHECKING:
    from misen.executor import Executor
    from misen.workspace import Workspace

__all__ = ["resolve_auto"]


@overload
def resolve_auto(*, workspace: Workspace | Literal["auto"], executor: None = None) -> Workspace: ...
@overload
def resolve_auto(*, workspace: None = None, executor: Executor | Literal["auto"]) -> Executor: ...
def resolve_auto(
    *,
    workspace: Workspace | Literal["auto"] | None = None,
    executor: Executor | Literal["auto"] | None = None,
) -> Workspace | Executor:
    """Automatically instantiate workspace or executor from settings file or default construction.

    Returns:
        The resolved Workspace or Executor instance.

    Raises:
        ValueError: If neither workspace nor executor is provided.
    """
    if workspace is not None:
        if workspace == "auto":
            from misen.workspace import Workspace

            return Workspace.auto()
        return workspace
    if executor is not None:
        if executor == "auto":
            from misen.executor import Executor

            return Executor.auto()
        return executor
    msg = "Either workspace or executor must be specified"
    raise ValueError(msg)
