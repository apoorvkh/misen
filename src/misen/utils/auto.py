from typing import Literal, overload

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
    if workspace is not None:
        if workspace == "auto":
            return Workspace.auto()
        return workspace
    if executor is not None:
        if executor == "auto":
            return Executor.auto()
        return executor
    msg = "Either workspace or executor must be specified"
    raise ValueError(msg)
