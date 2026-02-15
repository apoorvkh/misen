"""Helpers for task runtime execution flow."""

from __future__ import annotations

import tempfile
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from typing_extensions import assert_never

from misen.utils.hashes import ResultHash
from misen.utils.log_capture import capture_all_output
from misen.utils.nested_args import map_nested_leaves
from misen.utils.sentinels import ASSIGNED_RESOURCES, WORK_DIR

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from misen.task import Task
    from misen.utils.assigned_resources import AssignedResources
    from misen.utils.locks import LockLike
    from misen.workspace import Workspace

R = TypeVar("R")


def execute_task(
    task: Task[R],
    workspace: Workspace,
    dependency_results: dict[Task[Any], Any],
    assigned_resources: AssignedResources | None,
    job_id: str | None,
    lock_context: AbstractContextManager[LockLike | None, bool | None],
) -> R:
    """Execute task function under log capture and return result."""
    argument_resolver = _build_argument_resolver(
        task=task,
        workspace=workspace,
        dependency_results=dependency_results,
        assigned_resources=assigned_resources,
    )

    with lock_context:
        with workspace.open_task_log(task=task, mode="a", job_id=job_id, timestamp="current") as task_log:
            with capture_all_output(task_log, tee_to_stdout=True):
                args = (argument_resolver(value) for value in task.args)
                kwargs = {name: argument_resolver(value) for name, value in task.kwargs.items()}
                return task.func(*args, **kwargs)


def save_task_result(task: Task[Any], result: Any, workspace: Workspace) -> None:
    """Store task result metadata and cached payload when enabled."""
    match task.properties.index_by:
        case "task":
            index = task.resolved_hash(workspace=workspace)
        case "result":
            index = result
        case _:
            assert_never(task.properties.index_by)

    workspace.set_result_hash(task, ResultHash.from_object(index))

    if task.properties.cache:
        try:
            workspace.results[task] = result
        except Exception:
            del workspace.results[task]
            workspace.clear_result_hash(task=task)
            raise


def _build_argument_resolver(
    task: Task[Any],
    workspace: Workspace,
    dependency_results: dict[Task[Any], Any],
    assigned_resources: AssignedResources | None,
) -> Callable[[Any], Any]:
    """Return a function that resolves task/sentinel leaves before execution."""
    from misen.task import Task

    @cache
    def work_dir() -> Path:
        if task.properties.cache:
            return workspace.get_work_dir(task=task)

        resolved = task.resolved_hash(workspace=workspace).b32()
        return Path(tempfile.mkdtemp(prefix=f"misen-work-{resolved}-"))

    def argument_resolver(value: Any) -> Any:
        def leaf_resolver(leaf: Any) -> Any:
            if isinstance(leaf, Task):
                return dependency_results[leaf]
            if leaf is WORK_DIR:
                return work_dir()
            if leaf is ASSIGNED_RESOURCES:
                return assigned_resources
            return leaf

        return map_nested_leaves(value, leaf_resolver)

    return argument_resolver
