from __future__ import annotations

import asyncio
import types
from typing import Any

from . import Workspace, Executor, LocalExecutor

# Make immutable


class Task:
    def __init__(self, func: types.FunctionType, task_decorator: dict, kwargs: dict[str, Any]):
        self.func = func
        self.task_decorator = task_decorator
        self.kwargs = kwargs

    def __hash__(self):
        if not hasattr(self, "__cached_hash__"):
            hash_uuid = self.task_decorator["uuid"] or self.func.__qualname__
            self.__cached_hash__ = hash((hash_uuid, self.kwargs))
        return self.__cached_hash__

    def __repr__(self):
        return f"Task(func={self.func.__module__}.{self.func.__qualname__}, kwargs={self.kwargs})"

    def run(self, workspace: Workspace, executor: Executor) -> asyncio.Future:
        """This submits the task graph to the executor and is non-blocking. Returns the result as a Future, which the user could wait for."""
        return executor.submit(task, workspace)

    def result(self, workspace: Workspace):
        """This runs incomplete Tasks locally and blocks for the result."""
        return self.run(workspace=workspace, executor=LocalExecutor()).result()


# openssl rand -base64 3
def task(
    uuid: str | None = None,
    cache: bool = True,
    version: str = "",
    exclude: set[str] = set(),
    defaults: dict[str, Any] = {},
):
    def decorator(f):
        f.__task__ = dict(
            uuid=uuid,
            cache=cache,
            version=version,
            exclude=exclude,
            defaults=defaults,
        )
        return f

    return decorator
