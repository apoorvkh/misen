from __future__ import annotations
from typing import Any
import types


class Task:
    def __init__(self, func: types.FunctionType, task_decorator: dict, kwargs: dict[str, Any]):
        self.func = func
        self.task_decorator = task_decorator
        self.kwargs = kwargs

    # __hash__

    def __repr__(self):
        return f"Task(func={self.func.__module__}.{self.func.__qualname__}, kwargs={self.kwargs})"


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


class Workspace:
    pass


class Executor:
    def run(self, a, workspace: Workspace):
        pass
