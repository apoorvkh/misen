from __future__ import annotations

from types import FunctionType, MappingProxyType
from typing import Any, Callable
from dataclasses import dataclass
import inspect

from .workspace import Workspace

__all__ = ["Task", "task"]


@dataclass(frozen=True)
class TaskProperties:
    """Dataclass for task properties. Attributes are immutable so that Task.__hash__ will be constant and cacheable."""

    uuid: str | None
    cache: bool
    version: str
    exclude: frozenset[str]
    defaults: MappingProxyType


@dataclass(frozen=True)
class Task:
    """Class for wrapping a function as a task. kwargs"""

    func: FunctionType
    kwargs: MappingProxyType  # immutable view for dict
    properties: TaskProperties

    @staticmethod
    def factory(func: FunctionType) -> Callable[..., Task]:
        def _wrapper(*args, **kwargs):
            # TODO: this is deprecated; switch to signature.bind
            callargs = inspect.getcallargs(func, *args, **kwargs)
            return Task(func=func, kwargs=MappingProxyType(callargs), properties=func.__task__)  # pyright: ignore [reportAttributeAccessIssue]

        return _wrapper

    def __hash__(self):
        """Hashing function for task instance. Hash is cached (assuming this object and its attributes are immutable)."""
        if self.__dict__.get("__cached_hash__") is None:
            task_id = self.properties.uuid or self.func.__qualname__
            # TODO: handle self.properties.exclude and self.properties.defaults in kwargs
            # like SKIP_DEFAULT_ARGUMENTS and SKIP_ID_ARGUMENTS in
            # https://ai2-tango.readthedocs.io/en/latest/api/components/step.html#tango.step.Step.SKIP_DEFAULT_ARGUMENTS
            self.__dict__["__cached_hash__"] = hash((task_id, self.kwargs))
        return self.__dict__["__cached_hash__"]

    def __repr__(self):
        cached_hash = self.__dict__.get("__cached_hash__")
        return f"Task(func={self.func.__module__}.{self.func.__qualname__}, kwargs={self.kwargs}, cached_hash={cached_hash})"

    def result(self, workspace: Workspace | None = None):
        if self.properties.cache and workspace is not None and self in workspace:
            return workspace.result(self)

        execution_result = self.func(
            **{k: (v.result() if isinstance(v, Task) else v) for k, v in self.kwargs.items()}
        )
        if self.properties.cache and workspace is not None:
            workspace[self] = execution_result
        return execution_result


def task(
    uuid: str | None = None,  # openssl rand -base64 3
    cache: bool = False,
    version: str = "",
    exclude: set[str] = set(),
    defaults: dict[str, Any] = {},
):
    def decorator(f):
        f.__task__ = TaskProperties(
            uuid=uuid,
            cache=cache,
            version=version,
            exclude=frozenset(exclude),
            defaults=MappingProxyType(defaults),
        )
        return f

    return decorator
