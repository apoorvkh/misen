from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import FunctionType, MappingProxyType
from typing import TYPE_CHECKING, Any, Callable

from .utils.det_hash import deterministic_hashing

if TYPE_CHECKING:
    from .workspace import Workspace

__all__ = ["Task", "task"]


@dataclass(frozen=True)
class TaskProperties:
    """Dataclass for task properties. Attributes are immutable so that Task.__hash__ will be constant and cacheable."""

    id: str
    cacheable: bool
    version: str
    exclude: frozenset[str]
    defaults: MappingProxyType


@dataclass(frozen=True)
class Task:
    """Class for wrapping a function as a task. kwargs"""

    func: FunctionType
    kwargs: MappingProxyType  # immutable view for dict
    properties: TaskProperties

    def __post_init__(self) -> None:
        # compute and cache the hash
        with deterministic_hashing():
            self.__hash__()

    # TODO: is this caching thread safe?
    def __hash__(self):
        """Hashing function for task instance. Hash is cached (assuming this object and its attributes are immutable)."""
        if not hasattr(self, "__cached_hash__"):
            # TODO: handle self.properties.exclude and self.properties.defaults in kwargs
            # like SKIP_DEFAULT_ARGUMENTS and SKIP_ID_ARGUMENTS in
            # https://ai2-tango.readthedocs.io/en/latest/api/components/step.html#tango.step.Step.SKIP_DEFAULT_ARGUMENTS
            h = hash((self.properties.id, self.kwargs))
            object.__setattr__(self, "__cached_hash__", h)
        return self.__getattribute__("__cached_hash__")

    def __repr__(self):
        return f"Task(func={self.func.__module__}.{self.func.__qualname__}, kwargs={self.kwargs}, hash={self.__hash__()})"

    @staticmethod
    def _get_factory(func: FunctionType) -> Callable[..., Task]:
        """If you pass a function object to this method, it will return a factory function. If you call that function with arguments, it will return a Task object.

        This is useful to replace a function object in globals(). A call to that function will then return a Task, instead of executing the function.
        """

        def _factory(*args, **kwargs):
            # TODO: this is deprecated; switch to signature.bind
            callargs = inspect.getcallargs(func, *args, **kwargs)
            return Task(func=func, kwargs=MappingProxyType(callargs), properties=func.__task__)  # pyright: ignore [reportAttributeAccessIssue]

        return _factory

    def is_cached(self, workspace: Workspace | None = None) -> bool:
        # TODO: need a better name for this: this is basically "are sufficient computations pre-cached?"
        if workspace is None:
            return False
        if self.properties.cacheable:
            return self in workspace
        return all((v.is_cached(workspace) for v in self.kwargs.values() if isinstance(v, Task)))

    def result(self, workspace: Workspace | None = None, ensure_cached: bool = True):
        """This function directly computes the task graph and is blocking. Un-cached tasks will be executed.
        If ensure_cached is True, this will raise an error if any dependent task is cachable but not cached.
        """

        # TODO: should we actually submit to an executor here and block for the result?

        if ensure_cached and not self.is_cached(workspace):
            raise ValueError(f"Task {self} is not sufficiently cached but ensure_cached is True.")
        # is_cached is recursive, so subsequent calls to Task.result can set ensure_cached=False

        if self.properties.cacheable and workspace is not None and self in workspace:
            return workspace[self]

        execution_result = self.func(
            **{
                k: (v.result(ensure_cached=False) if isinstance(v, Task) else v)
                for k, v in self.kwargs.items()
            }
        )
        if self.properties.cacheable and workspace is not None:
            workspace[self] = execution_result
        return execution_result

    @property
    def work_dir(self, workspace: Workspace | None = None):
        # TODO: how can we pass a Task.work_dir to Task.from(func(work_dir))
        # this is cyclic. we probably need a special object e.g. misen.WORK_DIR that is ignored by the hash and is realized as Task.work_dir.
        # we could do something similar to pass a Task.logger to a task
        if workspace is None:
            return None
        return workspace.get_work_dir(self)


def task(
    uuid: str | None = None,  # openssl rand -base64 3
    cache: bool = False,
    version: str = "",
    exclude: set[str] = set(),
    defaults: dict[str, Any] = {},
):
    # Currently id is set as uuid (if specified) or f.__qualname__ (i.e. the function name)
    # It is nice to ignore f.__module__ so that this function can be moved into different files
    # TODO: In the future, it would also be nice to "import" tasks from other libraries
    # In that case, we should also consider the package it originates from

    def decorator(f):
        f.__task__ = TaskProperties(
            id=(uuid or f.__qualname__),
            cacheable=cache,
            version=version,
            exclude=frozenset(exclude),
            defaults=MappingProxyType(defaults),
        )
        return f

    return decorator
