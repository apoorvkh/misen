from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Generic, ParamSpec, TypeVar

from .utils.det_hash import deterministic_hashing

if TYPE_CHECKING:
    from .executor import Executor
    from .workspace import Workspace

__all__ = ["Task", "task"]


@dataclass(frozen=True)
class TaskProperties:
    """Dataclass for task properties. Attributes are immutable so that Task.__hash__ will be constant and cacheable."""

    id: str
    cacheable: bool = False
    version: int = 0
    exclude: frozenset[str] = frozenset()
    defaults: MappingProxyType = MappingProxyType({})


P = ParamSpec("P")
R = TypeVar("R")


class Task(Generic[R]):
    def __init__(self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        if hasattr(self.func, "__task__"):
            self.properties = getattr(self.func, "__task__")
        else:
            self.properties = TaskProperties(id=func.__qualname__)

        # compute and cache the hash
        with deterministic_hashing():
            self.__hash__()

    def as_argument(self) -> R:
        return self  # type: ignore

    # TODO: is this caching thread safe?
    def __hash__(self):
        """Hashing function for task instance. Hash is cached (assuming this object and its attributes are immutable)."""
        if not hasattr(self, "__cached_hash__"):
            # TODO: handle self.properties.exclude and self.properties.defaults in kwargs
            # like SKIP_DEFAULT_ARGUMENTS and SKIP_ID_ARGUMENTS in
            # https://ai2-tango.readthedocs.io/en/latest/api/components/step.html#tango.step.Step.SKIP_DEFAULT_ARGUMENTS
            h = hash((self.properties.id, self.args, self.kwargs))
            object.__setattr__(self, "__cached_hash__", h)
        return int(self.__getattribute__("__cached_hash__"), 16)  # is this necessary?

    def __repr__(self):
        return f"Task(func={self.func.__module__}.{self.func.__qualname__}, args={self.args}, kwargs={self.kwargs}, hash={self.__hash__()})"

    def is_cached(self, workspace: Workspace | None = None) -> bool:
        # TODO: need a better name for this: this is basically "are sufficient computations pre-cached?"
        if workspace is None:
            return False
        if self.properties.cacheable:
            return self in workspace
        return all((v.is_cached(workspace) for v in self.kwargs.values() if isinstance(v, Task)))

    def run(self, workspace: Workspace | None = None, executor: Executor | None = None):
        # TODO: executor cannot be None, but this is just until we have a LocalExecutor

        return executor.submit(self, workspace)  # type: ignore

    def result(
        self,
        workspace: Workspace | None = None,
        ensure_cached: bool = False,
        ensure_deps_cached: bool = True,
    ):
        """This function directly computes the task graph and is blocking. Un-cached tasks will be executed.
        If ensure_cached is True, this will raise an error if any dependent task is cachable but not cached.
        """

        # TODO: should we actually submit to an executor here and block for the result?

        if ensure_cached and not self.is_cached(workspace):
            raise ValueError(f"Task {self} is not sufficiently cached but ensure_cached is True.")
        # is_cached is recursive, so subsequent calls to Task.result can set ensure_cached=False

        if self.properties.cacheable and workspace is not None and self in workspace:
            return workspace[self]

        # TODO: process Tasks and special objects
        execution_result = self.func(*self.args, **self.kwargs)

        # execution_result = self.func(
        #     *(
        #         v.result(
        #             workspace=workspace,
        #             ensure_cached=ensure_deps_cached,
        #             ensure_deps_cached=ensure_deps_cached,
        #         )
        #         if isinstance(v, Task)
        #         else v
        #         for v in self.args
        #     ),
        #     **{
        #         k: (
        #             v.result(
        #                 workspace=workspace,
        #                 ensure_cached=ensure_deps_cached,
        #                 ensure_deps_cached=ensure_deps_cached,
        #             )
        #             if isinstance(v, Task)
        #             else v
        #         )
        #         for k, v in self.kwargs.items()
        #     }
        # )

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
    version: int = 0,
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
