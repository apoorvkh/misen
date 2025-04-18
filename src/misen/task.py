from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Generic, ParamSpec, TypeVar

from .utils.cached_property import cached_property
from .utils.det_hash import deterministic_hashing

if TYPE_CHECKING:
    from .executor import Executor
    from .workspace import Workspace

__all__ = ["Task", "task"]

P = ParamSpec("P")
R = TypeVar("R")


def task(
    uuid: str | None = None,  # openssl rand -base64 3
    cache: bool = False,
    version: int = 0,
    exclude: set[str] = set(),
    defaults: dict[str, Any] = {},
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    # Currently id is set as uuid (if specified) or f.__qualname__ (i.e. the function name)
    # It is nice to ignore f.__module__ so that this function can be moved into different files
    # TODO: In the future, it would also be nice to "import" tasks from other libraries
    # In that case, we might want to consider the package it originates from?

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        setattr(
            func,
            "__task__",
            TaskProperties(
                id=(uuid or func.__qualname__),
                cacheable=cache,
                version=version,
                exclude=frozenset(exclude),
                defaults=MappingProxyType(defaults),
            ),
        )
        return func

    return decorator


@dataclass(frozen=True)
class TaskProperties:
    """Dataclass for task properties. Attributes are immutable so that Task.__hash__ will be constant and cacheable."""

    id: str
    cacheable: bool = False
    version: int = 0
    exclude: frozenset[str] = frozenset()
    defaults: MappingProxyType = MappingProxyType({})


class Task(Generic[R]):
    def __init__(self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.properties = getattr(
            self.func,
            "__task__",
            TaskProperties(id=f"{func.__module__}.{func.__qualname__}"),
        )

        # compute and cache the hash
        with deterministic_hashing():
            self.__hash__()

    @property
    def T(self) -> R:
        return self  # type: ignore

    def __repr__(self):
        return f"Task(func={self.func.__module__}.{self.func.__qualname__}, args={self.args}, kwargs={self.kwargs}, hash={self.__hash__()})"

    def _resolve_args(
        self, workspace: Workspace | None = None, ensure_cached: bool = False
    ) -> tuple:
        args = (
            v._run(workspace=workspace, ensure_cached=ensure_cached) if isinstance(v, Task) else v
            for v in self.args
        )

        kwargs = {
            k: (
                v._run(workspace=workspace, ensure_cached=ensure_cached)
                if isinstance(v, Task)
                else v
            )
            for k, v in self.kwargs.items()
        }

        return args, kwargs  # pyright: ignore

    def is_cached(self, workspace: Workspace | None = None) -> bool:
        if workspace is None:
            return False
        if self.properties.cacheable:
            return self in workspace
        return all((v.is_cached(workspace) for v in self.kwargs.values() if isinstance(v, Task)))

    def _run(
        self,
        workspace: Workspace | None = None,
        ensure_cached: bool = False,
        ensure_deps_cached: bool = False,
    ) -> R:
        # run self.func
        # expect all subtasks to be cached, error if not

        is_cached = self.is_cached(workspace=workspace)

        if ensure_cached and self.properties.cacheable:
            assert is_cached, "ensure_cached=True: expecting cached Task"

        if (
            self.properties.cacheable
            and workspace is not None
            and is_cached
        ):
            return workspace[self]

        args, kwargs = self._resolve_args(workspace=workspace, ensure_cached=ensure_deps_cached)

        # TODO: fix typing? by returning tuple[P.args, P.kwargs] in _resolve_args
        execution_result = self.func(*args, **kwargs)  # pyright: ignore

        if self.properties.cacheable and workspace is not None:
            workspace[self] = execution_result

        return execution_result

    def run(self, workspace: Workspace | None = None, executor: Executor | None = None) -> Future:
        """
        Submit task to executor to fully execute the task graph.
        """
        # TODO:
        # if executor is None:
        #     executor = ...

        return executor.submit(task=self, workspace=workspace)  # type: ignore

    def result(self, workspace: Workspace | None = None) -> R:
        """
        Immediately get the result of cacheable tasks, or compute it if it's uncacheable.
        """
        return self._run(workspace=workspace, ensure_cached=True, ensure_deps_cached=True)

    @property
    def work_dir(self, workspace: Workspace | None = None):
        # TODO: how can we pass a Task.work_dir to Task.from(func(work_dir))
        # this is cyclic. we probably need a special object e.g. misen.WORK_DIR that is ignored by the hash and is realized as Task.work_dir.
        # we could do something similar to pass a Task.logger to a task
        if workspace is None:
            return None
        return workspace.get_work_dir(self)

    def __hash__(self):
        """Hashing function for task instance. Hash is cached (assuming this object and its attributes are immutable)."""
        # TODO: handle self.properties.exclude and self.properties.defaults in kwargs
        # like SKIP_DEFAULT_ARGUMENTS and SKIP_ID_ARGUMENTS in
        # https://ai2-tango.readthedocs.io/en/latest/api/components/step.html#tango.step.Step.SKIP_DEFAULT_ARGUMENTS
        h = hash((self.properties.id, self.args, self.kwargs))
        return int(h, 16)  # is this necessary?

    def __getattribute__(self, name):
        if name == "__hash__":
            return cached_property(self, type(self).__hash__, key="__cached_hash__")
        return super().__getattribute__(name)
