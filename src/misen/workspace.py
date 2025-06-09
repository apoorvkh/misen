from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

import dill
import msgspec
from msgspec import Struct

from .utils.settings import Settings

if TYPE_CHECKING:
    from .task import Task


class ObjectHash(int):
    pass


class ResolvedHash(int):
    pass


class ResultHash(int):
    pass


class Workspace(Struct, kw_only=True, dict=True):
    type: str | Literal["memory"] | None = None

    @staticmethod
    def load(settings: Settings | None = None) -> Workspace:
        settings = settings or Settings()

        if "workspace" in settings.toml_data:
            workspace = msgspec.convert(settings.toml_data["workspace"], type=Workspace)
            workspace_cls: type[Workspace] | None = workspace._resolve_type()
            if workspace_cls is not None:
                return msgspec.convert(
                    settings.toml_data["workspace"],
                    type=workspace_cls,
                )

        # fallback to default
        from .workspaces.memory import MemoryWorkspace

        return MemoryWorkspace()

    def _resolve_type(self) -> type[Workspace] | None:
        if self.type is None:
            return None

        match self.type:
            case "memory":
                from .workspaces.memory import MemoryWorkspace

                return MemoryWorkspace

        module, class_name = self.type.split(":", maxsplit=1)
        return getattr(import_module(module), class_name)

    def __post_init__(self):
        self.resolved_hashes: dict[ObjectHash, ResolvedHash] = {}
        self.result_hashes: dict[ResolvedHash, ResultHash] = {}
        self.results: dict[ResultHash, bytes] = {}

    def get_resolved_hash(self, task: Task) -> ResolvedHash | None:
        return self.resolved_hashes.get(cast("ObjectHash", task.__hash__()))

    def set_resolved_hash(self, task: Task, h: ResolvedHash) -> None:
        self.resolved_hashes[cast("ObjectHash", task.__hash__())] = h

    def get_result_hash(self, task: Task) -> ResultHash | None:
        return self.result_hashes.get(task.__resolved_hash__(workspace=self))

    def set_result_hash(self, task: Task, h: ResultHash) -> None:
        self.result_hashes[task.__resolved_hash__(workspace=self)] = h

    def get_result(self, task: Task) -> tuple[Callable[[bytes], Any], bytes] | None:
        try:
            return dill.loads(self.results[task.__result_hash__(workspace=self)])
        except (KeyError, RuntimeError):
            return None

    def set_result(self, task: Task, result: tuple[Callable[[bytes], Any], bytes]) -> None:
        self.results[task.__result_hash__(workspace=self)] = dill.dumps(result)

    def is_cached(self, task: Task) -> bool:
        """Check if the result of the task is cached."""
        try:
            return (
                task.properties.cache_result
                and task.__result_hash__(workspace=self) in self.results.keys()
            )
        except RuntimeError:
            return False

    # def get_logs(self, task):
    #     # TODO: A single task may be run multiple times and therefore have multiple logs.
    #     # How should we store and return logs?
    #     raise NotImplementedError

    # def get_work_dir(self, task):
    #     """Return a directory where the task can store working files. E.g. to cache intermediate results."""
    #     raise NotImplementedError
