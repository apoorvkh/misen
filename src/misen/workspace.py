from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .settings import ConfigABC, TargetABC

if TYPE_CHECKING:
    from .caches import (
        ResolvedHashCacheABC,
        ResultCacheABC,
        ResultHashCacheABC,
    )
    from .task import Task


class WorkspaceConfig(ConfigABC["WorkspaceConfig", "Workspace"], kw_only=True):
    type: str | Literal["memory"] | None = None

    @staticmethod
    def settings_key() -> str:
        return "workspace"

    def default(self) -> WorkspaceConfig:
        from .workspaces.memory import MemoryWorkspaceConfig

        return MemoryWorkspaceConfig(i=10)

    def resolve_target_type(self) -> type[Workspace]:
        match self.type:
            case "memory":
                from .workspaces.memory import MemoryWorkspace

                return MemoryWorkspace
        return super().resolve_target_type()


class Workspace(TargetABC[WorkspaceConfig]):
    resolved_hashes: ResolvedHashCacheABC
    result_hashes: ResultHashCacheABC
    results: ResultCacheABC

    def is_cached(self, task: Task) -> bool:
        """Check if the result of the task is cached."""
        return task.properties.cache_result and task in self.results

    # def get_logs(self, task):
    #     # TODO: A single task may be run multiple times and therefore have multiple logs.
    #     # How should we store and return logs?
    #     raise NotImplementedError

    # def get_work_dir(self, task):
    #     """Return a directory where the task can store working files. E.g. to cache intermediate results."""
    #     raise NotImplementedError
