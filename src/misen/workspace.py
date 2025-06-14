from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module
from typing import TYPE_CHECKING, ClassVar, Literal

import msgspec
from msgspec import Struct

from .settings import Settings

if TYPE_CHECKING:
    from .caches import (
        ResolvedHashCacheABC,
        ResultCacheABC,
        ResultHashCacheABC,
    )
    from .task import Task


class WorkspaceConfig(Struct, kw_only=True):
    type: str | Literal["memory"] | None = None

    def default(self) -> WorkspaceConfig:
        from .workspaces.memory import MemoryWorkspaceConfig

        return MemoryWorkspaceConfig(i=10)

    def from_settings(self, settings: Settings | None = None) -> WorkspaceConfig:
        if settings is None:
            settings = Settings()

        if (workspace_toml := settings.toml_data.get("workspace", None)) is not None:
            config = msgspec.convert(workspace_toml, type=WorkspaceConfig)
            return msgspec.convert(workspace_toml, type=config.resolve_config_type())

        return self.default()

    def resolve_workspace_type(self) -> type[Workspace]:
        if self.type is None:
            return self.from_settings().resolve_workspace_type()

        match self.type:
            case "memory":
                from .workspaces.memory import MemoryWorkspace

                return MemoryWorkspace

        module, class_name = self.type.split(":", maxsplit=1)
        return getattr(import_module(module), class_name)

    def resolve_config_type(self) -> type[WorkspaceConfig]:
        return self.resolve_workspace_type().ConfigT

    def load_workspace(self, settings: Settings | None = None) -> Workspace:
        """Load the workspace based on the configuration."""
        if self.type is None:
            config = self.from_settings(settings=settings)
        else:
            config = self
        workspace_cls = config.resolve_workspace_type()
        return workspace_cls(config=config)


class Workspace(ABC):
    ConfigT: ClassVar[type[WorkspaceConfig]]

    resolved_hashes: ResolvedHashCacheABC
    result_hashes: ResultHashCacheABC
    results: ResultCacheABC

    @abstractmethod
    def __init__(self, config: WorkspaceConfig):
        raise NotImplementedError

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
