from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Literal

import msgspec
from msgspec import Struct

from .settings import Settings

if TYPE_CHECKING:
    from .caches import (
        AbstractResolvedHashCache,
        AbstractResultCache,
        AbstractResultHashCache,
    )
    from .task import Task


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

    @property
    def resolved_hashes(self) -> AbstractResolvedHashCache:
        raise NotImplementedError

    @property
    def result_hashes(self) -> AbstractResultHashCache:
        raise NotImplementedError

    @property
    def results(self) -> AbstractResultCache:
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
