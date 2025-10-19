import tempfile
from functools import cache
from pathlib import Path
from typing import Iterator

from ..task import Task
from ..workspace import (
    LogStoreABC,
    ResolvedHashCacheABC,
    ResultCacheABC,
    ResultHashCacheABC,
    Workspace,
)

__all__ = ["MemoryWorkspace"]


class MemoryResolvedHashCache(ResolvedHashCacheABC):
    def __init__(self, workspace: Workspace):
        super().__init__()
        self.workspace = workspace
        self.mapping = {}


class MemoryResultHashCache(ResultHashCacheABC):
    def __init__(self, workspace: Workspace):
        super().__init__()
        self.workspace = workspace
        self.mapping = {}


class MemoryResultCache(ResultCacheABC):
    def __init__(self, workspace: Workspace):
        super().__init__()
        self.workspace = workspace
        self.mapping = {}


class MemoryLogStore(LogStoreABC):
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self._store: dict[Task, dict[str, str]] = {}

    def __getitem__(self, key: Task) -> dict[str, str]:
        return self._store[key]

    def __setitem__(self, key: Task, value: dict[str, str]) -> None:
        self._store[key] = value

    def __delitem__(self, key: Task) -> None:
        del self._store[key]

    def __len__(self) -> int:
        return len(self._store)

    def __iter__(self) -> Iterator[Task]:
        return iter(self._store)

    def __contains__(self, key: object, /) -> bool:
        return key in self._store


class MemoryWorkspace(Workspace):
    def __init__(self, i: int):
        self.i = i
        super().__init__(
            resolved_hashes=MemoryResolvedHashCache(workspace=self),
            result_hashes=MemoryResultHashCache(workspace=self),
            results=MemoryResultCache(workspace=self),
            logs=MemoryLogStore(workspace=self),
        )

    @cache
    def _temp_workspace_dir() -> Path:
        d = Path(tempfile.gettempdir())
        d.mkdir(exist_ok=True)
        return d

    def get_work_dir(self, task: Task) -> Path:
        d = self._temp_workspace_dir() / str(self._resolved_hash(task=task))
        d.mkdir(exist_ok=True)
        return d
