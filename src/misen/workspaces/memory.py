import tempfile
from functools import cache
from pathlib import Path

from ..task import Task
from ..workspace import (
    Workspace,
)

__all__ = ["MemoryWorkspace"]


class MemoryWorkspace(Workspace):
    def __init__(self, i: int):
        self.i = i
        super().__init__(
            resolved_hash_cache={},
            result_hash_cache={},
            result_cache={},
            log_store={},
        )

    @cache
    def _temp_workspace_dir() -> Path:
        d = Path(tempfile.gettempdir())
        d.mkdir(exist_ok=True)
        return d

    def get_work_dir(self, task: Task) -> Path:
        d = self._temp_workspace_dir() / str(task._resolved_hash(workspace=self))
        d.mkdir(exist_ok=True)
        return d
