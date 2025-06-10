from ..caches import (
    AbstractResolvedHashCache,
    AbstractResultCache,
    AbstractResultHashCache,
)
from ..workspace import Workspace


class MemoryResolvedHashCache(AbstractResolvedHashCache):
    def __init__(self, workspace: Workspace):
        super().__init__()
        self.workspace = workspace
        self.mapping = {}


class MemoryResultHashCache(AbstractResultHashCache):
    def __init__(self, workspace: Workspace):
        super().__init__()
        self.workspace = workspace
        self.mapping = {}


class MemoryResultCache(AbstractResultCache):
    def __init__(self, workspace: Workspace):
        super().__init__()
        self.workspace = workspace
        self.mapping = {}


class MemoryWorkspace(Workspace, kw_only=True):
    type = "memory"

    def __post_init__(self):
        self._resolved_hashes = MemoryResolvedHashCache(workspace=self)
        self._result_hashes = MemoryResultHashCache(workspace=self)
        self._results = MemoryResultCache(workspace=self)

    @property
    def resolved_hashes(self) -> MemoryResolvedHashCache:
        return self._resolved_hashes

    @property
    def result_hashes(self) -> MemoryResultHashCache:
        return self._result_hashes

    @property
    def results(self) -> MemoryResultCache:
        return self._results
