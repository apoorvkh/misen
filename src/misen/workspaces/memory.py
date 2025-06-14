from ..caches import (
    ResolvedHashCacheABC,
    ResultCacheABC,
    ResultHashCacheABC,
)
from ..workspace import Workspace, WorkspaceConfig


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


class MemoryWorkspaceConfig(WorkspaceConfig):
    type = "memory"
    i: int


class MemoryWorkspace(Workspace):
    ConfigT = MemoryWorkspaceConfig

    def __init__(self, config: MemoryWorkspaceConfig):
        self.config = config
        self.resolved_hashes = MemoryResolvedHashCache(workspace=self)
        self.result_hashes = MemoryResultHashCache(workspace=self)
        self.results = MemoryResultCache(workspace=self)
