from ..workspace import (
    ResolvedHashCacheABC,
    ResultCacheABC,
    ResultHashCacheABC,
    Workspace,
)


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


class MemoryWorkspace(Workspace):
    def __init__(self, i: int):
        self.i = i
        self.resolved_hashes = MemoryResolvedHashCache(workspace=self)
        self.result_hashes = MemoryResultHashCache(workspace=self)
        self.results = MemoryResultCache(workspace=self)
