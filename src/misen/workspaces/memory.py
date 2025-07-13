from ..workspace import (
    ResolvedHashCacheABC,
    ResultCacheABC,
    ResultHashCacheABC,
    Workspace,
    WorkspaceConfig,
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


class MemoryWorkspaceConfig(WorkspaceConfig):
    i: int


class MemoryWorkspace(Workspace):
    @staticmethod
    def config_type() -> type[WorkspaceConfig]:
        return MemoryWorkspaceConfig

    def __init__(self, config: MemoryWorkspaceConfig):
        super().__init__(config=config)
        self.resolved_hashes = MemoryResolvedHashCache(workspace=self)
        self.result_hashes = MemoryResultHashCache(workspace=self)
        self.results = MemoryResultCache(workspace=self)
