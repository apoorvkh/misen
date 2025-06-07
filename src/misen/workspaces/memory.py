from ..task import Task
from ..workspace import Workspace


class MemoryWorkspace(Workspace, kw_only=True):
    type = "memory"

    def __post_init__(self):
        self.resolved_hashes = {}
        self.result_hashes = {}
        self.results = {}
