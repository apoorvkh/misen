from ..task import Task
from ..workspace import Workspace


class MemoryWorkspace(Workspace, kw_only=True):
    type = "memory"

    def __post_init__(self):
        self.cache_hash = {}
        self.cache_value = {}

    def __len__(self):
        return len(self.cache_value)

    def __getitem__(self, task: Task):
        return self.cache_value[task.__hash__()]

    def __setitem__(self, task: Task, item):
        self.cache_value[task.__hash__()] = item

    def __delitem__(self, task: Task):
        del self.cache_value[task.__hash__()]

    def __iter__(self):
        return iter(self.cache_value.items())

    def __contains__(self, task: Task):
        return task.__hash__() in self.cache_value
