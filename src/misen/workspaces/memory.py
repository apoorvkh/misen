from ..workspace import Workspace


class MemoryWorkspace(Workspace, kw_only=True):
    type = "memory"

    # def __post_init__(self):
    #     self.d = {}
    #     self.mtx = Lock()

    # def __len__(self):
    #     with self.mtx:
    #         return len(self.d)

    # def __getitem__(self, task: Task):
    #     with self.mtx:
    #         return self.d[task.__hash__()]

    # def __setitem__(self, task: Task, item):
    #     with self.mtx:
    #         self.d[task.__hash__()] = item

    # def __delitem__(self, task: Task):
    #     with self.mtx:
    #         del self.d[task.__hash__()]

    # def __iter__(self):
    #     with self.mtx:
    #         return iter(self.d.items())

    # def __contains__(self, task: Task):
    #     with self.mtx:
    #         return task.__hash__() in self.d
