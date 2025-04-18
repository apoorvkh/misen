from __future__ import annotations

from collections.abc import MutableMapping
from threading import Lock
from typing import Any

from .task import Task


class Workspace(MutableMapping[Task, Any]):
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError

    def __setitem__(self, key, item):
        raise NotImplementedError

    def __delitem__(self, key):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __contains__(self, key):
        raise NotImplementedError

    def get(self, key, default=None):
        raise NotImplementedError

    def get_logs(self, task):
        # TODO: A single task may be run multiple times and therefore have multiple logs.
        # How should we store and return logs?
        raise NotImplementedError

    def get_work_dir(self, task):
        """Return a directory where the task can store working files. E.g. to cache intermediate results."""
        raise NotImplementedError


# TODO: implement LocalWorkspace using LMDB


# for testing only
class TestWorkSpace(Workspace):
    def __init__(self):
        self.d = {}
        self.mtx = Lock()

    def __len__(self):
        with self.mtx:
            return len(self.d)

    def __getitem__(self, task: Task):
        with self.mtx:
            return self.d[task.__hash__()]

    def __setitem__(self, task: Task, item):
        with self.mtx:
            self.d[task.__hash__()] = item

    def __delitem__(self, task: Task):
        with self.mtx:
            del self.d[task.__hash__()]

    def __iter__(self):
        with self.mtx:
            return iter(self.d.items())

    def __contains__(self, task: Task):
        with self.mtx:
            return task.__hash__() in self.d
