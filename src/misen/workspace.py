from __future__ import annotations

from collections.abc import MutableMapping
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


# TODO: implement LocalWorkspace using DiskCache
# https://grantjenks.com/docs/diskcache/tutorial.html
