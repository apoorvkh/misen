from __future__ import annotations

from importlib import import_module

from misen.utils.from_params import FromParamsABC

_builtin_workspaces = {
    "memory": "misen.workspaces.memory:MemoryWorkspace",
}


class Workspace(FromParamsABC):#, MutableMapping[Task, Any]):
    type: str

    @classmethod
    def from_params(cls, params: dict) -> Workspace:
        workspace_type = cls.from_params(params).type
        workspace_type = _builtin_workspaces.get(workspace_type, workspace_type)

        module, class_name = workspace_type.split(":", maxsplit=1)
        workspace_class = getattr(import_module(module), class_name)
        assert isinstance(workspace_class, type) and issubclass(workspace_class, Workspace)

        return workspace_class.from_params(params)

    @classmethod
    def default_params(cls) -> dict:
        return {"type": "memory"}

    @classmethod
    def toml_key(cls) -> str:
        return "workspace"

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
