from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from typing import MutableMapping

from flufl.lock import Lock, NotLockedError  # pyright: ignore

from ..caches import (
    ResolvedHashCacheABC,
    ResultCacheABC,
    ResultHash,
    ResultHashCacheABC,
)
from ..workspace import Workspace, WorkspaceConfig


class MemoryResolvedHashCache(ResolvedHashCacheABC):
    def __init__(self, workspace: Workspace, new_workspace: bool = False):
        super().__init__()
        self.workspace = workspace
        self.mapping = {}


class MemoryResultHashCache(ResultHashCacheABC):
    def __init__(self, workspace: Workspace, new_workspace: bool = False):
        super().__init__()
        self.workspace = workspace
        self.mapping = {}


class DiskResultCacheMapping(MutableMapping[ResultHash, bytes]):
    def __init__(self, cache: DiskResultCache):
        self.cache = cache

    def __aquire_lock(self, filename: Path) -> Lock:
        item_lock_filename = filename / ".lock"

        lock = Lock(str(item_lock_filename), lifetime=timedelta(hours=1))
        lock.lock()

        return lock

    def __get_key_filename(self, key: ResultHash) -> Path:
        return self.cache.result_cache_directory / f"result_{str(key)}"

    def __setitem__(self, key: ResultHash, value: bytes):
        item_filename = self.__get_key_filename(key)

        lock = self.__aquire_lock(item_filename)

        assert lock.is_locked, "Lock somehow lost immediately"

        with open(item_filename, "wb") as f:
            f.write(value)

        # necessary, or not?
        try:
            lock.unlock()
        except NotLockedError:
            raise Exception("Lock lost during writing")

    def __getitem__(self, key: ResultHash) -> bytes:
        item_filename = self.__get_key_filename(key)

        lock = self.__aquire_lock(item_filename)

        try:
            with open(item_filename, "rb") as f:
                value = f.read()
        except FileNotFoundError:
            raise KeyError(f"No such key {key} in cache.")

        # necessary, or not?
        try:
            lock.unlock()
        except NotLockedError:
            raise Exception("Lock lost during reading")

        return value

    def __delitem__(self, key: ResultHash):
        item_filename = self.__get_key_filename(key)

        lock = self.__aquire_lock(item_filename)
        try:
            os.remove(item_filename)
        except FileNotFoundError:
            raise KeyError(f"No such key {key} in cache.")

        # necessary, or not?
        try:
            lock.unlock()
        except NotLockedError:
            raise Exception("Lock lost during delete")

        # do not remove the lock file for now, it has a race condition

    def __iter__(self):
        for path in self.cache.result_cache_directory.iterdir():
            if path.is_file() and path.name.startswith("result_"):
                yield ResultHash(path.name[7:])

    def __len__(self) -> int:
        return len(
            list(
                filter(
                    lambda x: x.startswith("result_"), os.listdir(self.cache.result_cache_directory)
                )
            )
        )


class DiskResultCache(ResultCacheABC):
    def __init__(self, workspace: DiskWorkspace, new_workspace: bool = False):
        super().__init__()
        self.workspace = workspace
        self.result_cache_directory = workspace.workspace_directory / "result_cache"

        if new_workspace:
            os.mkdir(self.result_cache_directory)

        self.mapping = DiskResultCacheMapping(self)


class DiskWorkspaceConfig(WorkspaceConfig):
    type = "disk"
    directory: str | None


class DiskWorkspace(Workspace):
    @staticmethod
    def config_type() -> type[WorkspaceConfig]:
        return DiskWorkspaceConfig

    def __init__(self, config: DiskWorkspaceConfig):
        super().__init__(config=config)

        # open/create at specified directory or in CWD otherwise
        self.project_directory = Path(config.directory if config.directory is not None else "./")
        self.workspace_directory = self.project_directory / ".misen_workspace"

        # at the moment, only support one workspace per directory, TODO: support multiple
        new_workspace = False
        if not self.workspace_directory.exists():
            os.mkdir(self.workspace_directory)
            new_workspace = True

        self.resolved_hashes = MemoryResolvedHashCache(workspace=self, new_workspace=new_workspace)
        self.result_hashes = MemoryResultHashCache(workspace=self, new_workspace=new_workspace)
        self.results = DiskResultCache(workspace=self, new_workspace=new_workspace)
