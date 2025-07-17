from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from typing import MutableMapping, cast

import lmdb
from flufl.lock import Lock, NotLockedError  # pyright: ignore

from ..workspace import (
    Hash,
    ResolvedHashCacheABC,
    ResultCacheABC,
    ResultHash,
    ResultHashCacheABC,
    Workspace,
)


class LMDBMapping(MutableMapping[Hash, Hash]):
    def __init__(self, database_dir: Path):
        self.env = lmdb.open(str(database_dir), map_size=1_000_000_000, max_dbs=1)
        self.db = self.env.open_db()

    def __hash_encode(self, hash: Hash) -> bytes:
        return hash.to_bytes(8, byteorder="big")

    def __hash_decode(self, encoded_hash: bytes) -> Hash:
        return cast("Hash", int.from_bytes(encoded_hash, byteorder="big"))

    def __getitem__(self, key: Hash) -> Hash:
        with self.env.begin(db=self.db) as txn:
            value = txn.get(self.__hash_encode(key))
            if value is None:
                raise KeyError(f"Key {key} not found")
            return self.__hash_decode(value)

    def __setitem__(self, key: Hash, value: Hash):
        with self.env.begin(db=self.db, write=True) as txn:
            txn.put(self.__hash_encode(key), self.__hash_encode(value))

    def __delitem__(self, key: Hash):
        with self.env.begin(db=self.db, write=True) as txn:
            if not txn.delete(self.__hash_encode(key)):
                raise KeyError(f"Key {key} not found")

    def __iter__(self):
        with self.env.begin(db=self.db) as txn:
            cursor = txn.cursor()
            for encoded_key, _ in cursor:
                yield self.__hash_decode(encoded_key)

    def __len__(self):
        with self.env.begin(db=self.db) as txn:
            return txn.stat(db=self.db)["entries"]


class DiskResolvedHashCache(ResolvedHashCacheABC):
    def __init__(self, workspace: DiskWorkspace):
        super().__init__()
        self.workspace = workspace
        self.mapping = LMDBMapping(self.workspace.project_directory / "resolved_hash_cache")  # pyright: ignore


class DiskResultHashCache(ResultHashCacheABC):
    def __init__(self, workspace: DiskWorkspace):
        super().__init__()
        self.workspace = workspace
        self.mapping = LMDBMapping(workspace.workspace_directory / "result_hash_cache")  # pyright: ignore


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


class DiskWorkspace(Workspace):
    def __init__(self, directory: str):
        # open/create at specified directory or in CWD otherwise
        self.project_directory = Path(directory if directory is not None else "./")
        self.workspace_directory = self.project_directory / ".misen_workspace"

        # at the moment, only support one workspace per directory, TODO: support multiple
        new_workspace = False
        if not self.workspace_directory.exists():
            os.mkdir(self.workspace_directory)
            new_workspace = True

        self.resolved_hashes = DiskResolvedHashCache(workspace=self)
        self.result_hashes = DiskResultHashCache(workspace=self)
        self.results = DiskResultCache(workspace=self, new_workspace=new_workspace)
