from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from typing import MutableMapping, TypeVar, cast

import lmdb
from flufl.lock import Lock, NotLockedError  # pyright: ignore

from ..task import Hash, ResolvedTaskHash, ResultHash, TaskHash
from ..workspace import (
    Workspace,
)

KT = TypeVar("KT", bound=Hash)
VT = TypeVar("VT", bound=Hash)


class LMDBMapping(MutableMapping[KT, VT]):
    def __init__(self, database_dir: Path):
        self.env = lmdb.open(str(database_dir), map_size=1_000_000_000, max_dbs=1)
        self.db = self.env.open_db()

    def __hash_encode(self, hash: Hash) -> bytes:
        return hash.to_bytes(8, byteorder="big")

    def __hash_decode(self, encoded_hash: bytes, typ: type[Hash]) -> Hash:
        return typ(int.from_bytes(encoded_hash, byteorder="big"))

    def __getitem__(self, key: KT) -> VT:
        with self.env.begin(db=self.db) as txn:
            value = txn.get(self.__hash_encode(key))
            if value is None:
                raise KeyError(f"Key {key} not found")
            return cast(VT, self.__hash_decode(value, self.__orig_class__.__args__[1]))  # type: ignore

    def __setitem__(self, key: KT, value: VT):
        with self.env.begin(db=self.db, write=True) as txn:
            txn.put(self.__hash_encode(key), self.__hash_encode(value))

    def __delitem__(self, key: KT):
        with self.env.begin(db=self.db, write=True) as txn:
            if not txn.delete(self.__hash_encode(key)):
                raise KeyError(f"Key {key} not found")

    def __iter__(self):
        with self.env.begin(db=self.db) as txn:
            cursor = txn.cursor()
            for encoded_key, _ in cursor:
                yield cast("KT", self.__hash_decode(encoded_key, self.__orig_class__.__args__[0]))  # type: ignore

    def __len__(self):
        with self.env.begin(db=self.db) as txn:
            return txn.stat(db=self.db)["entries"]


# class DiskResolvedHashCache(MutableMapping[TaskHash, ResolvedTaskHash]):
#     def __init__(self, workspace: DiskWorkspace):
#         super().__init__()
#         self.workspace = workspace
#         self.mapping = LMDBMapping(self.workspace.workspace_directory / "resolved_hash_cache")  # pyright: ignore


# class DiskResultHashCache(MutableMapping[ResolvedTaskHash, ResultHash]):
#     def __init__(self, workspace: DiskWorkspace):
#         super().__init__()
#         self.workspace = workspace
#         self.mapping = LMDBMapping(self.workspace.workspace_directory / "result_hash_cache")  # pyright: ignore


class DiskResultCacheMapping(MutableMapping[ResultHash, bytes]):
    def __init__(self, result_cache_directory: Path):
        self.result_cache_directory = result_cache_directory

    def __aquire_lock(self, filename: Path) -> Lock:
        item_lock_filename = filename.name + ".lock"

        lock = Lock(item_lock_filename, lifetime=timedelta(hours=1))
        lock.lock()

        return lock

    def __get_key_filename(self, key: ResultHash) -> Path:
        return self.result_cache_directory / f"result_{str(key)}"

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
        for path in self.result_cache_directory.iterdir():
            if path.is_file() and path.name.startswith("result_"):
                yield ResultHash(path.name[7:])

    def __len__(self) -> int:
        return len(
            list(filter(lambda x: x.startswith("result_"), os.listdir(self.result_cache_directory)))
        )


# class DiskResultCache(MutableMapping[ResultHash, bytes]):
#     def __init__(self, workspace: DiskWorkspace, new_workspace: bool = False):
#         super().__init__()
#         self.workspace = workspace
#         self.result_cache_directory = workspace.workspace_directory / "result_cache"

#         if new_workspace:
#             os.mkdir(self.result_cache_directory)

#         self.mapping = DiskResultCacheMapping(self)


class DiskWorkspace(Workspace):
    def __init__(self, directory: str):
        # open/create at specified directory or in CWD otherwise
        self.project_directory = Path(directory if directory is not None else "./")
        self.workspace_directory = self.project_directory / ".misen_workspace"

        # at the moment, only support one workspace per directory, TODO: support multiple
        if not self.workspace_directory.exists():
            os.mkdir(self.workspace_directory)

        super().__init__(
            resolved_hash_cache=LMDBMapping[TaskHash, ResolvedTaskHash](
                database_dir=self.workspace_directory / "resolved_hash_cache"
            ),
            result_hash_cache=LMDBMapping[ResolvedTaskHash, ResultHash](
                database_dir=self.workspace_directory / "result_hash_cache"
            ),
            result_cache=DiskResultCacheMapping(
                result_cache_directory=self.workspace_directory / "result_cache"
            ),
            log_store={},
        )

        # self.resolved_hashes = DiskResolvedHashCache(workspace=self)
        # self.result_hashes = DiskResultHashCache(workspace=self)
        # self.results = DiskResultCache(workspace=self, new_workspace=new_workspace)
