from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from typing import Generator, Generic, MutableMapping, TypeVar

import lmdb
from flufl.lock._lockfile import Lock, NotLockedError

from ..task import Hash, ResolvedTaskHash, ResultHash, Task, TaskHash
from ..workspace import (
    Workspace,
)

KT = TypeVar("KT", bound=Hash)
VT = TypeVar("VT", bound=Hash)


class LMDBMapping(Generic[KT, VT], MutableMapping[KT, VT]):
    _key_type: type[KT]
    _value_type: type[VT]

    def __class_getitem__(cls, item: tuple[type[KT], type[VT]]):
        """Subclass dynamically bound to KT, VT."""
        key_t, val_t = item
        return type(
            f"{cls.__name__}[{key_t.__name__},{val_t.__name__}]",
            (cls,),
            {
                "_key_type": key_t,
                "_value_type": val_t,
                "__module__": cls.__module__,
            },
        )

    def __init__(self, database_path: Path):
        if not hasattr(self, "_key_type") or not hasattr(self, "_value_type"):
            raise TypeError("Construct as LMDBMapping[KeyType, ValueType](...)")

        # TODO: implement the lock (using flufl.lock)

        self.env = lmdb.Environment(
            database_path,
            subdir=False,
            lock=False,
            map_size=2**28,  # 256 MiB
            max_dbs=1,
        )

    def __getitem__(self, key: KT) -> VT:
        with self.env.begin() as txn:
            v: bytes | None = txn.get(key.encode(), default=None)  # type: ignore
            if v is None:
                raise KeyError(key)
            return self._value_type.decode(v)

    def __setitem__(self, key: KT, value: VT) -> None:
        _key, _value = key.encode(), value.encode()
        with self.env.begin(write=True) as txn:
            txn.put(_key, _value)

    def __delitem__(self, key: KT) -> None:
        _key = key.encode()
        with self.env.begin(write=True) as txn:
            success = txn.delete(_key)
        if not success:
            raise KeyError(key)

    def __iter__(self) -> Generator[KT]:
        with self.env.begin() as txn:
            for k, _ in txn.cursor():
                yield self._key_type.decode(k)

    def __len__(self) -> int:
        return self.env.stat()["entries"]

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, self._key_type):
            return False
        with self.env.begin() as txn:
            return txn.get(key.encode()) is not None


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
                self.workspace_directory / "resolved_hash_cache.mdb"
            ),
            result_hash_cache=LMDBMapping[ResolvedTaskHash, ResultHash](
                self.workspace_directory / "result_hash_cache.mdb"
            ),
            result_cache=DiskResultCacheMapping(
                result_cache_directory=self.workspace_directory / "result_cache"
            ),
            log_store={},
        )

    def get_work_dir(self, task: Task) -> Path:
        d = self.workspace_directory / "task" / str(task._resolved_hash(workspace=self))
        d.mkdir(exist_ok=True)
        return d
