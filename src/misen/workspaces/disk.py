from __future__ import annotations

import shutil
import tempfile
from collections.abc import Generator, Iterator, MutableMapping
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Literal, TextIO, TypeVar

import lmdb

from misen.utils.hashes import Hash, ResolvedTaskHash, ResultHash, TaskHash
from misen.utils.locks import LockLike, NFSLock
from misen.workspace import Workspace

if TYPE_CHECKING:
    from misen.task import Task

KT = TypeVar("KT", bound=Hash)
VT = TypeVar("VT", bound=Hash)


class LMDBMapping(MutableMapping[KT, VT], Generic[KT, VT]):
    _key_type: type[KT]
    _value_type: type[VT]
    lock: NFSLock
    env: lmdb.Environment  # ty:ignore[possibly-missing-attribute]

    __slots__ = ("_key_type", "_value_type", "env", "lock")

    def __class_getitem__(cls, item: tuple[type[KT], type[VT]]):
        key_t, val_t = item
        return type(
            f"{cls.__name__}[{key_t.__name__},{val_t.__name__}]",
            (cls,),
            {"_key_type": key_t, "_value_type": val_t, "__module__": cls.__module__},
        )

    def __init__(self, database_path: Path) -> None:
        if not hasattr(self, "_key_type") or not hasattr(self, "_value_type"):
            msg = "Construct as LMDBMapping[KeyType, ValueType](...)"
            raise TypeError(msg)

        self.lock = NFSLock(database_path.with_suffix(".lock"), lifetime=10)

        self.env = lmdb.Environment(  # ty:ignore[possibly-missing-attribute]
            str(database_path),
            subdir=False,
            lock=False,
            map_size=2**28,  # 256 MiB
            max_dbs=1,
        )

    def __len__(self) -> int:
        return self.env.stat()["entries"]

    def __iter__(self) -> Generator[KT]:
        with self.env.begin() as txn:
            for k, _ in txn.cursor():
                yield self._key_type.decode(k)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, self._key_type):
            return False
        with self.env.begin() as txn:
            return txn.get(key.encode()) is not None

    def __getitem__(self, key: KT) -> VT:
        with self.env.begin() as txn:
            v: bytes | None = txn.get(key.encode(), default=None)
            if v is None:
                raise KeyError(key)
            return self._value_type.decode(v)

    def __setitem__(self, key: KT, value: VT) -> None:
        _key, _value = key.encode(), value.encode()
        with self.lock.context(blocking=True):
            with self.env.begin(write=True) as txn:
                txn.put(_key, _value)

    def __delitem__(self, key: KT) -> None:
        _key = key.encode()
        with self.lock.context(blocking=True):
            with self.env.begin(write=True) as txn:
                success = txn.delete(_key)
        if not success:
            raise KeyError(key)

    def clear(self) -> None:
        with self.lock.context(blocking=True):
            with self.env.begin(write=True) as txn:
                for _k, _ in txn.cursor():
                    txn.delete(_k)


class DiskResultStore(MutableMapping[ResultHash, Path]):
    __slots__ = ("directory",)

    directory: Path

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _result_dir_path(self, key: ResultHash) -> Path:
        return self.directory / key.hex()[:2] / key.hex()

    def __contains__(self, key: object) -> bool:
        return isinstance(key, ResultHash) and self._result_dir_path(key).exists()

    def __getitem__(self, key: ResultHash) -> Path:
        result_dir_path = self._result_dir_path(key)
        if not result_dir_path.exists():
            raise KeyError(key)
        return result_dir_path

    def __setitem__(self, key: ResultHash, value: Path) -> None:
        result_dir_path = self._result_dir_path(key)
        if not result_dir_path.exists():
            shutil.move(value, result_dir_path)

    def __delitem__(self, key: ResultHash) -> None:
        result_dir_path = self._result_dir_path(key)
        if not result_dir_path.exists():
            raise KeyError(key)
        # atomic deletion
        trash_dir = tempfile.mkdtemp(dir=result_dir_path.parent, prefix=f"{result_dir_path.name}.", suffix=".trash")
        shutil.move(result_dir_path, trash_dir)
        shutil.rmtree(trash_dir)

    def __iter__(self) -> Iterator[ResultHash]:
        for p in self.directory.glob(("[0-9a-f]" * 2) + "/" + ("[0-9a-f]" * 16)):
            if (stem := p.stem)[:2] == p.parent.name:
                yield ResultHash(stem, base=16)

    def __len__(self) -> int:
        return sum(1 for _ in self)


# TODO: get clock time from NFS server


class DiskWorkspace(Workspace):
    directory: str = ".misen"

    def __post_init__(self) -> None:
        directory = Path(self.directory)
        directory.mkdir(exist_ok=True)
        self.get_temp_dir().mkdir(parents=True, exist_ok=True)
        (directory / "work").mkdir(parents=True, exist_ok=True)
        (directory / "logs").mkdir(parents=True, exist_ok=True)
        (self.get_temp_dir() / "task_locks").mkdir(parents=True, exist_ok=True)
        (self.get_temp_dir() / "result_locks").mkdir(parents=True, exist_ok=True)

        super().__post_init__(
            resolved_hash_cache=LMDBMapping[TaskHash, ResolvedTaskHash](directory / "resolved_hash_cache.mdb"),
            result_hash_cache=LMDBMapping[ResolvedTaskHash, ResultHash](directory / "result_hash_cache.mdb"),
            result_store=DiskResultStore(directory / "results"),
        )

    def lock(self, namespace: Literal["task", "result"], key: str) -> LockLike:
        return NFSLock(
            lockfile=(self.get_temp_dir() / f"{namespace}_locks" / f"{key}.lock"),
            lifetime=30,
            refresh_interval=20,
        )

    def get_temp_dir(self) -> Path:
        return Path(self.directory) / "tmp"

    def get_work_dir(self, task: Task) -> Path:
        key_hex = task.resolved_hash(workspace=self).hex()
        d = Path(self.directory) / "work" / key_hex[:2] / f"{key_hex}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_log_dir(self, task: Task) -> Path:
        key_hex = task.resolved_hash(workspace=self).hex()
        d = Path(self.directory) / "logs" / key_hex[:2] / f"{key_hex}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def open_log(self, task: Task, mode: Literal["a", "r"], timestamp: int | Literal["latest"] = "latest") -> TextIO:
        path = self.get_log_dir(task) / f"{timestamp}.log"
        return path.open(mode, buffering=1)
