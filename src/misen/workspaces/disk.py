from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Generator, Generic, MutableMapping, TypeVar

import lmdb
from flufl.lock._lockfile import Lock

from ..utils.hashes import Hash, ResolvedTaskHash, ResultHash, TaskHash
from ..workspace import Workspace, WorkspaceParameters

if TYPE_CHECKING:
    from ..task import Task

KT = TypeVar("KT", bound=Hash)
VT = TypeVar("VT", bound=Hash)


# TODO: determine that systems have synchronized clocks
# TODO: should locks be refreshing?


class LMDBMapping(Generic[KT, VT], MutableMapping[KT, VT]):
    _key_type: type[KT]
    _value_type: type[VT]

    def __class_getitem__(cls, item: tuple[type[KT], type[VT]]):
        key_t, val_t = item
        return type(
            f"{cls.__name__}[{key_t.__name__},{val_t.__name__}]",
            (cls,),
            {"_key_type": key_t, "_value_type": val_t, "__module__": cls.__module__},
        )

    def __init__(self, database_path: Path, lifetime_s: int = 30, timeout_s: int | None = None):
        if not hasattr(self, "_key_type") or not hasattr(self, "_value_type"):
            raise TypeError("Construct as LMDBMapping[KeyType, ValueType](...)")

        self._lock = Lock(
            lockfile=str(database_path.with_suffix(".lock")),
            lifetime=lifetime_s,
            default_timeout=timeout_s,
        )

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
            v: bytes | None = txn.get(key.encode(), default=None)  # type: ignore
            if v is None:
                raise KeyError(key)
            return self._value_type.decode(v)

    def __setitem__(self, key: KT, value: VT) -> None:
        _key, _value = key.encode(), value.encode()
        with self._lock:
            with self.env.begin(write=True) as txn:
                txn.put(_key, _value)

    def __delitem__(self, key: KT) -> None:
        _key = key.encode()
        with self._lock:
            with self.env.begin(write=True) as txn:
                success = txn.delete(_key)
        if not success:
            raise KeyError(key)

    def clear(self) -> None:
        with self._lock:
            with self.env.begin(write=True) as txn:
                for _k, _ in txn.cursor():
                    txn.delete(_k)


def _atomic_write(filepath: Path, data: bytes) -> None:
    tmp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb", dir=filepath.parent, prefix=f"{filepath.name}.", suffix=".tmp", delete=False
        ) as f:
            tmp_path = Path(f.name)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, filepath)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


class DiskResultCacheMapping(MutableMapping[ResultHash, bytes]):
    def __init__(self, directory: Path):
        self.directory = directory

    def _result_filepath(self, key: ResultHash) -> Path:
        key_hex = f"{key:016x}"  # zero-padded 16 hex chars (supports uint64)
        return self.directory / key_hex[:2] / f"{key_hex}.dill"

    @contextmanager
    def _acquire_lock(self, key: ResultHash):
        _result_filepath = self._result_filepath(key)
        _result_filepath.parent.mkdir(parents=True, exist_ok=True)
        with Lock(
            lockfile=str(_result_filepath.with_suffix(".lock")),
            lifetime=90,
            default_timeout=None,
        ):
            yield

    def __contains__(self, key: object) -> bool:
        return isinstance(key, ResultHash) and self._result_filepath(key).exists()

    def __getitem__(self, key: ResultHash) -> bytes:
        try:
            with open(self._result_filepath(key), "rb") as f:
                return f.read()
        except FileNotFoundError:
            raise KeyError(key)

    def __setitem__(self, key: ResultHash, value: bytes):
        with self._acquire_lock(key):
            _filepath = self._result_filepath(key)
            if not _filepath.exists():
                _atomic_write(_filepath, value)

    def __delitem__(self, key: ResultHash) -> None:
        with self._acquire_lock(key):
            try:
                self._result_filepath(key).unlink()
            except FileNotFoundError:
                raise KeyError(key)

    def __iter__(self):
        for p in self.directory.glob(("[0-9a-f]" * 2) + "/" + ("[0-9a-f]" * 16) + ".dill"):
            if (stem := p.stem)[:2] == p.parent.name:
                yield ResultHash(stem, base=16)

    def __len__(self) -> int:
        return sum(1 for _ in self)


class DiskWorkspace(Workspace):
    def __init__(self, directory: Path = Path(".misen")):
        self.directory = directory

        self.directory.mkdir(exist_ok=True)
        (self.directory / "result_cache").mkdir(exist_ok=True)

        super().__init__(
            resolved_hash_cache=LMDBMapping[TaskHash, ResolvedTaskHash](self.directory / "resolved_hash_cache.mdb"),
            result_hash_cache=LMDBMapping[ResolvedTaskHash, ResultHash](self.directory / "result_hash_cache.mdb"),
            result_cache=DiskResultCacheMapping(self.directory / "result_cache"),
            log_store={},
        )

    def to_params(self) -> WorkspaceParameters:
        return WorkspaceParameters(DiskWorkspace, directory=self.directory)

    def get_work_dir(self, task: Task) -> Path:
        key_hex = f"{task._resolved_hash(workspace=self):016x}"  # zero-padded 16 hex chars
        d = self.directory / "work" / key_hex[:2] / f"{key_hex}"
        d.mkdir(exist_ok=True)
        return d
