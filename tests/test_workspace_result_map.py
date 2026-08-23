# ruff: noqa: D100, D103, EM101, S101, TRY003

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import pytest

import misen.workspace as workspace_module
from misen import Task, meta
from misen.exceptions import CacheError, LockUnavailableError, SerializationError, StorageError
from misen.utils.hashing import ResultHash
from misen.workspace import ResultMap, Workspace
from misen.workspaces.memory import InMemoryWorkspace

if TYPE_CHECKING:
    from typing import NoReturn


@meta(id="test_result_map_task", cache=True)
def _task() -> int:
    return 1


class _WorkspaceStub:
    def __init__(self, result_hash: ResultHash | BaseException) -> None:
        self.result_hash = result_hash

    def get_result_hash(self, _task: Task) -> ResultHash:
        if isinstance(self.result_hash, BaseException):
            raise self.result_hash
        return self.result_hash


class _FailingResultStore(MutableMapping[ResultHash, Path]):
    def __getitem__(self, _key: ResultHash) -> NoReturn:
        raise OSError("result store is unavailable")

    def __setitem__(self, _key: ResultHash, _value: Path) -> None:
        raise AssertionError

    def __delitem__(self, _key: ResultHash) -> NoReturn:
        raise OSError("result store is unavailable")

    def __iter__(self) -> Iterator[ResultHash]:
        return iter(())

    def __len__(self) -> int:
        return 0


def _result_map(
    workspace: _WorkspaceStub,
    store: MutableMapping[ResultHash, Path] | None = None,
) -> ResultMap:
    return ResultMap(store if store is not None else {}, cast("Workspace", workspace))


def _operate(results: ResultMap, task: Task, operation: Literal["get", "delete"]) -> object:
    if operation == "get":
        return results[task]
    del results[task]
    return None


@pytest.mark.parametrize("operation", ["get", "delete"])
def test_missing_result_pointer_is_a_mapping_miss(operation: Literal["get", "delete"]) -> None:
    task = Task(_task)
    results = _result_map(_WorkspaceStub(CacheError("not computed")))

    with pytest.raises(KeyError, match="not found in cache") as raised:
        _operate(results, task, operation)

    assert isinstance(raised.value.__cause__, CacheError)


@pytest.mark.parametrize("operation", ["get", "delete"])
def test_result_hash_backend_errors_are_not_cache_misses(operation: Literal["get", "delete"]) -> None:
    task = Task(_task)
    results = _result_map(_WorkspaceStub(OSError("hash index is unavailable")))

    with pytest.raises(StorageError, match="hash index is unavailable") as raised:
        _operate(results, task, operation)

    assert isinstance(raised.value.__cause__, OSError)


@pytest.mark.parametrize("operation", ["get", "delete"])
def test_result_store_errors_are_not_cache_misses(operation: Literal["get", "delete"]) -> None:
    task = Task(_task)
    results = _result_map(_WorkspaceStub(ResultHash(1)), _FailingResultStore())

    with pytest.raises(StorageError, match="result store is unavailable") as raised:
        _operate(results, task, operation)

    assert isinstance(raised.value.__cause__, OSError)


def test_deserializer_key_error_is_not_reported_as_a_cache_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = Task(_task)
    result_hash = ResultHash(1)
    results = _result_map(_WorkspaceStub(result_hash), {result_hash: tmp_path})

    def corrupt_load(*_args: object, **_kwargs: object) -> None:
        raise KeyError("missing payload leaf")

    monkeypatch.setattr(workspace_module.serde, "load", corrupt_load)

    with pytest.raises(SerializationError, match="missing payload leaf") as raised:
        results[task]

    assert isinstance(raised.value.__cause__, KeyError)


def test_result_payload_is_not_published_after_losing_its_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = InMemoryWorkspace(directory=str(tmp_path / "workspace"))
    result_hash = ResultHash(7)
    real_lock = workspace.lock

    class _LostLock:
        def context(self, *, blocking: bool = True, timeout: int | None = None) -> object:
            del blocking, timeout
            return nullcontext(self)

        def is_locked(self) -> bool:
            return False

    def lock(namespace: str, key: str) -> object:
        return _LostLock() if namespace == "result" else real_lock(namespace=namespace, key=key)  # type: ignore[arg-type]

    monkeypatch.setattr(workspace, "lock", lock)

    with pytest.raises(LockUnavailableError, match="Lost the result lock"):
        workspace.results.store(Task(_task), 1, result_hash)

    assert result_hash not in workspace.results.result_store
