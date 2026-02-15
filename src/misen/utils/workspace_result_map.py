"""Workspace result map implementation."""

from __future__ import annotations

import contextlib
import shutil
from collections.abc import Iterator, MutableMapping
from typing import TYPE_CHECKING, Any, TypeVar

from misen.task import Task

if TYPE_CHECKING:
    from pathlib import Path

    from misen.utils.hashes import ResultHash
    from misen.workspace import Workspace

R = TypeVar("R")


class ResultMap(MutableMapping[Task[Any], Any]):
    """Mapping interface for task results stored in a workspace."""

    __slots__ = ("result_store", "workspace")

    def __init__(self, result_store: MutableMapping[ResultHash, Path], workspace: Workspace) -> None:
        """Initialize the result map wrapper."""
        self.result_store = result_store
        self.workspace = workspace

    def __getitem__(self, key: Task[R], /) -> R:
        """Return the cached result for a task.

        Raises:
            KeyError: If the result is not present in the cache.
        """
        try:
            result_hash = key.result_hash(workspace=self.workspace)
            directory = self.result_store[result_hash]
        except Exception as e:
            msg = f"Result for task {key} not found in cache."
            raise KeyError(msg) from e
        return key.properties.serializer.load(directory)

    def __setitem__(self, key: Task[R], value: R, /) -> None:
        """Persist a result for the given task."""
        result_hash = key.result_hash(workspace=self.workspace)
        with self.workspace.lock(namespace="result", key=result_hash.b32()).context(blocking=True, timeout=None):
            if result_hash not in self.result_store:
                tmp_dir = self.workspace.get_temp_dir() / "results" / result_hash.b32()
                tmp_dir.mkdir(parents=True, exist_ok=True)
                key.properties.serializer.save(value, tmp_dir)
                self.result_store[result_hash] = tmp_dir
                with contextlib.suppress(FileNotFoundError):
                    shutil.rmtree(tmp_dir)

    def __delitem__(self, key: Task[R], /) -> None:
        """Remove a cached result for a task.

        Raises:
            KeyError: If the result is not present in the cache.
        """
        try:
            result_hash = key.result_hash(workspace=self.workspace)
            del self.result_store[result_hash]
        except Exception as e:
            msg = f"Result for task {key} not found in cache."
            raise KeyError(msg) from e

    def __iter__(self) -> Iterator[Task]:
        """Iterate over tasks in the mapping."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of cached results."""
        return len(self.result_store)

    def __contains__(self, key: object, /) -> bool:
        """Return True if the task has a cached result."""
        if not isinstance(key, Task):
            return False
        try:
            result_hash = key.result_hash(workspace=self.workspace)
        except RuntimeError:
            return False
        return result_hash in self.result_store
