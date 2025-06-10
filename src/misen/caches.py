from __future__ import annotations

from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any, Callable, Iterator, cast

import dill

from .task import Task

if TYPE_CHECKING:
    from .workspace import Workspace


class ObjectHash(int):
    pass


class ResolvedHash(int):
    pass


class ResultHash(int):
    pass


class SerializedResult:
    def __init__(self, deserializer: Callable[[bytes], Any], data: bytes):
        self.deserializer = deserializer
        self.data = data

    def value(self) -> Any:
        return self.deserializer(self.data)


class AbstractResolvedHashCache(MutableMapping[Task, ResolvedHash]):
    mapping: MutableMapping[ObjectHash, ResolvedHash]
    workspace: Workspace

    def __getitem__(self, key: Task) -> ResolvedHash:
        return self.mapping[cast("ObjectHash", key.__hash__())]

    def __setitem__(self, key: Task, value: ResolvedHash) -> None:
        self.mapping[cast("ObjectHash", key.__hash__())] = value

    def __delitem__(self, key: Task) -> None:
        del self.mapping[cast("ObjectHash", key.__hash__())]

    def __len__(self) -> int:
        return len(self.mapping)

    def __iter__(self) -> Iterator[Task]:
        raise NotImplementedError


class AbstractResultHashCache(MutableMapping[Task, ResultHash]):
    mapping: MutableMapping[ResolvedHash, ResultHash]
    workspace: Workspace

    def __getitem__(self, key: Task) -> ResultHash:
        return self.mapping[key.__resolved_hash__(workspace=self.workspace)]

    def __setitem__(self, key: Task, value: ResultHash) -> None:
        self.mapping[key.__resolved_hash__(workspace=self.workspace)] = value

    def __delitem__(self, key: Task) -> None:
        del self.mapping[key.__resolved_hash__(workspace=self.workspace)]

    def __len__(self) -> int:
        return len(self.mapping)

    def __iter__(self) -> Iterator[Task]:
        raise NotImplementedError


class AbstractResultCache(MutableMapping[Task, SerializedResult]):
    mapping: MutableMapping[ResultHash, bytes]
    workspace: Workspace

    def __getitem__(self, key: Task) -> SerializedResult:
        return dill.loads(self.mapping[key.__result_hash__(workspace=self.workspace)])

    def __setitem__(self, key: Task, value: SerializedResult) -> None:
        self.mapping[key.__result_hash__(workspace=self.workspace)] = dill.dumps(value)

    def __delitem__(self, key: Task) -> None:
        del self.mapping[key.__result_hash__(workspace=self.workspace)]

    def __len__(self) -> int:
        return len(self.mapping)

    def __iter__(self) -> Iterator[Task]:
        raise NotImplementedError
