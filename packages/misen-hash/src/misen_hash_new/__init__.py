from abc import ABC, abstractmethod
from pickle import UnpicklingError
from typing import Any

import dill
import msgspec
from xxhash import xxh3_64_intdigest


def canonical_hash(obj: Any) -> int:
    return HashTree.digest(obj)


class HashTree:
    @staticmethod
    def digest(obj) -> int:
        obj_type = type(obj).__qualname__
        obj_hash = _hash_by_type(obj)
        return _hash_msgpack((obj_type, obj_hash))


class HashPrimitive(ABC):
    @staticmethod
    @abstractmethod
    def match(obj: Any) -> bool: ...

    @staticmethod
    @abstractmethod
    def digest(obj: Any) -> int: ...


class CollectionForHashing(ABC):
    @staticmethod
    @abstractmethod
    def match(obj: Any) -> bool: ...

    @staticmethod
    @abstractmethod
    def children(obj: Any) -> list[Any] | set[Any]: ...

    @classmethod
    def digest(cls, obj: Any) -> int:
        children = cls.children(obj)
        if isinstance(children, list):
            return _hash_msgpack([HashTree.digest(c) for c in children])
        elif isinstance(children, set):
            return _hash_msgpack({HashTree.digest(c) for c in children})
        raise ValueError(f"Unsupported collection type: {type(children)}")


def _hash_msgpack(obj: Any) -> int:
    return xxh3_64_intdigest(msgspec.msgpack.encode(obj, order="deterministic"), seed=0)


from ._builtins import DictCollection  # noqa: E402
from ._torch import TensorPrimitive  # noqa: E402


def _hash_by_type(obj: Any) -> int:
    for primitive_cls in [TensorPrimitive]:
        if primitive_cls.match(obj):
            return primitive_cls.digest(obj)

    for collection_cls in [DictCollection]:
        if collection_cls.match(obj):
            return collection_cls.digest(obj)

    try:
        return xxh3_64_intdigest(dill.dumps(obj, protocol=5))
    except UnpicklingError as e:
        raise ValueError("Unsupported type for hashing") from e
