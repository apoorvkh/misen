import io
from pickle import UnpicklingError
from typing import Any

import dill
import msgspec.msgpack
from xxhash import xxh3_64, xxh3_64_intdigest

from ._attrs import attrs_dictview, is_attrs_class
from ._builtins import (
    builtin_primitive_value,
    dataclass_dictview,
    hash_collection_items,
    is_builtin_collection,
    is_builtin_primitive,
    is_dataclass,
)
from ._msgspec import is_msgspec_struct, msgspec_struct_dictview
from ._torch import is_torch_object, normalize_torch_object

# TODO: numpy, pandas, polars, arrow, PyTorch datasets, HF datasets


def msgpack_hash(obj: Any) -> int:
    return xxh3_64_intdigest(msgspec.msgpack.encode(obj, order="sorted"), seed=0)


class _HashWriter(io.RawIOBase):
    """File-like sink that updates an xxhash object with whatever is written."""

    def __init__(self) -> None:
        self._h = xxh3_64(seed=0)

    def writable(self) -> bool:
        return True

    def write(self, b: bytes | bytearray | memoryview) -> int:
        mv = memoryview(b)
        self._h.update(mv)
        return mv.nbytes

    def intdigest(self) -> int:
        return self._h.intdigest()


def hash1(obj: Any) -> int:
    if is_builtin_primitive(obj):
        return msgpack_hash(builtin_primitive_value(obj))
    if is_builtin_collection(obj):
        return msgpack_hash(hash_collection_items(obj, hash_fn=hash2))
    elif is_dataclass(obj):
        return hash1(dataclass_dictview(obj))
    elif is_msgspec_struct(obj):
        return hash1(msgspec_struct_dictview(obj))
    elif is_attrs_class(obj):
        return hash1(attrs_dictview(obj))
    elif is_torch_object(obj):
        return hash1(normalize_torch_object(obj))
    elif hasattr(obj, "__getstate__"):
        return hash1(obj.__getstate__())

    try:
        hash_sink = _HashWriter()
        dill.dump(obj, hash_sink, protocol=5)
        return hash_sink.intdigest()
        # print logger warning to flag dill
    except UnpicklingError as e:
        raise ValueError("Unsupported type for hashing") from e


def hash2(obj: Any) -> int:
    return msgpack_hash((type(obj).__qualname__, hash1(obj)))


canonical_hash = hash2

__all__ = ["hash2", "canonical_hash"]
