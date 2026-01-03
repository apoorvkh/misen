from pickle import UnpicklingError
from typing import Any

import dill
import msgspec.msgpack
from xxhash import xxh3_64_intdigest

from ._attrs import attrs_dictview, is_attrs_class
from ._builtins import dataclass_dictview, is_builtin_primitive, is_dataclass
from ._msgspec import is_msgspec_struct, msgspec_struct_dictview
from ._torch import is_torch_object


def hash1(obj: Any, seed: int = 0) -> int:
    obj_type: type[Any] = type(obj)

    if is_builtin_primitive(obj_type) or obj_type in {list, tuple, set, frozenset, dict}:
        if obj_type in {list, tuple, set, frozenset}:
            obj = {hash2(o) for o in obj}
        elif obj_type is dict:
            obj = {str(hash2(k)): hash2(v) for k, v in obj.items()}
        return xxh3_64_intdigest(msgspec.msgpack.encode(obj, order="sorted"), seed=seed)
    elif is_dataclass(obj):
        return hash1(dataclass_dictview(obj))
    elif is_msgspec_struct(obj):
        return hash1(msgspec_struct_dictview(obj))
    elif is_attrs_class(obj):
        return hash1(attrs_dictview(obj))
    elif is_torch_object(obj):
        raise NotImplementedError
    elif hasattr(obj, "__getstate__"):
        return hash1(obj.__getstate__())

    try:
        return xxh3_64_intdigest(dill.dumps(obj), seed=seed)
        # print logger warning to flag dill
    except UnpicklingError as e:
        raise ValueError("Unsupported type for hashing") from e


def hash2(obj: Any, seed: int = 0) -> int:
    type_name = type(obj).__qualname__
    return xxh3_64_intdigest(
        msgspec.msgpack.encode((type_name, hash1(obj)), order="sorted"),
        seed=seed,
    )


canonical_hash = hash2

__all__ = ["hash2", "canonical_hash"]
