import sys
import types
import typing as t
import unittest.mock
from dataclasses import is_dataclass
from datetime import date, datetime, time
from enum import Enum
from functools import partial
from operator import itemgetter
from uuid import UUID

import ormsgpack
from xxhash import xxh3_64_hexdigest

# TODO: what about python functions with built-in __hash__, like np.csingle?
# TODO: check if numpy is consistent
# TODO: torch.tensor
# TODO: pydantic

# TODO: patch arbitrary types
# TODO: allow tagging classes with UUID


def class_identifier(cls_or_obj: type | t.Any):
    cls = cls_or_obj if isinstance(cls_or_obj, type) else type(cls_or_obj)
    class_name = cls.__qualname__
    module_name = cls.__module__
    if module_name == "__main__":
        module = sys.modules["__main__"]
        if hasattr(module, "__file__") and module.__file__ is not None:
            module_name = module.__file__.split("/")[-1].split(".")[0]
        else:
            return class_name
    return f"{module_name}.{cls.__qualname__}"


def is_bigint(i: int):
    return (i > 0 and i.bit_length() > 64) or (i < 0 and i.bit_length() > 63)


class bigint:
    pass


try:
    import numpy as np

    SERIALIZABLE_NUMPY_TYPES = (
        np.float64,
        np.float32,
        np.float16,
        np.int64,
        np.int32,
        np.int16,
        np.int8,
        np.uint64,
        np.uint32,
        np.uint16,
        np.uint8,
        np.uintp,
        np.intp,
        np.datetime64,
        np.bool_,
    )

    def is_np_array(o) -> bool:
        return isinstance(o, np.ndarray) and o.dtype.type in SERIALIZABLE_NUMPY_TYPES

    def is_np_type(o) -> bool:
        return type(o) in SERIALIZABLE_NUMPY_TYPES

except ImportError:

    def is_np_array(o) -> bool:
        return False

    def is_np_type(o) -> bool:
        return False


def det_hash(obj, seed: int = 0) -> str:
    obj_type = class_identifier(obj)
    obj_data = None
    ormsgpack_options = 0

    match obj:
        case None | bytes() | str() | bool() | UUID() | date() | time() | datetime():
            obj_data = obj
        case int() if is_bigint(obj):
            obj_type = class_identifier(bigint)
            obj_data = str(obj)
        case int() | float() if not is_np_type(obj):
            obj_data = obj
        case bytearray():
            obj_data = bytes(obj)
        case complex():
            obj_data = (obj.real, obj.imag)
        case Enum():
            obj_data = (det_hash(obj.name, seed=seed), det_hash(obj.value, seed=seed))
        case range() | tuple() | list():
            obj_data = [det_hash(o, seed=seed) for o in obj]
        case set() | frozenset():
            obj_data = sorted([det_hash(o, seed=seed) for o in obj])
        case dict() | types.MappingProxyType():
            obj_data = [(det_hash(k, seed=seed), det_hash(v, seed=seed)) for k, v in obj.items()]
            obj_data = sorted(obj_data, key=itemgetter(0))
        case _ if is_np_type(obj):
            obj_data = obj
            ormsgpack_options |= ormsgpack.OPT_SERIALIZE_NUMPY
        case _ if is_np_array(obj) and obj.data.c_contiguous:
            obj_type = (class_identifier(obj), class_identifier(obj.dtype.type))
            obj_data = obj
            ormsgpack_options |= ormsgpack.OPT_SERIALIZE_NUMPY
        case _ if is_np_array(obj):
            obj_data = obj.tolist()
        case _ if deterministic_hashing.is_enabled() and vars(type(obj)).get("__hash__"):
            try:
                obj_data = obj.__hash__()  # pyright: ignore [reportOptionalCall]
            except RecursionError as e:
                raise RecursionError(
                    f"Circular reference likely exists in __hash__ function of {obj_type} class"
                ) from e
        case _ if is_dataclass(obj):
            raise TypeError(
                f"Set `frozen=True` or `unsafe_hash=True` in @dataclass decorator to make {obj_type} hashable"
            )
        case _:
            raise TypeError(f"Unhashable type: {obj_type}")

    serialized_data = ormsgpack.packb((obj_type, obj_data), option=ormsgpack_options)
    return xxh3_64_hexdigest(serialized_data, seed=seed)


class deterministic_hashing:
    def __init__(self, seed: int = 0):
        self.patch_hash = None
        self.seed = seed

    @staticmethod
    def is_enabled() -> bool:
        return isinstance(hash, partial) and hash.func == det_hash  # pyright: ignore [reportFunctionMemberAccess]

    def __enter__(self):
        if not self.is_enabled():
            self.patch_hash = unittest.mock.patch(
                "builtins.hash", partial(det_hash, seed=self.seed)
            )
            self.patch_hash.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.patch_hash is not None:
            self.patch_hash.__exit__(exc_type, exc_val, exc_tb)
