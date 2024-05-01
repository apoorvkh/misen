import sys
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

# option=ormsgpack.OPT_SERIALIZE_NUMPY
# TODO: numpy.ndarray
# The array must be a contiguous C array (C_CONTIGUOUS) and one of the supported datatypes
# else obj.tolist() can be specified
# TODO: numpy.float64, numpy.float32, numpy.float16, numpy.int64, numpy.int32, numpy.int16, numpy.int8, numpy.uint64, numpy.uint32, numpy.uint16, numpy.uint8, numpy.uintp, numpy.intp, numpy.datetime64, and numpy.bool

# TODO: torch.tensor
# TODO: pydantic

# TODO: patch arbitrary types
# TODO: allow tagging classes with UUID


class bigint:
    pass


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


def det_hash(obj, seed: int = 0) -> str:
    obj_type = class_identifier(obj)

    # for serializing larger than 64-bit integers
    if isinstance(obj, int) and (
        (obj > 0 and obj.bit_length() > 64) or (obj < 0 and obj.bit_length() > 63)
    ):
        obj_type = class_identifier(bigint)
        obj = str(obj)

    ##

    obj_data = None
    native_type = True

    match obj:
        case (
            None
            | bytes()
            | str()
            | int()
            | float()
            | bool()
            | UUID()
            | date()
            | time()
            | datetime()
        ):
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
        case dict():
            obj_data = [(det_hash(k, seed=seed), det_hash(v, seed=seed)) for k, v in obj.items()]
            obj_data = sorted(obj_data, key=itemgetter(0))
        case _:
            native_type = False

    if not native_type:
        if deterministic_hashing.is_enabled() and vars(type(obj)).get("__hash__") is not None:
            try:
                obj_data = obj.__hash__()  # pyright: ignore [reportOptionalCall]
            except RecursionError as e:
                raise RecursionError(
                    f"Circular reference likely exists in __hash__ function of {obj_type} class"
                ) from e
        elif is_dataclass(obj):
            raise TypeError(
                f"Set `frozen=True` or `unsafe_hash=True` in @dataclass decorator to make {obj_type} hashable"
            )
        else:
            raise TypeError(f"Unhashable type: {obj_type}")

    ##

    serialized_data = ormsgpack.packb((obj_type, obj_data))
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
