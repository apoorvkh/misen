from __future__ import annotations

import dataclasses
import datetime
import functools
import sys
import uuid
from operator import itemgetter
from unittest.mock import patch

import ormsgpack
from xxhash import xxh3_64_hexdigest

# TODO: numpy.ndarray
# TODO: torch.tensor


def object_identifier(obj):
    cls = type(obj)
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
    obj_type = object_identifier(obj)

    ##

    obj_data = None

    match obj:
        case (
            None
            | bytes()
            | str()
            | int()
            | float()
            | bool()
            | uuid.UUID()
            | datetime.date()
            | datetime.time()
            | datetime.datetime()
        ):
            obj_data = obj
        case bytearray():
            obj_data = bytes(obj)
        case complex():
            obj_data = (obj.real, obj.imag)
        case range() | tuple() | list():
            obj_data = [det_hash(o, seed=seed) for o in obj]
        case set() | frozenset():
            obj_data = sorted([det_hash(o, seed=seed) for o in obj])
        case dict():
            obj_data = [(det_hash(k, seed=seed), det_hash(v, seed=seed)) for k, v in obj.items()]
            obj_data = sorted(obj_data, key=itemgetter(0))

    if obj is not None and obj_data is None:
        if hash == det_hash and vars(type(obj)).get("__hash__") is not None:
            try:
                obj_data = obj.__hash__()
            except RecursionError:
                raise RecursionError(f"Error when hashing {obj_type}: circular reference likely exists")
        else:
            if dataclasses.is_dataclass(obj):
                raise TypeError(
                    f"Set `frozen=True` or `unsafe_hash=True` in @dataclass decorator to make {obj_type} hashable"
                )

            raise TypeError(f"Unhashable type: {type(obj)}")

    ##

    serialized_data = ormsgpack.packb((obj_type, obj_data))
    return xxh3_64_hexdigest(serialized_data, seed=seed)


class deterministic_hashing:
    def __init__(self, seed: int = 0):
        self.seed = seed

    def __enter__(self):
        self.patch_hash = patch("builtins.hash", functools.partial(det_hash, seed=self.seed))
        self.patch_hash.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.patch_hash.__exit__(exc_type, exc_val, exc_tb)
