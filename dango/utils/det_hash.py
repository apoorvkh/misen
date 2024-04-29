from __future__ import annotations
from unittest.mock import patch
import datetime
import ormsgpack
import sys
from xxhash import xxh3_64_hexdigest
import uuid


def class_identifier(obj):
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


def det_hash(obj, obj_name: str | None = None) -> str:
    obj_type = class_identifier(obj)

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
            obj_data = [det_hash(o) for o in obj]
        case set() | frozenset():
            obj_data = sorted([det_hash(o) for o in obj])
        case dict():
            hashed_keys = {k: det_hash(k) for k in obj.keys()}
            sorted_keys = sorted(obj.keys(), key=hashed_keys.__getitem__)
            obj_data = [det_hash(obj[k], obj_name=hashed_keys[k]) for k in sorted_keys]
        case _:
            raise TypeError

    # dataclasses.dataclass
    # numpy.ndarray

    serialized_data = ormsgpack.packb((obj_name, obj_type, obj_data))
    return xxh3_64_hexdigest(serialized_data, seed=0)


class deterministic_hashing:
    def __enter__(self):
        self.patch_hash = patch("builtins.hash", det_hash)
        self.patch_hash.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.patch_hash.__exit__(exc_type, exc_val, exc_tb)
