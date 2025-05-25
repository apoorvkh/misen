import sys
from typing import Any

import dill
from xxhash import xxh3_64_intdigest


def class_identifier(cls_or_obj: type | Any) -> str:
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


def serialize(obj: Any) -> bytes:
    obj_type = class_identifier(obj)
    return dill.dumps((obj_type, obj))


def det_hash(obj: Any, seed: int = 0) -> int:
    serialized_data = serialize(obj)
    return xxh3_64_intdigest(serialized_data, seed=seed)
