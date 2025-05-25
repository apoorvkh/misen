import sys
from typing import Any

import dill
import msgspec
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


def det_hash(obj: Any, seed: int = 0) -> int:
    from ..task import Task

    obj_type = class_identifier(obj)

    match obj:
        case Task():
            obj_data = (obj.properties.id, {k: det_hash(v) for k, v in obj.bound_arguments.items()})
        case _:
            obj_data = obj

    serialized_data = msgspec.json.encode((obj_type, obj_data), enc_hook=dill.dumps, order="sorted")

    return xxh3_64_intdigest(serialized_data, seed=seed)
