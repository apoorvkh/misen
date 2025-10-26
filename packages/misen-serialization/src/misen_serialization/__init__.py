from dataclasses import is_dataclass
from typing import Any, Callable, Type

import dill
import msgspec.json
from xxhash import xxh3_64_intdigest

from ._attrs import is_attrs_class
from ._builtins import register_serializer_for_builtin_types
from ._msgspec import is_msgspec_struct
from ._torch import is_torch_object, torch_serializer


def msgspec_serializer(obj: Any) -> bytes:
    return msgspec.json.encode(obj, enc_hook=serialize, order="sorted")


def serialize(obj: Any) -> bytes:
    type_name = type(obj).__qualname__.encode("utf-8")
    serializer = _get_serializer(obj)
    payload = serializer(obj)
    return msgspec_serializer((type_name, payload))


def canonical_hash(obj: Any, seed: int = 0) -> int:
    return xxh3_64_intdigest(serialize(obj), seed=seed)


_SERIALIZERS: dict[Type, Callable[[Any], bytes]] = {}


def register_serializer(types: list[Type], fn: Callable[[Any], bytes]) -> None:
    for typ in types:
        _SERIALIZERS[typ] = fn


register_serializer_for_builtin_types()


def _get_serializer(obj: Any) -> Callable[[Any], bytes]:
    obj_type = type(obj)
    if obj_type in _SERIALIZERS:
        return _SERIALIZERS[obj_type]
    elif is_dataclass(obj) or is_msgspec_struct(obj) or is_attrs_class(obj):
        return msgspec_serializer
    elif is_torch_object(obj):
        return torch_serializer
    elif hasattr(obj, "__getstate__"):
        return lambda o: serialize(o.__getstate__())  # noqa: E731
    elif hasattr(obj, "__dict__"):
        return lambda o: serialize(o.__dict__)  # noqa: E731
    elif hasattr(obj, "__slots__"):
        return lambda o: serialize(  # noqa: E731
            {slot: getattr(o, slot) for slot in o.__slots__}
        )
    elif hasattr(obj, "__iter__"):
        return lambda o: serialize(list(o))  # noqa: E731

    # TODO: add logger.warning to flag fallback
    return dill.dumps


__all__ = ["canonical_hash", "msgspec_serializer", "serialize", "register_serializer"]
