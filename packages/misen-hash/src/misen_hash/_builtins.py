import dataclasses
import datetime
import decimal
import enum
import uuid
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

__all__ = [
    "dataclass_dictview",
    "hash_collection_items",
    "is_builtin_collection",
    "is_builtin_primitive",
    "builtin_primitive_value",
    "is_dataclass",
]


def is_builtin_primitive(obj: Any) -> bool:
    return isinstance(
        obj,
        (
            type(None),
            bool,
            int,
            float,
            str,
            bytes,
            bytearray,
            datetime.datetime,
            datetime.date,
            datetime.time,
            datetime.timedelta,
            uuid.UUID,
            decimal.Decimal,
            enum.Enum,
        ),
    )


def builtin_primitive_value(obj: Any) -> Any:
    if isinstance(obj, type(None)):
        return obj

    if isinstance(obj, enum.Enum):
        return obj.value

    # ensure int check is after {enum.Enum, bool}
    for _type in (bool, int, float, str, bytearray, bytes):
        if isinstance(obj, _type):
            return _type(obj)

    if isinstance(obj, datetime.datetime):
        return datetime.datetime(
            obj.year,
            obj.month,
            obj.day,
            obj.hour,
            obj.minute,
            obj.second,
            obj.microsecond,
            tzinfo=obj.tzinfo,
            fold=getattr(obj, "fold", 0),
        )

    if isinstance(obj, datetime.date):
        return datetime.date(obj.year, obj.month, obj.day)

    if isinstance(obj, datetime.time):
        return datetime.time(
            obj.hour,
            obj.minute,
            obj.second,
            obj.microsecond,
            tzinfo=obj.tzinfo,
            fold=getattr(obj, "fold", 0),
        )

    if isinstance(obj, datetime.timedelta):
        return datetime.timedelta(days=obj.days, seconds=obj.seconds, microseconds=obj.microseconds)

    if isinstance(obj, uuid.UUID):
        return uuid.UUID(bytes=obj.bytes)

    if isinstance(obj, decimal.Decimal):
        return decimal.Decimal(str(obj))

    raise TypeError(f"Unsupported type: {type(obj)!r}")


def is_builtin_collection(obj: Any) -> bool:
    return isinstance(obj, (list, tuple, set, frozenset, dict))


def hash_collection_items(
    obj: list | tuple | set | frozenset | dict, hash_fn: Callable[[Any], int]
) -> list[int] | set[int] | dict[str, int]:
    if isinstance(obj, (list, tuple)):
        return [hash_fn(o) for o in obj]
    if isinstance(obj, (set, frozenset)):
        return {hash_fn(o) for o in obj}
    if isinstance(obj, OrderedDict):
        return [hash_fn(i) for k, v in obj.items() for i in (k, v)]
    if isinstance(obj, dict):
        return {str(hash_fn(k)): hash_fn(v) for k, v in obj.items()}

    return obj


def is_dataclass(obj_type: type[Any]) -> bool:
    return dataclasses.is_dataclass(obj_type)


def dataclass_dictview(dataclass_obj) -> dict[str, Any]:
    return {f.name: getattr(dataclass_obj, f.name) for f in dataclasses.fields(dataclass_obj)}
