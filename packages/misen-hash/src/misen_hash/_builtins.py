import datetime
import decimal
import enum
import sys
import uuid
from typing import Any

__all__ = ["is_builtin_primitive", "is_dataclass", "dataclass_dictview"]

_builtin_primitive_types = {
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
    enum.IntEnum,
    enum.Flag,
    enum.IntFlag,
}

if sys.version_info >= (3, 11):
    _builtin_primitive_types.add(enum.StrEnum)


def is_builtin_primitive(obj_type: type[Any]) -> bool:
    return obj_type in _builtin_primitive_types


def is_dataclass(obj_type: type[Any]) -> bool:
    import dataclasses

    return dataclasses.is_dataclass(obj_type)


def dataclass_dictview(dataclass_obj) -> dict[str, Any]:
    from dataclasses import fields

    return {f.name: getattr(dataclass_obj, f.name) for f in fields(dataclass_obj)}
