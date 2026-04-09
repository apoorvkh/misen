"""Canonical-hash handlers for Python standard library value types."""

import array
import dataclasses
import datetime
import decimal
import enum
import fractions
import ipaddress
import math
import pathlib
import re
import types
import uuid
import zoneinfo
from collections import ChainMap, Counter, OrderedDict, UserDict, UserList, UserString, defaultdict, deque
from collections.abc import Callable, Iterable
from typing import Any

from misen.utils.hashing.handler_base import (
    CollectionHandler,
    Handler,
    HandlerTypeList,
    HandlerTypeRegistry,
    PrimitiveHandler,
    hash_values,
    qualified_type_name,
)

__all__ = ["stdlib_handlers", "stdlib_handlers_by_type"]

_DICT_KEYS_TYPE = type({}.keys())
_DICT_VALUES_TYPE = type({}.values())
_DICT_ITEMS_TYPE = type({}.items())


def _normalized_float(value: float) -> float | str:
    if math.isfinite(value):
        return value
    return str(value)


def _digest_mapping_items(items: Iterable[tuple[Any, Any]], element_hash: Callable[[Any], int] | None) -> int:
    if element_hash is None:
        msg = "Mapping handlers require element_hash."
        raise ValueError(msg)

    return hash_values(
        {
            hash_values(
                (
                    element_hash(key),
                    element_hash(value),
                )
            )
            for key, value in items
        }
    )


class NoneHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return obj is None

    @staticmethod
    def digest(obj: Any) -> int:  # noqa: ARG004
        return hash_values(None)


class EnumHandler(Handler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, enum.Enum)

    @staticmethod
    def digest(obj: Any, element_hash: Callable[[Any], int] | None) -> int:
        if element_hash is None:
            msg = "EnumHandler requires element_hash."
            raise ValueError(msg)
        return element_hash(obj.value)


class BoolHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, bool)

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_values(bool(obj))


class IntHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, int) and not isinstance(obj, (bool, enum.Enum))

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_values(int(obj))


class FloatHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, float)

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_values(_normalized_float(float(obj)))


class ComplexHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, complex)

    @staticmethod
    def digest(obj: Any) -> int:
        value = complex(obj)
        return hash_values(
            (
                _normalized_float(float(value.real)),
                _normalized_float(float(value.imag)),
            )
        )


class StrHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, str)

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_values(str(obj))


class BytearrayHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, bytearray)

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_values(bytes(obj))


class BytesHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, bytes)

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_values(bytes(obj))


class MemoryviewHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, memoryview)

    @staticmethod
    def digest(obj: Any) -> int:
        view = memoryview(obj)
        return hash_values(
            (
                view.format,
                view.ndim,
                view.shape,
                view.strides,
                view.readonly,
                view.tobytes(),
            )
        )


class DatetimeHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, datetime.datetime)

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_values(
            (
                obj.year,
                obj.month,
                obj.day,
                obj.hour,
                obj.minute,
                obj.second,
                obj.microsecond,
                str(obj.tzinfo),
                obj.fold,
            )
        )


class DateHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, datetime.date) and not isinstance(obj, datetime.datetime)

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_values((obj.year, obj.month, obj.day))


class TimeHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, datetime.time)

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_values(
            (
                obj.hour,
                obj.minute,
                obj.second,
                obj.microsecond,
                str(obj.tzinfo),
                obj.fold,
            )
        )


class TimedeltaHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, datetime.timedelta)

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_values((obj.days, obj.seconds, obj.microseconds))


class UUIDHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, uuid.UUID)

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_values(obj.bytes)


class DecimalHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, decimal.Decimal)

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_values(str(obj))


class FractionHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, fractions.Fraction)

    @staticmethod
    def digest(obj: Any) -> int:
        value = fractions.Fraction(obj)
        return hash_values((value.numerator, value.denominator))


class RangeHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, range)

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_values((obj.start, obj.stop, obj.step))


class SliceHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, slice)

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_values((obj.start, obj.stop, obj.step))


class PathHandler(PrimitiveHandler):
    @staticmethod
    def type_name(obj: Any) -> str:
        # pathlib.Path() produces PosixPath on Unix and WindowsPath on
        # Windows.  Collapse concrete paths to a single stable name so
        # hashes are cross-platform.  Pure variants (PurePosixPath,
        # PureWindowsPath) are explicitly chosen by the user and stay
        # distinct.
        if isinstance(obj, pathlib.Path):
            return "pathlib.Path"
        return qualified_type_name(type(obj))

    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, pathlib.PurePath)

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_values((obj.drive, obj.root, obj.parts))


class PatternHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, re.Pattern)

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_values((obj.pattern, obj.flags))


class ZoneInfoHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, zoneinfo.ZoneInfo)

    @staticmethod
    def digest(obj: Any) -> int:
        key = getattr(obj, "key", None)
        if key is None:
            msg = "ZoneInfo objects must expose a stable key."
            raise ValueError(msg)
        return hash_values(key)


class IPAddressHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(
            obj,
            (
                ipaddress.IPv4Address,
                ipaddress.IPv6Address,
                ipaddress.IPv4Network,
                ipaddress.IPv6Network,
                ipaddress.IPv4Interface,
                ipaddress.IPv6Interface,
            ),
        )

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_values(str(obj))


class ArrayHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, array.array)

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_values((obj.typecode, obj.tolist()))


class EllipsisHandler(PrimitiveHandler):
    """Hash the Ellipsis singleton."""

    @staticmethod
    def match(obj: Any) -> bool:
        return obj is ...

    @staticmethod
    def digest(obj: Any) -> int:  # noqa: ARG004
        return hash_values("...")


class TypeHandler(PrimitiveHandler):
    """Hash type objects (classes) by their qualified name."""

    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, type)

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_values(qualified_type_name(obj))


class SimpleNamespaceHandler(Handler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, types.SimpleNamespace)

    @staticmethod
    def digest(obj: Any, element_hash: Callable[[Any], int] | None) -> int:
        return _digest_mapping_items(vars(obj).items(), element_hash=element_hash)


class UserDictHandler(Handler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, UserDict)

    @staticmethod
    def digest(obj: Any, element_hash: Callable[[Any], int] | None) -> int:
        return _digest_mapping_items(obj.data.items(), element_hash=element_hash)


class UserListHandler(CollectionHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, UserList)

    @staticmethod
    def elements(obj: Any) -> list[Any]:
        return list(obj.data)


class UserStringHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, UserString)

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_values(str(obj))


class NamedTupleHandler(CollectionHandler):
    """Hash named tuples by field name/value pairs (like DataclassHandler)."""

    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, tuple) and hasattr(type(obj), "_fields")

    @staticmethod
    def elements(obj: Any) -> list[Any]:
        return [(f, getattr(obj, f)) for f in type(obj)._fields]


class ListHandler(CollectionHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, (list, tuple, set, frozenset))

    @staticmethod
    def elements(obj: Any) -> list[Any] | set[Any]:
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, frozenset):
            return set(obj)
        return obj


class DequeHandler(CollectionHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, deque)

    @staticmethod
    def elements(obj: Any) -> list[Any]:
        return list(obj)


class OrderedDictHandler(CollectionHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, OrderedDict)

    @staticmethod
    def elements(obj: Any) -> list[Any]:
        return list(obj.items())


class DefaultDictHandler(Handler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, defaultdict)

    @staticmethod
    def digest(obj: Any, element_hash: Callable[[Any], int] | None) -> int:
        default_factory_hash = None
        if obj.default_factory is not None:
            if element_hash is None:
                msg = "DefaultDictHandler requires element_hash."
                raise ValueError(msg)
            default_factory_hash = element_hash(obj.default_factory)
        return hash_values(
            (
                default_factory_hash,
                _digest_mapping_items(obj.items(), element_hash=element_hash),
            )
        )


class CounterHandler(Handler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, Counter)

    @staticmethod
    def digest(obj: Any, element_hash: Callable[[Any], int] | None) -> int:
        return _digest_mapping_items(obj.items(), element_hash=element_hash)


class DictHandler(Handler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, dict) and not isinstance(obj, (OrderedDict, defaultdict, Counter))

    @staticmethod
    def digest(obj: Any, element_hash: Callable[[Any], int] | None) -> int:
        return _digest_mapping_items(obj.items(), element_hash=element_hash)


class DictKeysViewHandler(CollectionHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, _DICT_KEYS_TYPE)

    @staticmethod
    def elements(obj: Any) -> set[Any]:
        return set(obj)


class DictValuesViewHandler(CollectionHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, _DICT_VALUES_TYPE)

    @staticmethod
    def elements(obj: Any) -> list[Any]:
        return list(obj)


class DictItemsViewHandler(Handler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, _DICT_ITEMS_TYPE)

    @staticmethod
    def digest(obj: Any, element_hash: Callable[[Any], int] | None) -> int:
        return _digest_mapping_items(obj, element_hash=element_hash)


class ChainMapHandler(Handler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, ChainMap)

    @staticmethod
    def digest(obj: Any, element_hash: Callable[[Any], int] | None) -> int:
        return hash_values([_digest_mapping_items(mapping.items(), element_hash=element_hash) for mapping in obj.maps])


class MappingProxyHandler(Handler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, types.MappingProxyType)

    @staticmethod
    def digest(obj: Any, element_hash: Callable[[Any], int] | None) -> int:
        return _digest_mapping_items(obj.items(), element_hash=element_hash)


class DataclassHandler(CollectionHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return dataclasses.is_dataclass(obj) and not isinstance(obj, type)

    @staticmethod
    def elements(obj: Any) -> list[Any]:
        return [(f.name, getattr(obj, f.name)) for f in dataclasses.fields(obj)]


stdlib_handlers: HandlerTypeList = [
    NoneHandler,
    EnumHandler,
    BoolHandler,
    IntHandler,
    FloatHandler,
    ComplexHandler,
    StrHandler,
    BytearrayHandler,
    BytesHandler,
    MemoryviewHandler,
    DatetimeHandler,
    DateHandler,
    TimeHandler,
    TimedeltaHandler,
    UUIDHandler,
    DecimalHandler,
    FractionHandler,
    RangeHandler,
    SliceHandler,
    PathHandler,
    PatternHandler,
    ZoneInfoHandler,
    IPAddressHandler,
    ArrayHandler,
    EllipsisHandler,
    TypeHandler,
    SimpleNamespaceHandler,
    UserDictHandler,
    UserListHandler,
    UserStringHandler,
    NamedTupleHandler,
    ListHandler,
    DequeHandler,
    OrderedDictHandler,
    DefaultDictHandler,
    CounterHandler,
    DictHandler,
    DictKeysViewHandler,
    DictValuesViewHandler,
    DictItemsViewHandler,
    ChainMapHandler,
    MappingProxyHandler,
    DataclassHandler,
]

# Exact-type and base-type fast-path map by fully-qualified type name.
_stdlib_handlers_by_type: dict[type[Any], type[Handler]] = {
    None.__class__: NoneHandler,
    enum.Enum: EnumHandler,
    enum.IntEnum: EnumHandler,
    enum.Flag: EnumHandler,
    enum.IntFlag: EnumHandler,
    bool: BoolHandler,
    int: IntHandler,
    float: FloatHandler,
    complex: ComplexHandler,
    str: StrHandler,
    bytearray: BytearrayHandler,
    bytes: BytesHandler,
    memoryview: MemoryviewHandler,
    datetime.datetime: DatetimeHandler,
    datetime.date: DateHandler,
    datetime.time: TimeHandler,
    datetime.timedelta: TimedeltaHandler,
    uuid.UUID: UUIDHandler,
    decimal.Decimal: DecimalHandler,
    fractions.Fraction: FractionHandler,
    range: RangeHandler,
    slice: SliceHandler,
    pathlib.PurePath: PathHandler,
    pathlib.PurePosixPath: PathHandler,
    pathlib.PureWindowsPath: PathHandler,
    pathlib.PosixPath: PathHandler,
    pathlib.WindowsPath: PathHandler,
    re.Pattern: PatternHandler,
    zoneinfo.ZoneInfo: ZoneInfoHandler,
    ipaddress.IPv4Address: IPAddressHandler,
    ipaddress.IPv6Address: IPAddressHandler,
    ipaddress.IPv4Network: IPAddressHandler,
    ipaddress.IPv6Network: IPAddressHandler,
    ipaddress.IPv4Interface: IPAddressHandler,
    ipaddress.IPv6Interface: IPAddressHandler,
    array.array: ArrayHandler,
    type(...).__class__: EllipsisHandler,
    type: TypeHandler,
    types.SimpleNamespace: SimpleNamespaceHandler,
    UserDict: UserDictHandler,
    UserList: UserListHandler,
    UserString: UserStringHandler,
    list: ListHandler,
    tuple: ListHandler,
    set: ListHandler,
    frozenset: ListHandler,
    deque: DequeHandler,
    OrderedDict: OrderedDictHandler,
    defaultdict: DefaultDictHandler,
    Counter: CounterHandler,
    dict: DictHandler,
    _DICT_KEYS_TYPE: DictKeysViewHandler,
    _DICT_VALUES_TYPE: DictValuesViewHandler,
    _DICT_ITEMS_TYPE: DictItemsViewHandler,
    ChainMap: ChainMapHandler,
    types.MappingProxyType: MappingProxyHandler,
}

stdlib_handlers_by_type: HandlerTypeRegistry = {
    qualified_type_name(obj_type): handler_cls for obj_type, handler_cls in _stdlib_handlers_by_type.items()
}
