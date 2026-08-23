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

from misen.exceptions import HashError
from misen.utils.hashing.base import (
    CollectionHandler,
    ElementHasher,
    Handler,
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


def _digest_mapping_items(items: Iterable[tuple[Any, Any]], element_hash: ElementHasher) -> int:
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


class _InstanceMatch:
    """Match instances declaratively while preserving each handler's digest."""

    _types: type[Any] | tuple[type[Any], ...]
    _excluded_types: tuple[type[Any], ...] = ()

    @classmethod
    def match(cls, obj: Any) -> bool:
        return isinstance(obj, cls._types) and not isinstance(obj, cls._excluded_types)


class _ProjectedPrimitive(_InstanceMatch, PrimitiveHandler):
    """Hash a primitive through its handler-specific canonical projection."""

    _project: Callable[[Any], Any]

    @classmethod
    def digest(cls, obj: Any, _element_hash: ElementHasher, /) -> int:
        return hash_values(cls._project(obj))


class NoneHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return obj is None

    @classmethod
    def digest(cls, _obj: Any, _element_hash: ElementHasher, /) -> int:
        return hash_values(None)


class EnumHandler(_InstanceMatch, Handler):
    _types = enum.Enum

    @classmethod
    def digest(cls, obj: Any, element_hash: ElementHasher, /) -> int:
        return element_hash(obj.value)


class BoolHandler(_ProjectedPrimitive):
    _types = bool
    _project = bool


class IntHandler(_ProjectedPrimitive):
    _types = int
    _excluded_types = (bool, enum.Enum)
    _project = int


class FloatHandler(_InstanceMatch, PrimitiveHandler):
    _types = float

    @classmethod
    def digest(cls, obj: Any, _element_hash: ElementHasher, /) -> int:
        return hash_values(_normalized_float(float(obj)))


class ComplexHandler(_InstanceMatch, PrimitiveHandler):
    _types = complex

    @classmethod
    def digest(cls, obj: Any, _element_hash: ElementHasher, /) -> int:
        value = complex(obj)
        return hash_values(
            (
                _normalized_float(float(value.real)),
                _normalized_float(float(value.imag)),
            )
        )


class StrHandler(_ProjectedPrimitive):
    _types = str
    _project = str


class BytearrayHandler(_ProjectedPrimitive):
    _types = bytearray
    _project = bytes


class BytesHandler(_ProjectedPrimitive):
    _types = bytes
    _project = bytes


class MemoryviewHandler(_InstanceMatch, PrimitiveHandler):
    _types = memoryview

    @classmethod
    def digest(cls, obj: Any, _element_hash: ElementHasher, /) -> int:
        try:
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
        except ValueError as exc:
            msg = "Cannot produce a stable hash for a released memoryview."
            raise HashError(msg) from exc


class DatetimeHandler(_InstanceMatch, PrimitiveHandler):
    _types = datetime.datetime

    @classmethod
    def digest(cls, obj: Any, _element_hash: ElementHasher, /) -> int:
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


class DateHandler(_InstanceMatch, PrimitiveHandler):
    _types = datetime.date
    _excluded_types = (datetime.datetime,)

    @classmethod
    def digest(cls, obj: Any, _element_hash: ElementHasher, /) -> int:
        return hash_values((obj.year, obj.month, obj.day))


class TimeHandler(_InstanceMatch, PrimitiveHandler):
    _types = datetime.time

    @classmethod
    def digest(cls, obj: Any, _element_hash: ElementHasher, /) -> int:
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


class TimedeltaHandler(_InstanceMatch, PrimitiveHandler):
    _types = datetime.timedelta

    @classmethod
    def digest(cls, obj: Any, _element_hash: ElementHasher, /) -> int:
        return hash_values((obj.days, obj.seconds, obj.microseconds))


class UUIDHandler(_InstanceMatch, PrimitiveHandler):
    _types = uuid.UUID

    @classmethod
    def digest(cls, obj: Any, _element_hash: ElementHasher, /) -> int:
        return hash_values(obj.bytes)


class DecimalHandler(_ProjectedPrimitive):
    _types = decimal.Decimal
    _project = str


class FractionHandler(_InstanceMatch, PrimitiveHandler):
    _types = fractions.Fraction

    @classmethod
    def digest(cls, obj: Any, _element_hash: ElementHasher, /) -> int:
        value = fractions.Fraction(obj)
        return hash_values((value.numerator, value.denominator))


class RangeHandler(_InstanceMatch, PrimitiveHandler):
    _types = range

    @classmethod
    def digest(cls, obj: Any, _element_hash: ElementHasher, /) -> int:
        return hash_values((obj.start, obj.stop, obj.step))


class SliceHandler(_InstanceMatch, PrimitiveHandler):
    _types = slice

    @classmethod
    def digest(cls, obj: Any, _element_hash: ElementHasher, /) -> int:
        return hash_values((obj.start, obj.stop, obj.step))


class PathHandler(_InstanceMatch, PrimitiveHandler):
    _types = pathlib.PurePath

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

    @classmethod
    def digest(cls, obj: Any, _element_hash: ElementHasher, /) -> int:
        return hash_values((obj.drive, obj.root, obj.parts))


class PatternHandler(_InstanceMatch, PrimitiveHandler):
    _types = re.Pattern

    @classmethod
    def digest(cls, obj: Any, _element_hash: ElementHasher, /) -> int:
        return hash_values((obj.pattern, obj.flags))


class ZoneInfoHandler(_InstanceMatch, PrimitiveHandler):
    _types = zoneinfo.ZoneInfo

    @classmethod
    def digest(cls, obj: Any, _element_hash: ElementHasher, /) -> int:
        key = getattr(obj, "key", None)
        if key is None:
            msg = "ZoneInfo objects must expose a stable key."
            raise HashError(msg)
        return hash_values(key)


class IPAddressHandler(_ProjectedPrimitive):
    _types = (
        ipaddress.IPv4Address,
        ipaddress.IPv6Address,
        ipaddress.IPv4Network,
        ipaddress.IPv6Network,
        ipaddress.IPv4Interface,
        ipaddress.IPv6Interface,
    )

    _project = str


class ArrayHandler(_InstanceMatch, PrimitiveHandler):
    _types = array.array

    @classmethod
    def digest(cls, obj: Any, _element_hash: ElementHasher, /) -> int:
        return hash_values((obj.typecode, obj.tolist()))


class EllipsisHandler(PrimitiveHandler):
    """Hash the Ellipsis singleton."""

    @staticmethod
    def match(obj: Any) -> bool:
        return obj is ...

    @classmethod
    def digest(cls, _obj: Any, _element_hash: ElementHasher, /) -> int:
        return hash_values("...")


class TypeHandler(_InstanceMatch, Handler):
    """Hash type objects (classes) by their qualified name.

    Parameterized generic aliases (``list[int]``, ``dict[str, int]``,
    ``tuple[int, ...]``) are hashed by origin *and* type arguments, so
    that ``list[int]`` and ``list[str]`` are distinguishable.  Without
    the args, :func:`qualified_type_name` on a ``types.GenericAlias``
    returns only the origin's qualified name (``builtins.list``) and
    every parameterization of the same origin would collide.
    """

    _types = (type, types.GenericAlias)

    @classmethod
    def digest(cls, obj: Any, element_hash: ElementHasher, /) -> int:
        origin = getattr(obj, "__origin__", None)
        if origin is None:
            return hash_values(qualified_type_name(obj))
        # Generic alias: hash origin + args.  Args are recursed through
        # ``element_hash`` so nested aliases (``list[list[int]]``) and
        # non-type args (``tuple[int, ...]`` → Ellipsis) are handled
        # uniformly.
        args = getattr(obj, "__args__", ())
        return hash_values(
            (
                qualified_type_name(origin),
                [element_hash(arg) for arg in args],
            )
        )


class SimpleNamespaceHandler(_InstanceMatch, Handler):
    _types = types.SimpleNamespace

    @classmethod
    def digest(cls, obj: Any, element_hash: ElementHasher, /) -> int:
        return _digest_mapping_items(vars(obj).items(), element_hash)


class UserDictHandler(_InstanceMatch, Handler):
    _types = UserDict

    @classmethod
    def digest(cls, obj: Any, element_hash: ElementHasher, /) -> int:
        return _digest_mapping_items(obj.data.items(), element_hash)


class UserListHandler(_InstanceMatch, CollectionHandler):
    _types = UserList

    @staticmethod
    def elements(obj: Any) -> list[Any]:
        return list(obj.data)


class UserStringHandler(_ProjectedPrimitive):
    _types = UserString
    _project = str


class NamedTupleHandler(CollectionHandler):
    """Hash named tuples by field name/value pairs (like DataclassHandler)."""

    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, tuple) and hasattr(type(obj), "_fields")

    @staticmethod
    def elements(obj: Any) -> list[Any]:
        return [(f, getattr(obj, f)) for f in type(obj)._fields]


class ListHandler(_InstanceMatch, CollectionHandler):
    _types = (list, tuple, set, frozenset)

    @staticmethod
    def elements(obj: Any) -> list[Any] | set[Any]:
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, frozenset):
            return set(obj)
        return obj


class DequeHandler(_InstanceMatch, CollectionHandler):
    _types = deque

    @staticmethod
    def elements(obj: Any) -> list[Any]:
        return list(obj)


class OrderedDictHandler(_InstanceMatch, CollectionHandler):
    _types = OrderedDict

    @staticmethod
    def elements(obj: Any) -> list[Any]:
        return list(obj.items())


class DefaultDictHandler(_InstanceMatch, Handler):
    _types = defaultdict

    @classmethod
    def digest(cls, obj: Any, element_hash: ElementHasher, /) -> int:
        default_factory_hash = None if obj.default_factory is None else element_hash(obj.default_factory)
        return hash_values(
            (
                default_factory_hash,
                _digest_mapping_items(obj.items(), element_hash),
            )
        )


class CounterHandler(_InstanceMatch, Handler):
    _types = Counter

    @classmethod
    def digest(cls, obj: Any, element_hash: ElementHasher, /) -> int:
        return _digest_mapping_items(obj.items(), element_hash)


class DictHandler(_InstanceMatch, Handler):
    _types = dict
    _excluded_types = (OrderedDict, defaultdict, Counter)

    @classmethod
    def digest(cls, obj: Any, element_hash: ElementHasher, /) -> int:
        return _digest_mapping_items(obj.items(), element_hash)


class DictKeysViewHandler(_InstanceMatch, CollectionHandler):
    _types = _DICT_KEYS_TYPE

    @staticmethod
    def elements(obj: Any) -> set[Any]:
        return set(obj)


class DictValuesViewHandler(_InstanceMatch, CollectionHandler):
    _types = _DICT_VALUES_TYPE

    @staticmethod
    def elements(obj: Any) -> list[Any]:
        return list(obj)


class DictItemsViewHandler(_InstanceMatch, Handler):
    _types = _DICT_ITEMS_TYPE

    @classmethod
    def digest(cls, obj: Any, element_hash: ElementHasher, /) -> int:
        return _digest_mapping_items(obj, element_hash)


class ChainMapHandler(_InstanceMatch, Handler):
    _types = ChainMap

    @classmethod
    def digest(cls, obj: Any, element_hash: ElementHasher, /) -> int:
        return hash_values([_digest_mapping_items(mapping.items(), element_hash) for mapping in obj.maps])


class MappingProxyHandler(_InstanceMatch, Handler):
    _types = types.MappingProxyType

    @classmethod
    def digest(cls, obj: Any, element_hash: ElementHasher, /) -> int:
        return _digest_mapping_items(obj.items(), element_hash)


class DataclassHandler(CollectionHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return dataclasses.is_dataclass(obj) and not isinstance(obj, type)

    @staticmethod
    def elements(obj: Any) -> list[Any]:
        return [(f.name, getattr(obj, f.name)) for f in dataclasses.fields(obj)]


stdlib_handlers: list[type[Handler]] = [
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
