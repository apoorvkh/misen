"""Built-in canonical-hash handlers for core Python and stdlib value types."""

import array
import dataclasses
import datetime
import decimal
import enum
import fractions
import math
import pathlib
import re
import types
import uuid
from collections import ChainMap, Counter, OrderedDict, defaultdict, deque
from collections.abc import Callable, Iterable
from typing import Any

from misen_hash.handler_base import (
    CollectionHandler,
    Handler,
    HandlerTypeList,
    HandlerTypeRegistry,
    PrimitiveHandler,
    qualified_type_name,
)
from misen_hash.hash import hash_msgspec

__all__ = ["builtin_handlers", "builtin_handlers_by_type"]


def _normalized_float(value: float) -> float | str:
    if math.isfinite(value):
        return value
    return str(value)


def _digest_mapping_items(items: Iterable[tuple[Any, Any]], element_hash: Callable[[Any], int] | None) -> int:
    if element_hash is None:
        msg = "Mapping handlers require element_hash."
        raise ValueError(msg)

    return hash_msgspec(
        {
            hash_msgspec(
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
    def digest(obj: Any) -> int:
        return hash_msgspec(obj)


class EnumHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, enum.Enum)

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_msgspec(obj.value)


class BoolHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, bool)

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_msgspec(bool(obj))


class IntHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, int) and not isinstance(obj, (bool, enum.Enum))

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_msgspec(int(obj))


class FloatHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, float)

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_msgspec(_normalized_float(float(obj)))


class ComplexHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, complex)

    @staticmethod
    def digest(obj: Any) -> int:
        value = complex(obj)
        return hash_msgspec(
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
        return hash_msgspec(str(obj))


class BytearrayHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, bytearray)

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_msgspec(bytes(obj))


class BytesHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, bytes)

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_msgspec(bytes(obj))


class MemoryviewHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, memoryview)

    @staticmethod
    def digest(obj: Any) -> int:
        view = memoryview(obj)
        return hash_msgspec(
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
        return hash_msgspec(
            datetime.datetime(
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
        )


class DateHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, datetime.date) and not isinstance(obj, datetime.datetime)

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_msgspec(datetime.date(obj.year, obj.month, obj.day))


class TimeHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, datetime.time)

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_msgspec(
            datetime.time(
                obj.hour,
                obj.minute,
                obj.second,
                obj.microsecond,
                tzinfo=obj.tzinfo,
                fold=getattr(obj, "fold", 0),
            )
        )


class TimedeltaHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, datetime.timedelta)

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_msgspec(datetime.timedelta(days=obj.days, seconds=obj.seconds, microseconds=obj.microseconds))


class UUIDHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, uuid.UUID)

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_msgspec(uuid.UUID(bytes=obj.bytes))


class DecimalHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, decimal.Decimal)

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_msgspec(decimal.Decimal(str(obj)))


class FractionHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, fractions.Fraction)

    @staticmethod
    def digest(obj: Any) -> int:
        value = fractions.Fraction(obj)
        return hash_msgspec((value.numerator, value.denominator))


class RangeHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, range)

    @staticmethod
    def digest(obj: Any) -> int:
        value = range(obj.start, obj.stop, obj.step)
        return hash_msgspec((value.start, value.stop, value.step))


class SliceHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, slice)

    @staticmethod
    def digest(obj: Any) -> int:
        value = slice(obj.start, obj.stop, obj.step)
        return hash_msgspec((value.start, value.stop, value.step))


class PathHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, pathlib.PurePath)

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_msgspec((obj.drive, obj.root, obj.parts))


class PatternHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, re.Pattern)

    @staticmethod
    def digest(obj: Any) -> int:
        return hash_msgspec((obj.pattern, obj.flags))


class ArrayHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, array.array)

    @staticmethod
    def digest(obj: Any) -> int:
        value = array.array(obj.typecode, obj)
        return hash_msgspec((value.typecode, value.tolist()))


class SimpleNamespaceHandler(Handler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, types.SimpleNamespace)

    @staticmethod
    def digest(obj: Any, element_hash: Callable[[Any], int] | None) -> int:
        return _digest_mapping_items(vars(obj).items(), element_hash=element_hash)


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
        return hash_msgspec(
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


class ChainMapHandler(Handler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, ChainMap)

    @staticmethod
    def digest(obj: Any, element_hash: Callable[[Any], int] | None) -> int:
        return hash_msgspec([_digest_mapping_items(mapping.items(), element_hash=element_hash) for mapping in obj.maps])


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


builtin_handlers: HandlerTypeList = [
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
    ArrayHandler,
    SimpleNamespaceHandler,
    ListHandler,
    DequeHandler,
    OrderedDictHandler,
    DefaultDictHandler,
    CounterHandler,
    DictHandler,
    ChainMapHandler,
    MappingProxyHandler,
    DataclassHandler,
]

# Exact-type and base-type fast-path map by fully-qualified type name.
_builtin_handlers_by_type: dict[type[Any], type[Handler]] = {
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
    array.array: ArrayHandler,
    types.SimpleNamespace: SimpleNamespaceHandler,
    list: ListHandler,
    tuple: ListHandler,
    set: ListHandler,
    frozenset: ListHandler,
    deque: DequeHandler,
    OrderedDict: OrderedDictHandler,
    defaultdict: DefaultDictHandler,
    Counter: CounterHandler,
    dict: DictHandler,
    ChainMap: ChainMapHandler,
    types.MappingProxyType: MappingProxyHandler,
}

builtin_handlers_by_type: HandlerTypeRegistry = {
    qualified_type_name(obj_type): handler_cls for obj_type, handler_cls in _builtin_handlers_by_type.items()
}
