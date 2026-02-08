import dataclasses
import datetime
import decimal
import enum
import math
import uuid
from collections import OrderedDict
from typing import Any

from misen_hash import CollectionHandler, Handler, PrimitiveHandler
from misen_hash.utils import hash_msgspec

__all__ = ["builtin_handlers", "builtin_handlers_by_type"]


class NoneHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, type(None))

    @staticmethod
    def digest(obj: Any, element_hash: None = None) -> int:
        return hash_msgspec(obj)


class EnumHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, enum.Enum)

    @staticmethod
    def digest(obj: Any, element_hash: None = None) -> int:
        return hash_msgspec(obj.value)


class BoolHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, bool)

    @staticmethod
    def digest(obj: Any, element_hash: None = None) -> int:
        return hash_msgspec(bool(obj))


class IntHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, int) and not isinstance(obj, (bool, enum.Enum))

    @staticmethod
    def digest(obj: Any, element_hash: None = None) -> int:
        return hash_msgspec(int(obj))


class FloatHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, float)

    @staticmethod
    def digest(obj: Any, element_hash: None = None) -> int:
        obj = float(obj)
        if math.isfinite(obj):
            return hash_msgspec(obj)
        return hash_msgspec(str(obj))


class StrHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, str)

    @staticmethod
    def digest(obj: Any, element_hash: None = None) -> int:
        return hash_msgspec(str(obj))


class BytearrayHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, bytearray)

    @staticmethod
    def digest(obj: Any, element_hash: None = None) -> int:
        return hash_msgspec(bytearray(obj))


class BytesHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, bytes)

    @staticmethod
    def digest(obj: Any, element_hash: None = None) -> int:
        return hash_msgspec(bytes(obj))


class DatetimeHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, datetime.datetime)

    @staticmethod
    def digest(obj: Any, element_hash: None = None) -> int:
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
    def digest(obj: Any, element_hash: None = None) -> int:
        return hash_msgspec(datetime.date(obj.year, obj.month, obj.day))


class TimeHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, datetime.time)

    @staticmethod
    def digest(obj: Any, element_hash: None = None) -> int:
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
    def digest(obj: Any, element_hash: None = None) -> int:
        return hash_msgspec(datetime.timedelta(days=obj.days, seconds=obj.seconds, microseconds=obj.microseconds))


class UUIDHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, uuid.UUID)

    @staticmethod
    def digest(obj: Any, element_hash: None = None) -> int:
        return hash_msgspec(uuid.UUID(bytes=obj.bytes))


class DecimalHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, decimal.Decimal)

    @staticmethod
    def digest(obj: Any, element_hash: None = None) -> int:
        return hash_msgspec(decimal.Decimal(str(obj)))


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


class OrderedDictHandler(CollectionHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, OrderedDict)

    @staticmethod
    def elements(obj: Any) -> list[Any]:
        return list(obj.items())


class DictHandler(CollectionHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, dict) and not isinstance(obj, OrderedDict)

    @staticmethod
    def elements(obj: Any) -> set[Any]:
        return set(obj.items())


class DataclassHandler(CollectionHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        return dataclasses.is_dataclass(obj)

    @staticmethod
    def elements(obj: Any) -> set[Any]:
        return {(f.name, getattr(obj, f.name)) for f in dataclasses.fields(obj)}


builtin_handlers = [
    NoneHandler,
    EnumHandler,
    BoolHandler,
    IntHandler,
    FloatHandler,
    StrHandler,
    BytearrayHandler,
    BytesHandler,
    DatetimeHandler,
    DateHandler,
    TimeHandler,
    TimedeltaHandler,
    UUIDHandler,
    DecimalHandler,
    ListHandler,
    OrderedDictHandler,
    DictHandler,
    DataclassHandler,
]

builtin_handlers_by_type: dict[type[Any], Handler] = {
    None.__class__: NoneHandler,
    enum.Enum: EnumHandler,
    enum.IntEnum: EnumHandler,
    enum.Flag: EnumHandler,
    enum.IntFlag: EnumHandler,
    bool: BoolHandler,
    int: IntHandler,
    float: FloatHandler,
    str: StrHandler,
    bytearray: BytearrayHandler,
    bytes: BytesHandler,
    datetime.datetime: DatetimeHandler,
    datetime.date: DateHandler,
    datetime.time: TimeHandler,
    datetime.timedelta: TimedeltaHandler,
    uuid.UUID: UUIDHandler,
    decimal.Decimal: DecimalHandler,
    list: ListHandler,
    OrderedDict: OrderedDictHandler,
    dict: DictHandler,
}
