"""Serializers for Python builtins and standard library types."""

import dataclasses
import datetime
import decimal
import enum
import fractions
import importlib
import ipaddress
import pathlib
import re
import types
import uuid
import zoneinfo
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any

import msgspec.msgpack

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerClass,
    SerializerTypeList,
    SerializerTypeRegistry,
    qualified_type_name,
    write_meta,
)

__all__ = ["stdlib_serializers", "stdlib_serializers_by_type"]

# ---------------------------------------------------------------------------
# Tagged encoding/decoding for msgpack round-trip fidelity
# ---------------------------------------------------------------------------

_TAG = "__t"
_VAL = "v"

# Types that pass through msgpack without tagging.
_PASSTHROUGH_TYPES = (type(None), bool, int, float, str, bytes)

# All types supported by the tagged encoding (for _is_msgpack_safe).
_SAFE_SCALAR_TYPES = (
    type(None),
    bool,
    int,
    float,
    str,
    bytes,
    bytearray,
    complex,
    datetime.datetime,
    datetime.date,
    datetime.time,
    datetime.timedelta,
    uuid.UUID,
    decimal.Decimal,
    fractions.Fraction,
    range,
    slice,
    re.Pattern,
    zoneinfo.ZoneInfo,
    ipaddress.IPv4Address,
    ipaddress.IPv6Address,
    ipaddress.IPv4Network,
    ipaddress.IPv6Network,
    ipaddress.IPv4Interface,
    ipaddress.IPv6Interface,
)


def _is_msgpack_safe(obj: Any, _seen: set[int] | None = None) -> bool:  # noqa: PLR0911
    """Check whether *obj* can be fully encoded by the tagged msgpack scheme.

    Uses an ``id()`` set to detect circular references.
    """
    if _seen is None:
        _seen = set()

    if isinstance(obj, _PASSTHROUGH_TYPES):
        return True

    if isinstance(obj, _SAFE_SCALAR_TYPES):
        return True

    if isinstance(obj, enum.Enum):
        return True

    if isinstance(obj, pathlib.PurePath):
        return True

    if isinstance(obj, types.SimpleNamespace):
        return True  # checked recursively below via vars()

    obj_id = id(obj)
    if obj_id in _seen:
        return False  # circular reference
    _seen.add(obj_id)

    try:
        if isinstance(obj, types.SimpleNamespace):
            return all(_is_msgpack_safe(v, _seen) for v in vars(obj).values())

        if isinstance(obj, dict):
            return all(_is_msgpack_safe(k, _seen) and _is_msgpack_safe(v, _seen) for k, v in obj.items())

        if isinstance(obj, deque):
            return all(_is_msgpack_safe(item, _seen) for item in obj)

        if isinstance(obj, (list, tuple, set, frozenset)):
            return all(_is_msgpack_safe(item, _seen) for item in obj)
    finally:
        _seen.discard(obj_id)

    return False


def _encode_tagged(obj: Any) -> Any:  # noqa: PLR0911
    """Convert *obj* into a msgpack-native representation with type tags."""
    # Passthrough types: msgpack handles these natively and round-trips them.
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, bytes):
        return obj

    # Tagged scalar types.
    if isinstance(obj, bytearray):
        return {_TAG: "bytearray", _VAL: bytes(obj)}

    if isinstance(obj, complex):
        return {_TAG: "complex", _VAL: [obj.real, obj.imag]}

    if isinstance(obj, datetime.datetime):
        return {_TAG: "datetime", _VAL: obj.isoformat()}

    if isinstance(obj, datetime.date):
        return {_TAG: "date", _VAL: obj.isoformat()}

    if isinstance(obj, datetime.time):
        return {_TAG: "time", _VAL: obj.isoformat()}

    if isinstance(obj, datetime.timedelta):
        return {_TAG: "timedelta", _VAL: [obj.days, obj.seconds, obj.microseconds]}

    if isinstance(obj, uuid.UUID):
        return {_TAG: "uuid", _VAL: obj.hex}

    if isinstance(obj, decimal.Decimal):
        return {_TAG: "decimal", _VAL: str(obj)}

    if isinstance(obj, fractions.Fraction):
        return {_TAG: "fraction", _VAL: [obj.numerator, obj.denominator]}

    if isinstance(obj, enum.Enum):
        return {_TAG: "enum", _VAL: _encode_tagged(obj.value), "cls": qualified_type_name(type(obj))}

    if isinstance(obj, pathlib.PurePath):
        return {_TAG: "path", _VAL: str(obj), "cls": type(obj).__name__}

    if isinstance(obj, range):
        return {_TAG: "range", _VAL: [obj.start, obj.stop, obj.step]}

    if isinstance(obj, slice):
        return {_TAG: "slice", _VAL: [_encode_tagged(obj.start), _encode_tagged(obj.stop), _encode_tagged(obj.step)]}

    if isinstance(obj, re.Pattern):
        return {_TAG: "pattern", _VAL: obj.pattern, "flags": obj.flags}

    if isinstance(obj, zoneinfo.ZoneInfo):
        return {_TAG: "zoneinfo", _VAL: str(obj)}

    if isinstance(
        obj,
        (
            ipaddress.IPv4Address,
            ipaddress.IPv6Address,
            ipaddress.IPv4Network,
            ipaddress.IPv6Network,
            ipaddress.IPv4Interface,
            ipaddress.IPv6Interface,
        ),
    ):
        return {_TAG: "ipaddress", _VAL: str(obj), "cls": type(obj).__name__}

    if isinstance(obj, types.SimpleNamespace):
        return {_TAG: "namespace", _VAL: {k: _encode_tagged(v) for k, v in vars(obj).items()}}

    # Containers.
    # NamedTuple must be checked before plain tuple.
    if isinstance(obj, tuple) and hasattr(type(obj), "_fields"):
        return {
            _TAG: "namedtuple",
            _VAL: {f: _encode_tagged(getattr(obj, f)) for f in type(obj)._fields},
            "cls": qualified_type_name(type(obj)),
        }

    if isinstance(obj, OrderedDict):
        return {_TAG: "OrderedDict", _VAL: [[_encode_tagged(k), _encode_tagged(v)] for k, v in obj.items()]}

    if isinstance(obj, deque):
        return {_TAG: "deque", _VAL: [_encode_tagged(item) for item in obj], "maxlen": obj.maxlen}

    if isinstance(obj, dict):
        encoded = {_encode_tagged(k): _encode_tagged(v) for k, v in obj.items()}
        # Escape dicts that collide with our tag convention.
        if _TAG in encoded:
            return {_TAG: "escaped_dict", _VAL: list(encoded.items())}
        return encoded

    if isinstance(obj, list):
        return [_encode_tagged(item) for item in obj]

    if isinstance(obj, tuple):
        return {_TAG: "tuple", _VAL: [_encode_tagged(item) for item in obj]}

    if isinstance(obj, frozenset):
        return {_TAG: "frozenset", _VAL: sorted((_encode_tagged(item) for item in obj), key=repr)}

    if isinstance(obj, set):
        return {_TAG: "set", _VAL: sorted((_encode_tagged(item) for item in obj), key=repr)}

    msg = f"_encode_tagged does not support {type(obj).__name__!r}."
    raise TypeError(msg)


_PATH_CLS_MAP: dict[str, type[pathlib.PurePath]] = {
    "PurePosixPath": pathlib.PurePosixPath,
    "PureWindowsPath": pathlib.PureWindowsPath,
    "PosixPath": pathlib.PosixPath,
    "WindowsPath": pathlib.WindowsPath,
    "Path": pathlib.Path,
}


def _import_type(qualified_name: str) -> type[Any]:
    """Import a type by its ``module.qualname``."""
    module_name, _, attr_name = qualified_name.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)  # type: ignore[no-any-return]


def _decode_tagged(obj: Any) -> Any:  # noqa: PLR0911
    """Restore Python objects from their tagged msgpack representation."""
    if isinstance(obj, dict):
        tag = obj.get(_TAG)
        if tag is not None:
            val = obj[_VAL]
            if tag == "tuple":
                return tuple(_decode_tagged(item) for item in val)
            if tag == "set":
                return {_decode_tagged(item) for item in val}
            if tag == "frozenset":
                return frozenset(_decode_tagged(item) for item in val)
            if tag == "bytearray":
                return bytearray(val)
            if tag == "complex":
                return complex(val[0], val[1])
            if tag == "datetime":
                return datetime.datetime.fromisoformat(val)
            if tag == "date":
                return datetime.date.fromisoformat(val)
            if tag == "time":
                return datetime.time.fromisoformat(val)
            if tag == "timedelta":
                return datetime.timedelta(days=val[0], seconds=val[1], microseconds=val[2])
            if tag == "uuid":
                return uuid.UUID(val)
            if tag == "decimal":
                return decimal.Decimal(val)
            if tag == "fraction":
                return fractions.Fraction(val[0], val[1])
            if tag == "enum":
                enum_cls = _import_type(obj["cls"])
                return enum_cls(_decode_tagged(val))
            if tag == "path":
                path_cls = _PATH_CLS_MAP.get(obj["cls"], pathlib.PurePosixPath)
                return path_cls(val)
            if tag == "range":
                return range(val[0], val[1], val[2])
            if tag == "slice":
                return slice(_decode_tagged(val[0]), _decode_tagged(val[1]), _decode_tagged(val[2]))
            if tag == "pattern":
                return re.compile(val, obj.get("flags", 0))
            if tag == "zoneinfo":
                return zoneinfo.ZoneInfo(val)
            if tag == "ipaddress":
                ip_cls = getattr(ipaddress, obj["cls"])
                return ip_cls(val)
            if tag == "namespace":
                return types.SimpleNamespace(**{k: _decode_tagged(v) for k, v in val.items()})
            if tag == "namedtuple":
                nt_cls = _import_type(obj["cls"])
                return nt_cls(**{k: _decode_tagged(v) for k, v in val.items()})
            if tag == "OrderedDict":
                return OrderedDict((_decode_tagged(k), _decode_tagged(v)) for k, v in val)
            if tag == "deque":
                return deque((_decode_tagged(item) for item in val), maxlen=obj.get("maxlen"))
            if tag == "escaped_dict":
                return {_decode_tagged(k): _decode_tagged(v) for k, v in val}
            msg = f"Unknown serde tag: {tag!r}"
            raise ValueError(msg)
        # Regular dict — recurse into keys and values.
        return {_decode_tagged(k): _decode_tagged(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_decode_tagged(item) for item in obj]

    return obj


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------


class MsgpackSerializer(Serializer[Any]):
    """Serialize objects composed of builtins and stdlib value types via msgpack."""

    version = 1

    @staticmethod
    def match(obj: Any) -> bool:
        return _is_msgpack_safe(obj)

    @staticmethod
    def save(obj: Any, directory: Path) -> None:
        tagged = _encode_tagged(obj)
        data = msgspec.msgpack.encode(tagged)
        (directory / "data.msgpack").write_bytes(data)
        write_meta(directory, MsgpackSerializer)

    @staticmethod
    def load(directory: Path) -> Any:
        data = (directory / "data.msgpack").read_bytes()
        raw = msgspec.msgpack.decode(data)
        return _decode_tagged(raw)


class BytesSerializer(Serializer[bytes | bytearray]):
    """Serialize raw ``bytes`` or ``bytearray`` objects."""

    version = 1

    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, (bytes, bytearray))

    @staticmethod
    def save(obj: bytes | bytearray, directory: Path) -> None:
        (directory / "data.bin").write_bytes(obj)
        write_meta(directory, BytesSerializer, original_type=type(obj).__name__)

    @staticmethod
    def load(directory: Path) -> bytes | bytearray:
        from misen.utils.serde.serializer_base import read_meta

        raw = (directory / "data.bin").read_bytes()
        meta = read_meta(directory)
        if meta and meta.get("original_type") == "bytearray":
            return bytearray(raw)
        return raw


class DataclassSerializer(Serializer[Any]):
    """Serialize ``@dataclass`` instances with msgpack-safe fields."""

    version = 1

    @staticmethod
    def match(obj: Any) -> bool:
        if not (dataclasses.is_dataclass(obj) and not isinstance(obj, type)):
            return False
        try:
            return _is_msgpack_safe(dataclasses.asdict(obj))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def save(obj: Any, directory: Path) -> None:
        data = dataclasses.asdict(obj)
        tagged = _encode_tagged(data)
        encoded = msgspec.msgpack.encode(tagged)
        (directory / "data.msgpack").write_bytes(encoded)
        write_meta(directory, DataclassSerializer, dataclass_type=qualified_type_name(type(obj)))

    @staticmethod
    def load(directory: Path) -> Any:
        from misen.utils.serde.serializer_base import read_meta

        meta = read_meta(directory)
        if meta is None:
            msg = "DataclassSerializer requires serde_meta.json"
            raise ValueError(msg)

        raw = msgspec.msgpack.decode((directory / "data.msgpack").read_bytes())
        data = _decode_tagged(raw)

        cls = _import_type(meta["dataclass_type"])
        return cls(**data)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

stdlib_serializers: SerializerTypeList = [
    BytesSerializer,
    DataclassSerializer,
    MsgpackSerializer,  # broadest match — must be last
]

_stdlib_serializers_by_type: dict[type[Any], SerializerClass] = {
    bytes: BytesSerializer,
    bytearray: BytesSerializer,
    type(None): MsgpackSerializer,
    bool: MsgpackSerializer,
    int: MsgpackSerializer,
    float: MsgpackSerializer,
    str: MsgpackSerializer,
    list: MsgpackSerializer,
    tuple: MsgpackSerializer,
    set: MsgpackSerializer,
    frozenset: MsgpackSerializer,
    dict: MsgpackSerializer,
    complex: MsgpackSerializer,
    datetime.datetime: MsgpackSerializer,
    datetime.date: MsgpackSerializer,
    datetime.time: MsgpackSerializer,
    datetime.timedelta: MsgpackSerializer,
    uuid.UUID: MsgpackSerializer,
    decimal.Decimal: MsgpackSerializer,
    fractions.Fraction: MsgpackSerializer,
    range: MsgpackSerializer,
    slice: MsgpackSerializer,
    pathlib.PurePath: MsgpackSerializer,
    pathlib.PurePosixPath: MsgpackSerializer,
    pathlib.PureWindowsPath: MsgpackSerializer,
    pathlib.PosixPath: MsgpackSerializer,
    pathlib.WindowsPath: MsgpackSerializer,
    re.Pattern: MsgpackSerializer,
    zoneinfo.ZoneInfo: MsgpackSerializer,
    ipaddress.IPv4Address: MsgpackSerializer,
    ipaddress.IPv6Address: MsgpackSerializer,
    ipaddress.IPv4Network: MsgpackSerializer,
    ipaddress.IPv6Network: MsgpackSerializer,
    ipaddress.IPv4Interface: MsgpackSerializer,
    ipaddress.IPv6Interface: MsgpackSerializer,
    types.SimpleNamespace: MsgpackSerializer,
    OrderedDict: MsgpackSerializer,
    deque: MsgpackSerializer,
}

stdlib_serializers_by_type: SerializerTypeRegistry = {
    qualified_type_name(obj_type): ser_cls for obj_type, ser_cls in _stdlib_serializers_by_type.items()
}
