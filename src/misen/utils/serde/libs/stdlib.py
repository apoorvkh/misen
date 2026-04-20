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
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import msgspec.msgpack

from misen.utils.serde import (
    Serializer,
    SerializerTypeRegistry,
    UnserializableTypeError,
)
from misen.utils.type_registry import qualified_type_name

__all__ = ["stdlib_serializers", "stdlib_serializers_by_type"]

# ---------------------------------------------------------------------------
# Tagged encoding/decoding for msgpack round-trip fidelity
# ---------------------------------------------------------------------------
#
# msgspec.msgpack natively encodes only a subset of Python types
# (None, bool, int within [-2**63, 2**64-1], float, str, bytes, list, dict,
# set, frozenset, and timezone-aware datetime).  Untyped decode reverses
# only the set with a 1:1 wire mapping — sets/frozensets/tuples all decode
# as ``list``, naive datetimes/uuids/decimals decode as ``str``, and so on.
#
# To round-trip those types we tag values that would otherwise lose their
# Python identity, then unwrap them on decode.  See ``test_serde_stdlib.py``
# for the executable contract.
#
# Dict *keys* are passed through to msgspec without tagging: msgspec.msgpack
# already supports str/int/float/bool/None/bytes/datetime/tuple keys
# natively, and tagging would turn a hashable key into an unhashable dict.
# Keys whose type msgspec cannot encode raise ``UnserializableTypeError``
# from ``MsgpackSerializer.write``.

_TAG = "__t"
_VAL = "v"


def _encode_tagged(obj: Any) -> Any:
    """Convert *obj* into a msgpack-native representation with type tags.

    Strict ``type(obj) is X`` checks are used for builtin types where
    silently downcasting a subclass would lose information; subclasses
    fall through and raise ``TypeError`` so ``MsgpackSerializer.write``
    can wrap them in :class:`UnserializableTypeError`.
    """
    # Passthrough primitives — strict type check so subclasses don't slip through.
    if obj is None:
        return None
    if type(obj) is bool:
        return obj
    if type(obj) is int:
        return obj
    if type(obj) is float:
        return obj
    if type(obj) is str:
        return obj
    if type(obj) is bytes:
        return obj

    # Tagged scalar types.
    if type(obj) is bytearray:
        return {_TAG: "bytearray", _VAL: bytes(obj)}

    if type(obj) is complex:
        return {_TAG: "complex", _VAL: [obj.real, obj.imag]}

    if type(obj) is datetime.datetime:
        return {_TAG: "datetime", _VAL: obj.isoformat(), "fold": obj.fold}

    if type(obj) is datetime.date:
        return {_TAG: "date", _VAL: obj.isoformat()}

    if type(obj) is datetime.time:
        return {_TAG: "time", _VAL: obj.isoformat(), "fold": obj.fold}

    if type(obj) is datetime.timedelta:
        return {_TAG: "timedelta", _VAL: [obj.days, obj.seconds, obj.microseconds]}

    if type(obj) is uuid.UUID:
        return {_TAG: "uuid", _VAL: obj.hex}

    if type(obj) is decimal.Decimal:
        return {_TAG: "decimal", _VAL: str(obj)}

    if type(obj) is fractions.Fraction:
        return {_TAG: "fraction", _VAL: [obj.numerator, obj.denominator]}

    if isinstance(obj, enum.Enum):
        return {_TAG: "enum", _VAL: _encode_tagged(obj.value), "cls": qualified_type_name(type(obj))}

    if isinstance(obj, pathlib.PurePath):
        return {_TAG: "path", _VAL: str(obj), "cls": qualified_type_name(type(obj))}

    if type(obj) is range:
        return {_TAG: "range", _VAL: [obj.start, obj.stop, obj.step]}

    if type(obj) is slice:
        return {_TAG: "slice", _VAL: [_encode_tagged(obj.start), _encode_tagged(obj.stop), _encode_tagged(obj.step)]}

    if isinstance(obj, re.Pattern):
        # Pattern can be either str or bytes; both pass through msgspec natively.
        return {_TAG: "pattern", _VAL: obj.pattern, "flags": obj.flags}

    if type(obj) is zoneinfo.ZoneInfo:
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

    if type(obj) is types.SimpleNamespace:
        return {_TAG: "namespace", _VAL: {k: _encode_tagged(v) for k, v in vars(obj).items()}}

    # Containers.
    # NamedTuple must be checked before plain tuple.
    if isinstance(obj, tuple) and hasattr(type(obj), "_fields"):
        return {
            _TAG: "namedtuple",
            _VAL: {f: _encode_tagged(getattr(obj, f)) for f in type(obj)._fields},
            "cls": qualified_type_name(type(obj)),
        }

    if type(obj) is OrderedDict:
        return {_TAG: "OrderedDict", _VAL: [[k, _encode_tagged(v)] for k, v in obj.items()]}

    if type(obj) is deque:
        return {_TAG: "deque", _VAL: [_encode_tagged(item) for item in obj], "maxlen": obj.maxlen}

    # Dataclass instances — tagged with qualified class name and field values.
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            _TAG: "dataclass",
            _VAL: {f.name: _encode_tagged(getattr(obj, f.name)) for f in dataclasses.fields(obj)},
            "cls": qualified_type_name(type(obj)),
        }

    if type(obj) is dict:
        # Keys pass through unchanged so msgspec.msgpack handles them natively
        # (str/int/float/bool/None/bytes/datetime/tuple keys are supported).
        encoded = {k: _encode_tagged(v) for k, v in obj.items()}
        # Escape dicts whose contents would collide with our tag convention.
        if _TAG in encoded:
            return {_TAG: "escaped_dict", _VAL: list(encoded.items())}
        return encoded

    if type(obj) is list:
        return [_encode_tagged(item) for item in obj]

    if type(obj) is tuple:
        return {_TAG: "tuple", _VAL: [_encode_tagged(item) for item in obj]}

    if type(obj) is frozenset:
        return {_TAG: "frozenset", _VAL: sorted((_encode_tagged(item) for item in obj), key=repr)}

    if type(obj) is set:
        return {_TAG: "set", _VAL: sorted((_encode_tagged(item) for item in obj), key=repr)}

    msg = f"Cannot encode value of type {qualified_type_name(type(obj))!r} for msgpack serialization."
    raise TypeError(msg)


def _import_type(qualified_name: str) -> type[Any]:
    """Import a type by its ``module.qualname``."""
    module_name, _, attr_name = qualified_name.rpartition(".")
    module = importlib.import_module(module_name)
    cls: type[Any] = getattr(module, attr_name)
    return cls


def _decode_tagged(obj: Any) -> Any:
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
                return datetime.datetime.fromisoformat(val).replace(fold=obj.get("fold", 0))
            if tag == "date":
                return datetime.date.fromisoformat(val)
            if tag == "time":
                return datetime.time.fromisoformat(val).replace(fold=obj.get("fold", 0))
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
                return _import_type(obj["cls"])(val)
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
                return OrderedDict((k, _decode_tagged(v)) for k, v in val)
            if tag == "deque":
                return deque((_decode_tagged(item) for item in val), maxlen=obj.get("maxlen"))
            if tag == "dataclass":
                cls = _import_type(obj["cls"])
                return cls(**{k: _decode_tagged(v) for k, v in val.items()})
            if tag == "escaped_dict":
                return {k: _decode_tagged(v) for k, v in val}
            msg = f"Unknown serde tag: {tag!r}"
            raise ValueError(msg)
        # Regular dict — recurse into values; keys were not tagged on encode.
        return {k: _decode_tagged(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_decode_tagged(item) for item in obj]

    return obj


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------


class MsgpackSerializer(Serializer[Any]):
    """Catch-all serializer for builtins, stdlib value types, and dataclasses.

    This is the final fallback in :data:`stdlib_serializers`.  It always
    matches; if the value contains anything ``msgspec.msgpack`` cannot
    encode (e.g. arbitrary-precision integers outside
    ``[-2**63, 2**64-1]``, subclasses of builtin types, or user-defined
    classes without a registered serializer) the failure surfaces from
    :meth:`save` as :class:`UnserializableTypeError`.
    """

    @staticmethod
    def match(obj: Any) -> bool:  # noqa: ARG004
        """Always return True — :class:`MsgpackSerializer` is the catch-all."""
        return True

    @staticmethod
    def write(obj: Any, directory: Path) -> Mapping[str, Any] | None:
        try:
            tagged = _encode_tagged(obj)
            data = msgspec.msgpack.encode(tagged)
        except (TypeError, OverflowError) as exc:
            msg = f"Cannot serialize value of type {qualified_type_name(type(obj))!r} with MsgpackSerializer: {exc}"
            raise UnserializableTypeError(msg) from exc
        (directory / "data.msgpack").write_bytes(data)
        return None

    @staticmethod
    def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
        data = (directory / "data.msgpack").read_bytes()
        raw = msgspec.msgpack.decode(data)
        return _decode_tagged(raw)


class BytesSerializer(Serializer[bytes | bytearray]):
    """Serialize raw ``bytes`` or ``bytearray`` objects."""

    @staticmethod
    def match(obj: Any) -> bool:
        return type(obj) is bytes or type(obj) is bytearray

    @staticmethod
    def write(obj: bytes | bytearray, directory: Path) -> Mapping[str, Any]:
        (directory / "data.bin").write_bytes(obj)
        return {"original_type": type(obj).__name__}

    @staticmethod
    def read(directory: Path, *, meta: Mapping[str, Any]) -> bytes | bytearray:
        raw = (directory / "data.bin").read_bytes()
        if meta.get("original_type") == "bytearray":
            return bytearray(raw)
        return raw


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

stdlib_serializers: list[type[Serializer]] = [
    BytesSerializer,
    MsgpackSerializer,  # catch-all — must be last
]

# NOTE: ``dict`` and ``OrderedDict`` are intentionally omitted from this fast
# path.  They are registered as ``volatile_types`` on the serde registry so
# dispatch re-evaluates per call — that lets content-sensitive serializers
# (e.g. ``DictOfTensorsSerializer`` in ``libs/torch.py``) intercept dicts
# whose values are tensors/ndarrays before falling through to
# :class:`MsgpackSerializer` here as the catch-all.
_stdlib_serializers_by_type: dict[type[Any], type[Serializer]] = {
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
    deque: MsgpackSerializer,
}

stdlib_serializers_by_type: SerializerTypeRegistry = {
    qualified_type_name(obj_type): ser_cls for obj_type, ser_cls in _stdlib_serializers_by_type.items()
}
