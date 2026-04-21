"""Recursion-aware container serializers + predicate-gated msgpack leaf.

:class:`MsgpackLeafSerializer` is tried first and matches any value
whose entire transitive structure is msgpack-native — so a deeply
nested dict of primitives collapses to a single leaf.  When it
doesn't match (because the structure contains a tensor, ndarray, etc.
somewhere), dispatch falls through to a container serializer
(:class:`DictSerializer`, :class:`ListSerializer`,
:class:`TupleSerializer`) which recurses so each child is dispatched
independently.

Since the predicate runs at every recursion step, msgpack-native
*subtrees* within an otherwise-mixed structure also collapse — e.g.
``{"config": {"lr": 0.001}, "weights": tensor}`` writes ``config`` as
one msgpack leaf and ``weights`` as a tensor leaf.
"""

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

from misen.utils.serde.base import BaseSerializer, Container, DecodeCtx, EncodeCtx, LeafSerializer, Node, Serializer
from misen.utils.type_registry import qualified_type_name

__all__ = [
    "DictSerializer",
    "ListSerializer",
    "MsgpackLeafSerializer",
    "TupleSerializer",
    "stdlib_serializers",
    "stdlib_serializers_by_type",
    "stdlib_volatile_types",
]


# ---------------------------------------------------------------------------
# Tagged msgpack encoder/decoder for stdlib-type round-trip fidelity
# ---------------------------------------------------------------------------
#
# ``msgspec.msgpack`` natively handles a subset of Python types; for the
# rest (sets, frozensets, tuples, datetimes, UUIDs, dataclasses,
# namedtuples, dict-subclasses, ...) we wrap values in a ``{_TAG: ...}``
# envelope on encode and unwrap on decode so they round-trip cleanly.
# :class:`MsgpackLeafSerializer` uses these; :func:`_is_msgpack_native`
# below mirrors the type tree without actually encoding.

_TAG = "__t"
_VAL = "v"


def _encode_tagged(obj: Any) -> Any:
    """Convert *obj* into a msgpack-native representation with type tags.

    Strict ``type(obj) is X`` checks for builtin types where silently
    downcasting a subclass would lose information; subclasses fall
    through and raise ``TypeError``.
    """
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
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            _TAG: "dataclass",
            _VAL: {f.name: _encode_tagged(getattr(obj, f.name)) for f in dataclasses.fields(obj)},
            "cls": qualified_type_name(type(obj)),
        }
    if type(obj) is dict:
        encoded = {k: _encode_tagged(v) for k, v in obj.items()}
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
    return getattr(module, attr_name)


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
                return _import_type(obj["cls"])(_decode_tagged(val))
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
                return getattr(ipaddress, obj["cls"])(val)
            if tag == "namespace":
                return types.SimpleNamespace(**{k: _decode_tagged(v) for k, v in val.items()})
            if tag == "namedtuple":
                return _import_type(obj["cls"])(**{k: _decode_tagged(v) for k, v in val.items()})
            if tag == "OrderedDict":
                return OrderedDict((k, _decode_tagged(v)) for k, v in val)
            if tag == "deque":
                return deque((_decode_tagged(item) for item in val), maxlen=obj.get("maxlen"))
            if tag == "dataclass":
                return _import_type(obj["cls"])(**{k: _decode_tagged(v) for k, v in val.items()})
            if tag == "escaped_dict":
                return {k: _decode_tagged(v) for k, v in val}
            msg = f"Unknown serde tag: {tag!r}"
            raise ValueError(msg)
        return {k: _decode_tagged(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_decode_tagged(item) for item in obj]

    return obj


# ---------------------------------------------------------------------------
# Msgpack-nativity predicate
# ---------------------------------------------------------------------------
#
# Mirrors the type tree of ``_encode_tagged`` in
# :mod:`misen.utils.serde.libs.stdlib` without actually encoding, so
# dispatch can cheaply ask "would _encode_tagged accept this value?"
# before committing to the msgpack leaf path.

# Leaf scalars — no sub-object recursion needed.
_MSGPACK_NATIVE_LEAF_TYPES: frozenset[type] = frozenset({
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
    zoneinfo.ZoneInfo,
})

# Key types msgspec.msgpack accepts natively — keep conservative;
# non-scalar keys (e.g. tuples) exist in principle but are rare and
# complicate the predicate for little gain.
_MSGPACK_NATIVE_KEY_TYPES: frozenset[type] = frozenset({
    str,
    int,
    float,
    bool,
    type(None),
    bytes,
    datetime.datetime,
})


def _is_msgpack_native_key(key: Any) -> bool:
    return type(key) in _MSGPACK_NATIVE_KEY_TYPES


def _is_msgpack_native(obj: Any, _seen: set[int] | None = None) -> bool:
    """Return ``True`` iff ``_encode_tagged`` + ``msgspec.msgpack`` can encode ``obj``.

    Walks the object's transitive structure.  Used by
    :class:`MsgpackLeafSerializer.match` to decide whether the value
    (and everything inside it) belongs in the msgpack batched blob.
    """
    t = type(obj)

    if t in _MSGPACK_NATIVE_LEAF_TYPES:
        return True

    # Polymorphic scalar types that ``_encode_tagged`` accepts via ``isinstance``.
    if isinstance(obj, (enum.Enum, pathlib.PurePath, re.Pattern)):
        return True
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
        return True

    # Recursive types — guard against cycles.
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return False
    _seen.add(oid)
    try:
        if t is slice:
            return all(_is_msgpack_native(x, _seen) for x in (obj.start, obj.stop, obj.step))

        if t is types.SimpleNamespace:
            return all(_is_msgpack_native(v, _seen) for v in vars(obj).values())

        # NamedTuple — must come before the plain ``tuple`` check.
        if isinstance(obj, tuple) and hasattr(t, "_fields"):
            return all(_is_msgpack_native(getattr(obj, f), _seen) for f in t._fields)

        if t is dict or t is OrderedDict:
            return all(
                _is_msgpack_native_key(k) and _is_msgpack_native(v, _seen) for k, v in obj.items()
            )

        if t in (list, tuple, set, frozenset, deque):
            return all(_is_msgpack_native(item, _seen) for item in obj)

        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return all(_is_msgpack_native(getattr(obj, f.name), _seen) for f in dataclasses.fields(obj))

        return False
    finally:
        _seen.discard(oid)


class DictSerializer(BaseSerializer[Any]):
    """Recursive serializer for ``dict`` / ``OrderedDict`` with ``str`` keys.

    Non-str-keyed dicts fall through to :class:`MsgpackLeafSerializer`
    which handles them via the tagged encoder.  Each value is
    recursively dispatched, so a dict of tensors produces a
    :class:`Container` whose leaves are batched by
    :class:`TorchTensorSerializer`.
    """

    @staticmethod
    def match(obj: Any) -> bool:
        if type(obj) is not dict and type(obj) is not OrderedDict:
            return False
        return all(isinstance(k, str) for k in obj)

    @classmethod
    def encode(cls, obj: Any, ctx: EncodeCtx) -> Node:
        children = {k: ctx.encode(v) for k, v in obj.items()}
        meta: dict[str, Any] = {}
        if type(obj) is OrderedDict:
            meta["type"] = "OrderedDict"
        return Container(serializer=qualified_type_name(cls), children=children, meta=meta)

    @classmethod
    def decode(cls, node: Node, ctx: DecodeCtx) -> Any:
        assert isinstance(node, Container)
        items = [(k, ctx.decode(v)) for k, v in node.children.items()]
        if node.meta.get("type") == "OrderedDict":
            return OrderedDict(items)
        return dict(items)


class ListSerializer(BaseSerializer[Any]):
    """Recursive serializer for ``list``."""

    @staticmethod
    def match(obj: Any) -> bool:
        return type(obj) is list

    @classmethod
    def encode(cls, obj: Any, ctx: EncodeCtx) -> Node:
        return Container(
            serializer=qualified_type_name(cls),
            children=[ctx.encode(v) for v in obj],
            meta={},
        )

    @classmethod
    def decode(cls, node: Node, ctx: DecodeCtx) -> Any:
        assert isinstance(node, Container)
        return [ctx.decode(c) for c in node.children]


class TupleSerializer(BaseSerializer[Any]):
    """Recursive serializer for ``tuple`` (plain, non-namedtuple)."""

    @staticmethod
    def match(obj: Any) -> bool:
        # Exclude namedtuple — let msgpack handle it for now.
        return type(obj) is tuple

    @classmethod
    def encode(cls, obj: Any, ctx: EncodeCtx) -> Node:
        return Container(
            serializer=qualified_type_name(cls),
            children=[ctx.encode(v) for v in obj],
            meta={},
        )

    @classmethod
    def decode(cls, node: Node, ctx: DecodeCtx) -> Any:
        assert isinstance(node, Container)
        return tuple(ctx.decode(c) for c in node.children)


class MsgpackLeafSerializer(LeafSerializer[Any]):
    """Predicate-gated leaf for values that are fully msgpack-native.

    All leaves of kind ``"msgpack"`` in a save end up in one
    ``data.msgpack`` file keyed by ``leaf_id``.  By gating on
    :func:`_is_msgpack_native`, this serializer claims whole subtrees
    that contain no specialized types (tensors, ndarrays, ...) so a
    primitive-only dict/list/tuple is encoded as a *single* leaf
    rather than recursed into a Container of per-item leaves.

    Because the match predicate runs at every recursion step, msgpack
    subtrees within otherwise-mixed structures also collapse.
    """

    leaf_kind = "msgpack"

    @staticmethod
    def match(obj: Any) -> bool:
        return _is_msgpack_native(obj)

    @staticmethod
    def write_batch(
        entries: list[tuple[str, Any, Mapping[str, Any]]],
        directory: Path,
    ) -> Mapping[str, Any]:
        blob = {leaf_id: _encode_tagged(payload) for leaf_id, payload, _ in entries}
        data = msgspec.msgpack.encode(blob)
        (directory / "data.msgpack").write_bytes(data)
        return {}

    @staticmethod
    def read_batch(directory: Path, kind_meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
        data = (directory / "data.msgpack").read_bytes()
        blob = msgspec.msgpack.decode(data)

        def reader(leaf_id: str) -> Any:
            return _decode_tagged(blob[leaf_id])

        return reader


# Order matters: :class:`MsgpackLeafSerializer` runs first so it can
# claim fully-msgpack-native subtrees as a single leaf.  The container
# serializers only see values whose tree contains at least one non-native
# element, and recurse so each child is dispatched independently.
stdlib_serializers: list[type[BaseSerializer]] = [
    MsgpackLeafSerializer,  # predicate-gated — claims fully-native subtrees
    DictSerializer,
    ListSerializer,
    TupleSerializer,
]


# ---------------------------------------------------------------------------
# Dispatch hints for :class:`TypeDispatchRegistry`
# ---------------------------------------------------------------------------

# Types whose dispatch is stable regardless of contents — safe for the
# by-type fast path.  Scalars go to ``MsgpackLeafSerializer`` because
# their value has no sub-objects that could contain e.g. a tensor.
# Polymorphic sibling types (``enum.Enum``, ``pathlib.PurePath``, ...)
# are handled by the MRO walk via their concrete subclasses — or fall
# through to the linear scan for types not listed here.
_stdlib_fast_path_types: dict[type, type[Serializer]] = {
    type(None): MsgpackLeafSerializer,
    bool: MsgpackLeafSerializer,
    int: MsgpackLeafSerializer,
    float: MsgpackLeafSerializer,
    str: MsgpackLeafSerializer,
    bytes: MsgpackLeafSerializer,
    bytearray: MsgpackLeafSerializer,
    complex: MsgpackLeafSerializer,
    datetime.datetime: MsgpackLeafSerializer,
    datetime.date: MsgpackLeafSerializer,
    datetime.time: MsgpackLeafSerializer,
    datetime.timedelta: MsgpackLeafSerializer,
    uuid.UUID: MsgpackLeafSerializer,
    decimal.Decimal: MsgpackLeafSerializer,
    fractions.Fraction: MsgpackLeafSerializer,
    range: MsgpackLeafSerializer,
    zoneinfo.ZoneInfo: MsgpackLeafSerializer,
    re.Pattern: MsgpackLeafSerializer,
    pathlib.PurePath: MsgpackLeafSerializer,
    pathlib.PurePosixPath: MsgpackLeafSerializer,
    pathlib.PureWindowsPath: MsgpackLeafSerializer,
    pathlib.PosixPath: MsgpackLeafSerializer,
    pathlib.WindowsPath: MsgpackLeafSerializer,
    ipaddress.IPv4Address: MsgpackLeafSerializer,
    ipaddress.IPv6Address: MsgpackLeafSerializer,
    ipaddress.IPv4Network: MsgpackLeafSerializer,
    ipaddress.IPv6Network: MsgpackLeafSerializer,
    ipaddress.IPv4Interface: MsgpackLeafSerializer,
    ipaddress.IPv6Interface: MsgpackLeafSerializer,
}

stdlib_serializers_by_type: dict[str, type[BaseSerializer]] = {
    qualified_type_name(t): ser for t, ser in _stdlib_fast_path_types.items()
}

# Container types — dispatch depends on contents (msgpack-native vs
# mixed), so we skip both the cache and the by-type fast path and
# re-evaluate the predicate every call.  ``slice`` and
# ``SimpleNamespace`` are scalar but the nativity predicate still
# recurses into their sub-objects, so treat them as volatile too.
stdlib_volatile_types: frozenset[type] = frozenset(
    {
        dict,
        OrderedDict,
        list,
        tuple,
        set,
        frozenset,
        deque,
        slice,
        types.SimpleNamespace,
    }
)
