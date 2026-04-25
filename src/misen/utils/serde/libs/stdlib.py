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

import array
import datetime
import decimal
import enum
import fractions
import ipaddress
import pathlib
import re
import types
import uuid
import zoneinfo
from collections import ChainMap, Counter, OrderedDict, defaultdict, deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import msgspec.msgpack

from misen.exceptions import SerializationError
from misen.utils.serde.base import BaseSerializer, Container, DecodeCtx, EncodeCtx, LeafSerializer, Node, Serializer
from misen.utils.type_registry import import_by_qualified_name, qualified_type_name

__all__ = [
    "ChainMapSerializer",
    "CounterSerializer",
    "DefaultDictSerializer",
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
# rest (sets, frozensets, tuples, datetimes, UUIDs, namedtuples,
# dict-subclasses, ...) we wrap values in a ``{_TAG: ...}`` envelope on
# encode and unwrap on decode so they round-trip cleanly.  Dataclasses
# are handled by :class:`DataclassSerializer` (recursive field walk),
# not here.
# :class:`MsgpackLeafSerializer` uses these; :func:`_is_msgpack_native`
# below mirrors the type tree without actually encoding.

_TAG = "__t"
_VAL = "v"


def _encode_callable_name(fn: Any) -> str | None:
    """Return a round-trippable ``module.qualname`` for *fn*, or ``None``.

    Used for ``defaultdict.default_factory`` — classes (``list``, ``dict``)
    and module-level functions round-trip; lambdas and locally-defined
    functions have ``<`` in their qualname and can't be re-imported, so
    we drop the factory rather than produce a broken reference.
    """
    if fn is None:
        return None
    qname = getattr(fn, "__qualname__", None)
    module = getattr(fn, "__module__", None)
    if not qname or not module or "<" in qname:
        return None
    return f"{module}.{qname}"


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

    if type(obj) is array.array:
        return {_TAG: "array", _VAL: obj.tobytes(), "typecode": obj.typecode}

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
    # Counter / defaultdict / ChainMap come before plain dict — they're dict
    # subclasses (and ChainMap presents dict-like items()) so the strict dict
    # branch below would silently downcast and lose subclass identity.
    if type(obj) is Counter:
        return {_TAG: "Counter", _VAL: [[_encode_tagged(k), _encode_tagged(v)] for k, v in obj.items()]}
    if type(obj) is defaultdict:
        return {
            _TAG: "defaultdict",
            _VAL: [[_encode_tagged(k), _encode_tagged(v)] for k, v in obj.items()],
            "factory": _encode_callable_name(obj.default_factory),
        }
    if type(obj) is ChainMap:
        return {_TAG: "ChainMap", _VAL: [_encode_tagged(m) for m in obj.maps]}
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
                return import_by_qualified_name(obj["cls"])(_decode_tagged(val))
            if tag == "path":
                return import_by_qualified_name(obj["cls"])(val)
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
                return import_by_qualified_name(obj["cls"])(**{k: _decode_tagged(v) for k, v in val.items()})
            if tag == "OrderedDict":
                return OrderedDict((k, _decode_tagged(v)) for k, v in val)
            if tag == "deque":
                return deque((_decode_tagged(item) for item in val), maxlen=obj.get("maxlen"))
            if tag == "Counter":
                return Counter({_decode_tagged(k): _decode_tagged(v) for k, v in val})
            if tag == "defaultdict":
                factory_name = obj.get("factory")
                factory: Any = None
                if factory_name:
                    try:
                        factory = import_by_qualified_name(factory_name)
                    except ImportError:
                        factory = None
                dd: Any = defaultdict(factory)
                for k, v in val:
                    dd[_decode_tagged(k)] = _decode_tagged(v)
                return dd
            if tag == "ChainMap":
                return ChainMap(*(_decode_tagged(m) for m in val))
            if tag == "array":
                out = array.array(obj["typecode"])
                out.frombytes(val)
                return out
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
    # ``array.array`` holds only primitive numbers under the hood — we
    # encode it as tagged bytes + typecode, no sub-object walk.
    array.array,
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
    t = type(key)
    if t in _MSGPACK_NATIVE_KEY_TYPES:
        return True
    # Tuples-as-keys are supported by msgspec as long as each element
    # is itself a native key type (recursively).
    if t is tuple:
        return all(_is_msgpack_native_key(e) for e in key)
    return False


# Container-ish types tracked in ``_graph_seen`` so a shared instance
# appearing twice in the walk bails out of msgpack.  Immutable types
# (tuple, frozenset) are included on purpose: inlining them into the
# msgpack blob would duplicate their contents, whereas routing the
# enclosing object through the ``Container`` path lets leaf-id memo
# preserve shared identity on decode.
_MSGPACK_GRAPH_TYPES: tuple[type, ...] = (
    dict,
    OrderedDict,
    list,
    tuple,
    set,
    frozenset,
    deque,
    types.SimpleNamespace,
    Counter,
    defaultdict,
    ChainMap,
)


def _is_msgpack_native(
    obj: Any,
    _path: set[int] | None = None,
    _graph_seen: set[int] | None = None,
) -> bool:
    """Return ``True`` iff ``_encode_tagged`` + ``msgspec.msgpack`` can encode ``obj``.

    Walks the object's transitive structure.  Used by
    :class:`MsgpackLeafSerializer.match` to decide whether the value
    (and everything inside it) belongs in the msgpack batched blob.
    Shared recursive containers return ``False`` so graph identity can
    be represented by the higher-level ``Container``/``Ref`` manifest.
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
    if _path is None:
        _path = set()
    if _graph_seen is None:
        _graph_seen = set()
    oid = id(obj)
    if oid in _path:
        return False
    if isinstance(obj, _MSGPACK_GRAPH_TYPES):
        if oid in _graph_seen:
            return False
        _graph_seen.add(oid)
    _path.add(oid)
    try:
        if t is slice:
            return all(_is_msgpack_native(x, _path, _graph_seen) for x in (obj.start, obj.stop, obj.step))

        if t is types.SimpleNamespace:
            return all(_is_msgpack_native(v, _path, _graph_seen) for v in vars(obj).values())

        # NamedTuple — must come before the plain ``tuple`` check.
        if isinstance(obj, tuple) and hasattr(t, "_fields"):
            return all(_is_msgpack_native(getattr(obj, f), _path, _graph_seen) for f in t._fields)

        if t is dict or t is OrderedDict:
            return all(
                _is_msgpack_native_key(k) and _is_msgpack_native(v, _path, _graph_seen) for k, v in obj.items()
            )

        if t is Counter or t is defaultdict:
            # Counter/defaultdict encode keys through ``_encode_tagged`` (not
            # as msgspec dict keys), so any natively-encodable object is a
            # valid key.  For defaultdict, a non-importable factory (lambda
            # or local function) would silently drop the factory on decode,
            # so we force such instances through the Container path where
            # the encode error surfaces explicitly.
            if t is defaultdict:
                factory = obj.default_factory
                if factory is not None and _encode_callable_name(factory) is None:
                    return False
            return all(
                _is_msgpack_native(k, _path, _graph_seen) and _is_msgpack_native(v, _path, _graph_seen)
                for k, v in obj.items()
            )

        if t is ChainMap:
            return all(_is_msgpack_native(m, _path, _graph_seen) for m in obj.maps)

        if t in (list, tuple, set, frozenset, deque):
            return all(_is_msgpack_native(item, _path, _graph_seen) for item in obj)

        return False
    finally:
        _path.discard(oid)


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
        if not isinstance(node, Container):
            msg = f"{qualified_type_name(cls)} expected a Container node, got {type(node).__name__}."
            raise SerializationError(msg)
        out: Any = OrderedDict() if node.meta.get("type") == "OrderedDict" else {}
        ctx.remember_node(node, out)
        items = [(k, ctx.decode(v)) for k, v in node.children.items()]
        if node.meta.get("type") == "OrderedDict":
            out.update(items)
            return out
        out.update(items)
        return out


class CounterSerializer(BaseSerializer[Any]):
    """Recursive serializer for ``collections.Counter`` with ``str`` keys.

    Non-str-keyed Counters route through :class:`MsgpackLeafSerializer`
    via the tagged encoder.  Each value dispatches independently so a
    Counter holding e.g. an ndarray is handled correctly.
    """

    @staticmethod
    def match(obj: Any) -> bool:
        if type(obj) is not Counter:
            return False
        return all(isinstance(k, str) for k in obj)

    @classmethod
    def encode(cls, obj: Any, ctx: EncodeCtx) -> Node:
        return Container(
            serializer=qualified_type_name(cls),
            children={k: ctx.encode(v) for k, v in obj.items()},
            meta={},
        )

    @classmethod
    def decode(cls, node: Node, ctx: DecodeCtx) -> Any:
        if not isinstance(node, Container):
            msg = f"{qualified_type_name(cls)} expected a Container node, got {type(node).__name__}."
            raise SerializationError(msg)
        out: Counter[Any] = Counter()
        ctx.remember_node(node, out)
        for k, v in node.children.items():
            out[k] = ctx.decode(v)
        return out


class DefaultDictSerializer(BaseSerializer[Any]):
    """Recursive serializer for ``collections.defaultdict`` with ``str`` keys.

    The ``default_factory`` must be importable by qualified name (a
    class or module-level function); lambdas and local factories would
    silently turn into ``None`` on decode, so they fall through here and
    surface as an encode error instead.
    """

    @staticmethod
    def match(obj: Any) -> bool:
        if type(obj) is not defaultdict:
            return False
        if not all(isinstance(k, str) for k in obj):
            return False
        factory = obj.default_factory
        return factory is None or _encode_callable_name(factory) is not None

    @classmethod
    def encode(cls, obj: Any, ctx: EncodeCtx) -> Node:
        factory_name = _encode_callable_name(obj.default_factory)
        meta: dict[str, Any] = {}
        if factory_name is not None:
            meta["factory"] = factory_name
        return Container(
            serializer=qualified_type_name(cls),
            children={k: ctx.encode(v) for k, v in obj.items()},
            meta=meta,
        )

    @classmethod
    def decode(cls, node: Node, ctx: DecodeCtx) -> Any:
        if not isinstance(node, Container):
            msg = f"{qualified_type_name(cls)} expected a Container node, got {type(node).__name__}."
            raise SerializationError(msg)
        factory: Any = None
        factory_name = node.meta.get("factory")
        if factory_name:
            try:
                factory = import_by_qualified_name(factory_name)
            except ImportError:
                factory = None
        out: defaultdict[Any, Any] = defaultdict(factory)
        ctx.remember_node(node, out)
        for k, v in node.children.items():
            out[k] = ctx.decode(v)
        return out


class ChainMapSerializer(BaseSerializer[Any]):
    """Recursive serializer for ``collections.ChainMap``.

    Each map in :attr:`ChainMap.maps` dispatches independently through
    :func:`ctx.encode` — so a ChainMap whose maps contain tensors or
    ndarrays still round-trips.  Requires each map to be a plain
    ``dict`` with ``str`` keys (consistent with :class:`DictSerializer`).
    """

    @staticmethod
    def match(obj: Any) -> bool:
        if type(obj) is not ChainMap:
            return False
        for m in obj.maps:
            if type(m) is not dict or not all(isinstance(k, str) for k in m):
                return False
        return True

    @classmethod
    def encode(cls, obj: Any, ctx: EncodeCtx) -> Node:
        return Container(
            serializer=qualified_type_name(cls),
            children=[ctx.encode(m) for m in obj.maps],
            meta={},
        )

    @classmethod
    def decode(cls, node: Node, ctx: DecodeCtx) -> Any:
        if not isinstance(node, Container):
            msg = f"{qualified_type_name(cls)} expected a Container node, got {type(node).__name__}."
            raise SerializationError(msg)
        # ChainMap's own object identity is published before decoding children
        # via ``remember_node``.  Each child is a dict, so the dict serializer
        # handles its own recursive decoding.
        maps = [ctx.decode(c) for c in node.children]
        out = ChainMap(*maps)
        ctx.remember_node(node, out)
        return out


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
        if not isinstance(node, Container):
            msg = f"{qualified_type_name(cls)} expected a Container node, got {type(node).__name__}."
            raise SerializationError(msg)
        out: list[Any] = []
        ctx.remember_node(node, out)
        out.extend(ctx.decode(c) for c in node.children)
        return out


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
        if not isinstance(node, Container):
            msg = f"{qualified_type_name(cls)} expected a Container node, got {type(node).__name__}."
            raise SerializationError(msg)
        out = tuple(ctx.decode(c) for c in node.children)
        ctx.remember_node(node, out)
        return out


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
        # Wrap msgspec's raw ``TypeError``/``OverflowError`` in the
        # package-level :class:`SerializationError` so callers can catch
        # one thing.  Common triggers: integers outside msgspec's
        # ``[-2**63, 2**64-1]`` range, builtin subclasses the predicate
        # let through via ``isinstance``, unpicklable objects.
        try:
            blob = {leaf_id: _encode_tagged(payload) for leaf_id, payload, _ in entries}
            data = msgspec.msgpack.encode(blob)
        except (TypeError, OverflowError) as exc:
            msg = f"Cannot encode value for msgpack batch: {exc}"
            raise SerializationError(msg) from exc
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
    CounterSerializer,
    DefaultDictSerializer,
    ChainMapSerializer,
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
    array.array: MsgpackLeafSerializer,
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
        Counter,
        defaultdict,
        ChainMap,
    }
)
