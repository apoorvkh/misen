"""Core handler abstractions and canonical primitive encoding.

Handlers decompose Python objects into primitive trees (nested structures of
None, bool, int, float, str, bytes, tuple, list, set).  The ``digest``
function encodes those trees into deterministic bytes and hashes them with
xxh3-64.
"""

import struct
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeAlias, assert_never

from xxhash import xxh3_64_intdigest

from misen.utils.type_registry import qualified_type_name

__all__ = [
    "CollectionHandler",
    "ElementHasher",
    "Handler",
    "HandlerTypeRegistry",
    "PrimitiveHandler",
    "hash_values",
    "qualified_type_name",
]

# ---------------------------------------------------------------------------
# Canonical byte encoding
# ---------------------------------------------------------------------------

# Type tags — each value starts with one of these bytes.
_TAG_NONE = 0
_TAG_BOOL = 1
_TAG_INT = 2
_TAG_FLOAT = 3
_TAG_STR = 4
_TAG_BYTES = 5
_TAG_TUPLE = 6
_TAG_LIST = 7
_TAG_SET = 8

_STRUCT_DOUBLE = struct.Struct(">d")
_STRUCT_UINT64 = struct.Struct(">Q")


def _encode_length(n: int) -> bytes:
    return _STRUCT_UINT64.pack(n)


def _encode_int(n: int) -> bytes:
    if n == 0:
        return b"\x01\x00"  # length=1, value=0
    byte_length = (n.bit_length() + 8) // 8  # +8 for sign bit
    return _encode_length(byte_length) + n.to_bytes(byte_length, "big", signed=True)


def _encode_sequence(items: Any) -> bytes:
    parts = [_encode(item) for item in items]
    header = _encode_length(len(parts))
    return header + b"".join(_encode_length(len(p)) + p for p in parts)


def _encode(obj: Any) -> bytes:
    """Encode a limited set of Python types into canonical, deterministic bytes.

    Supported types: None, bool, int, float, str, bytes, tuple, list, set, frozenset.
    ``set`` and ``frozenset`` share ``_TAG_SET`` — safe because ``stable_hash``
    wraps every value in ``(version, type_name, obj_hash)`` at the top level,
    so the outer type name disambiguates them.
    """
    if obj is None:
        return bytes([_TAG_NONE])

    if isinstance(obj, bool):
        return bytes([_TAG_BOOL, obj])

    if isinstance(obj, int):
        return bytes([_TAG_INT]) + _encode_int(obj)

    if isinstance(obj, float):
        # Normalize negative zero.
        value = 0.0 if obj == 0.0 else obj
        return bytes([_TAG_FLOAT]) + _STRUCT_DOUBLE.pack(value)

    if isinstance(obj, str):
        encoded = obj.encode("utf-8")
        return bytes([_TAG_STR]) + _encode_length(len(encoded)) + encoded

    if isinstance(obj, bytes):
        return bytes([_TAG_BYTES]) + _encode_length(len(obj)) + obj

    if isinstance(obj, tuple):
        return bytes([_TAG_TUPLE]) + _encode_sequence(obj)

    if isinstance(obj, list):
        return bytes([_TAG_LIST]) + _encode_sequence(obj)

    if isinstance(obj, (set, frozenset)):
        # Sort by encoded bytes for determinism regardless of iteration order.
        parts = sorted(_encode(item) for item in obj)
        header = _encode_length(len(parts))
        return bytes([_TAG_SET]) + header + b"".join(_encode_length(len(p)) + p for p in parts)

    msg = (
        f"_encode does not support {type(obj).__name__!r}. "
        "Handler digests must decompose objects into primitives "
        "(None, bool, int, float, str, bytes) and containers (tuple, list, set)."
    )
    raise TypeError(msg)


def hash_values(obj: Any) -> int:
    """Encode a primitive tree with canonical bytes and hash with xxh3-64."""
    return xxh3_64_intdigest(_encode(obj), seed=0)


# ---------------------------------------------------------------------------
# Handler base classes
# ---------------------------------------------------------------------------

ElementHasher: TypeAlias = Callable[[Any], int]


class Handler(ABC):
    """Base handler protocol for canonical hashing.

    Subclasses implement three hooks:

    - :meth:`match` — return ``True`` if this handler can digest a value.
    - :meth:`digest` — return a stable integer digest for a value.  Takes
      ``element_hash`` as its second positional arg; collection handlers
      recurse through it, primitive handlers ignore it.
    - :meth:`type_name` (optional) — return the canonical type name recorded
      in the hash.  Override when the runtime ``type(obj).__module__`` may
      vary across platforms (see :class:`PathHandler`).

    ``digest`` is declared positional-only so primitive handlers can rename
    unused parameters (e.g. ``_element_hash``) without breaking LSP.
    """

    version: int = 1

    @staticmethod
    def type_name(obj: Any) -> str:
        """Return the canonical type name included in the hash.

        The default uses the runtime ``module.qualname``.  Override in
        handlers where the runtime location may change across Python
        versions or platforms (e.g. ``pathlib.PosixPath`` vs
        ``pathlib.WindowsPath``).
        """
        return qualified_type_name(type(obj))

    @staticmethod
    @abstractmethod
    def match(obj: Any) -> bool:
        """Return whether this handler can digest ``obj``."""

    @classmethod
    @abstractmethod
    def digest(cls, obj: Any, element_hash: ElementHasher, /) -> int:
        """Return a stable digest for ``obj``.

        ``element_hash`` is always supplied by ``stable_hash``; primitive
        handlers simply ignore it.
        """


HandlerTypeRegistry: TypeAlias = dict[str, type[Handler]]


class PrimitiveHandler(Handler):
    """Marker base for handlers that do not recurse into sub-objects.

    Primitive handlers never call ``element_hash`` from within :meth:`digest`
    and can be implemented with a single-argument form by ignoring the
    second parameter (``_element_hash``).  The marker exists purely for
    documentation and grep-ability; ``stable_hash`` treats primitive and
    collection handlers uniformly.
    """


class CollectionHandler(Handler):
    """Handler for structured types hashed by recursively hashing elements.

    Subclasses implement :meth:`elements` returning either a ``list`` (order
    preserved) or a ``set`` (order independent).  :meth:`digest` recurses
    through ``element_hash`` over each element and aggregates the results.
    """

    @staticmethod
    @abstractmethod
    def elements(obj: Any) -> list[Any] | set[Any]:
        """Return digest inputs as a list or set of elements."""

    @classmethod
    def digest(cls, obj: Any, element_hash: ElementHasher, /) -> int:
        """Hash a collection by hashing each element then hashing the aggregate."""
        match cls.elements(obj):
            case list() as elements:
                return hash_values([element_hash(i) for i in elements])
            case set() as elements:
                return hash_values({element_hash(i) for i in elements})
            case elements:
                assert_never(elements)
