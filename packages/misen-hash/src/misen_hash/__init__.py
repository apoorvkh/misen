"""Canonical object hashing with extensible type handlers.

Design notes:

- Hashes are type-aware: runtime type name is included in the top-level digest.
- Built-in handlers cover common primitives/collections deterministically.
- Torch-specific handlers are matched dynamically when torch is installed.
- Unknown types fall back to dill serialization.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, cast

from misen_hash.utils import hash_msgspec

__all__ = ["CollectionHandler", "PrimitiveHandler", "canonical_hash"]


def canonical_hash(obj: Any) -> int:
    """Return a stable 64-bit hash for an arbitrary Python object."""
    obj_type = type(obj).__qualname__
    obj_hash = _lookup_handler(obj).digest(obj, element_hash=canonical_hash)
    return hash_msgspec((obj_type, obj_hash))


class Handler(ABC):
    """Base handler protocol for canonical hashing."""

    @staticmethod
    @abstractmethod
    def match(obj: Any) -> bool: ...

    @staticmethod
    @abstractmethod
    def digest(obj: Any, element_hash: Callable[[Any], int] | None) -> int: ...


class PrimitiveHandler(Handler):
    """Handler for primitive/opaque types hashed as a single unit."""

    @staticmethod
    @abstractmethod
    def digest(obj: Any, element_hash: None = None) -> int: ...


class CollectionHandler(Handler):
    """Handler for structured types hashed by recursively hashing elements."""

    @staticmethod
    @abstractmethod
    def elements(obj: Any) -> list[Any] | set[Any]: ...

    @classmethod
    def digest(cls, obj: Any, element_hash: Callable[[Any], int]) -> int:
        """Hash a collection by hashing each element then hashing the aggregate."""
        elements = cls.elements(obj)
        if isinstance(elements, list):
            return hash_msgspec([element_hash(i) for i in elements])
        if isinstance(elements, set):
            return hash_msgspec({element_hash(i) for i in elements})
        msg = f"Unsupported collection type: {type(elements)}"
        raise ValueError(msg)


from misen_hash._builtins import builtin_handlers, builtin_handlers_by_type  # noqa: E402
from misen_hash._dill import DillHandler  # noqa: E402
from misen_hash._torch import TorchModuleHandler, TorchTensorHandler  # noqa: E402

_handlers_type_cache: dict[type[Any], Handler] = {**builtin_handlers_by_type}


def _lookup_handler(obj: Any) -> Handler:
    """Resolve the most specific handler class for ``obj`` and memoize by type."""
    obj_type = type(obj)

    if obj_type not in _handlers_type_cache:
        for hash_cls in cast("list[Handler]", [*builtin_handlers, TorchTensorHandler, TorchModuleHandler]):
            if hash_cls.match(obj):
                _handlers_type_cache[obj_type] = hash_cls
                break
        else:
            _handlers_type_cache[obj_type] = DillHandler

    return _handlers_type_cache[obj_type]
