"""Core handler abstractions and shared handler type aliases."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeAlias

from typing_extensions import assert_never

from misen_hash.hash import canonical_hash

__all__ = [
    "CollectionHandler",
    "ElementHasher",
    "Handler",
    "HandlerClass",
    "HandlerTypeList",
    "HandlerTypeRegistry",
    "PrimitiveHandler",
    "qualified_type_name",
]

ElementHasher: TypeAlias = Callable[[Any], int]


class Handler(ABC):
    """Base handler protocol for canonical hashing."""

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
        ...

    @staticmethod
    @abstractmethod
    def digest(obj: Any, element_hash: ElementHasher | None) -> int:
        """Return a stable digest for ``obj``."""
        ...


HandlerClass: TypeAlias = type[Handler]
HandlerTypeList: TypeAlias = list[HandlerClass]
HandlerTypeRegistry: TypeAlias = dict[str, HandlerClass]


def qualified_type_name(obj_type: type[Any]) -> str:
    """Return the fully-qualified ``module.qualname`` for ``obj_type``."""
    return f"{obj_type.__module__}.{obj_type.__qualname__}"


class PrimitiveHandler(Handler):
    """Handler for primitive/opaque types hashed as a single unit."""

    @staticmethod
    @abstractmethod
    def digest(obj: Any) -> int:  # ty:ignore[invalid-method-override]
        """Return a digest for ``obj`` without recursive hashing."""
        ...


class CollectionHandler(Handler):
    """Handler for structured types hashed by recursively hashing elements."""

    @staticmethod
    @abstractmethod
    def elements(obj: Any) -> list[Any] | set[Any]:
        """Return digest inputs as a list or set of elements."""
        ...

    @classmethod
    def digest(cls, obj: Any, element_hash: ElementHasher) -> int:  # ty:ignore[invalid-method-override]
        """Hash a collection by hashing each element then hashing the aggregate."""
        match cls.elements(obj):
            case list() as elements:
                return canonical_hash([element_hash(i) for i in elements])
            case set() as elements:
                return canonical_hash({element_hash(i) for i in elements})
            case elements:
                assert_never(elements)
