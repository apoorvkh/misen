"""Generic type-dispatch registry shared by handler and serializer lookups.

Both :mod:`misen.utils.hashing` and :mod:`misen.utils.serde` need to resolve a
registered class for an arbitrary Python object using the same algorithm:

1. Memoized cache keyed by the object's runtime ``type``.
2. Walk ``type(obj).__mro__`` and look up each base by ``qualified_type_name``
   in a fast ``dict`` registry.
3. Fall back to a linear scan over registered candidates calling a predicate
   (typically ``cls.match(obj)``) for duck-typed matches.

This module factors that pattern into a single generic class.
"""

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Generic, TypeVar

__all__ = ["TypeDispatchRegistry", "qualified_type_name"]

T = TypeVar("T")


def qualified_type_name(obj_type: type[Any]) -> str:
    """Return the fully-qualified ``module.qualname`` for ``obj_type``."""
    return f"{obj_type.__module__}.{obj_type.__qualname__}"


class TypeDispatchRegistry(Generic[T]):
    """Resolve a registered value for a Python object via type-based dispatch.

    Parameters
    ----------
    by_type_name:
        Mapping from ``qualified_type_name`` to a registered value.  Used for
        the fast MRO-walk lookup path.
    candidates:
        Iterable of registered values scanned linearly when no MRO base matches
        ``by_type_name``.  Each value is tested with ``predicate(value, obj)``.
    predicate:
        Callable used during the linear-scan fallback to decide whether a
        candidate value matches ``obj``.

    Notes:
    -----
    Lookup results are memoized by the runtime ``type(obj)``, so subsequent
    calls with the same type are O(1).
    """

    __slots__ = ("_by_type_name", "_cache", "_candidates", "_predicate")

    def __init__(
        self,
        *,
        by_type_name: Mapping[str, T],
        candidates: Iterable[T],
        predicate: Callable[[T, Any], bool],
    ) -> None:
        """Build a registry from a name-keyed mapping and a candidate list."""
        self._by_type_name: dict[str, T] = dict(by_type_name)
        self._candidates: list[T] = list(candidates)
        self._predicate = predicate
        self._cache: dict[type[Any], T] = {}

    @property
    def by_type_name(self) -> Mapping[str, T]:
        """Read-only view of the type-name registry."""
        return self._by_type_name

    @property
    def candidates(self) -> list[T]:
        """Registered candidates considered during the linear-scan fallback."""
        return self._candidates

    def lookup(self, obj: Any) -> T | None:
        """Return the registered value for ``obj``, or ``None`` if not found.

        The caller is responsible for raising a domain-specific error when
        ``None`` is returned, so each call site can produce its own message.
        """
        obj_type = type(obj)

        cached = self._cache.get(obj_type)
        if cached is not None:
            return cached

        for base_type in obj_type.__mro__:
            value = self._by_type_name.get(qualified_type_name(base_type))
            if value is not None:
                self._cache[obj_type] = value
                return value

        for value in self._candidates:
            if self._predicate(value, obj):
                self._cache[obj_type] = value
                return value

        return None
