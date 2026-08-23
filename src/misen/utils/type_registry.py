"""Generic type dispatch shared by hashing and serialization registries."""

import importlib
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Generic, TypeVar

__all__ = ["TypeDispatchRegistry", "import_by_qualified_name", "qualified_type_name"]

T = TypeVar("T")


def qualified_type_name(obj_type: type[Any]) -> str:
    """Return the fully-qualified ``module.qualname`` for ``obj_type``."""
    return f"{obj_type.__module__}.{obj_type.__qualname__}"


def import_by_qualified_name(qualified: str) -> Any:
    """Inverse of :func:`qualified_type_name` — resolve a ``module.qualname`` string.

    Handles nested classes (``__qualname__`` containing dots, e.g.
    ``Outer.Inner``) by scanning for the longest module prefix that
    actually imports, then walking the remainder via :func:`getattr`.
    A naive ``rpartition('.')`` split misplaces the boundary for nested
    classes because the last dot may be inside the qualname, not between
    module and qualname.
    """
    parts = qualified.split(".")
    last_error: Exception | None = None
    # Prefer the longest prefix — a real submodule should win over a
    # same-named class attribute on a parent module.
    for i in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:i])
        try:
            obj: Any = importlib.import_module(module_name)
        except ImportError as exc:
            last_error = exc
            continue
        try:
            for attr in parts[i:]:
                obj = getattr(obj, attr)
        except AttributeError as exc:
            last_error = exc
            continue
        return obj
    msg = f"Cannot resolve qualified name {qualified!r}"
    raise ImportError(msg) from last_error


class TypeDispatchRegistry(Generic[T]):
    """Resolve and memoize values by MRO, then by a fallback predicate.

    ``volatile_types`` bypass both type-based paths because their dispatch
    depends on each object's contents.
    """

    __slots__ = ("_by_type_name", "_cache", "_candidates", "_predicate", "_volatile_types")

    def __init__(
        self,
        *,
        by_type_name: Mapping[str, T],
        candidates: Iterable[T],
        predicate: Callable[[T, Any], bool],
        volatile_types: Iterable[type[Any]] | None = None,
    ) -> None:
        """Build a registry from a name-keyed mapping and a candidate list."""
        self._by_type_name: dict[str, T] = dict(by_type_name)
        self._candidates: list[T] = list(candidates)
        self._predicate = predicate
        self._cache: dict[type[Any], T] = {}
        self._volatile_types: frozenset[type[Any]] = frozenset(volatile_types or ())

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

        volatile = obj_type in self._volatile_types
        if not volatile:
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
                if not volatile:
                    self._cache[obj_type] = value
                return value

        return None
