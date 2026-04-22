"""Operator-overloading and fluent-chaining mixin for :class:`misen.tasks.Task`.

Each method builds ``Task(func, ...)`` from ``self``. The :func:`_build` factory
does a lazy import of :class:`Task` to avoid a circular dependency with
:mod:`misen.tasks`; Python caches the import after the first call.

Comparison operators (``==``, ``<``, ...) and ``__bool__`` are intentionally
omitted — ``__eq__`` and ``__hash__`` on :class:`Task` are load-bearing for
task identity and set/dict membership, so overloading them would break DAG
semantics.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any

from misen.task_metadata import meta
from misen.utils.function_introspection import external_callable_id

if TYPE_CHECKING:
    from collections.abc import Callable

    from misen.tasks import Task


class TaskOperatorsMixin:
    """Arithmetic, bitwise, subscript, and fluent-chaining methods that build Tasks."""

    @staticmethod
    def _task_type() -> type[Task]:
        from misen.tasks import Task

        return Task

    def apply(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Task[Any]:
        """Build ``Task(func, self, *args, **kwargs)`` — apply ``func`` to this task's result.

        Args:
            func: Callable applied to this task's result, followed by ``args`` and ``kwargs``.
            *args: Extra positional arguments forwarded to ``func``.
            **kwargs: Extra keyword arguments forwarded to ``func``.

        Returns:
            A new :class:`Task` wrapping the application.
        """
        return self._task_type()(func, self, *args, **kwargs)

    def attr(self, name: str) -> Task[Any]:
        """Build a Task that reads ``name`` off this task's result.

        Args:
            name: Attribute name to look up on the result.

        Returns:
            A new :class:`Task` wrapping the attribute access.
        """
        return self._task_type()(_task_getattr, self, name)

    def __add__(self, other: Any) -> Task[Any]:  # noqa: D105
        return self._task_type()(operator.add, self, other)

    def __radd__(self, other: Any) -> Task[Any]:  # noqa: D105
        return self._task_type()(operator.add, other, self)

    def __sub__(self, other: Any) -> Task[Any]:  # noqa: D105
        return self._task_type()(operator.sub, self, other)

    def __rsub__(self, other: Any) -> Task[Any]:  # noqa: D105
        return self._task_type()(operator.sub, other, self)

    def __mul__(self, other: Any) -> Task[Any]:  # noqa: D105
        return self._task_type()(operator.mul, self, other)

    def __rmul__(self, other: Any) -> Task[Any]:  # noqa: D105
        return self._task_type()(operator.mul, other, self)

    def __truediv__(self, other: Any) -> Task[Any]:  # noqa: D105
        return self._task_type()(operator.truediv, self, other)

    def __rtruediv__(self, other: Any) -> Task[Any]:  # noqa: D105
        return self._task_type()(operator.truediv, other, self)

    def __floordiv__(self, other: Any) -> Task[Any]:  # noqa: D105
        return self._task_type()(operator.floordiv, self, other)

    def __rfloordiv__(self, other: Any) -> Task[Any]:  # noqa: D105
        return self._task_type()(operator.floordiv, other, self)

    def __mod__(self, other: Any) -> Task[Any]:  # noqa: D105
        return self._task_type()(operator.mod, self, other)

    def __rmod__(self, other: Any) -> Task[Any]:  # noqa: D105
        return self._task_type()(operator.mod, other, self)

    def __pow__(self, other: Any) -> Task[Any]:  # noqa: D105
        return self._task_type()(operator.pow, self, other)

    def __rpow__(self, other: Any) -> Task[Any]:  # noqa: D105
        return self._task_type()(operator.pow, other, self)

    def __matmul__(self, other: Any) -> Task[Any]:  # noqa: D105
        return self._task_type()(operator.matmul, self, other)

    def __rmatmul__(self, other: Any) -> Task[Any]:  # noqa: D105
        return self._task_type()(operator.matmul, other, self)

    def __and__(self, other: Any) -> Task[Any]:  # noqa: D105
        return self._task_type()(operator.and_, self, other)

    def __rand__(self, other: Any) -> Task[Any]:  # noqa: D105
        return self._task_type()(operator.and_, other, self)

    def __or__(self, other: Any) -> Task[Any]:  # noqa: D105
        return self._task_type()(operator.or_, self, other)

    def __ror__(self, other: Any) -> Task[Any]:  # noqa: D105
        return self._task_type()(operator.or_, other, self)

    def __xor__(self, other: Any) -> Task[Any]:  # noqa: D105
        return self._task_type()(operator.xor, self, other)

    def __rxor__(self, other: Any) -> Task[Any]:  # noqa: D105
        return self._task_type()(operator.xor, other, self)

    def __lshift__(self, other: Any) -> Task[Any]:  # noqa: D105
        return self._task_type()(operator.lshift, self, other)

    def __rlshift__(self, other: Any) -> Task[Any]:  # noqa: D105
        return self._task_type()(operator.lshift, other, self)

    def __rshift__(self, other: Any) -> Task[Any]:  # noqa: D105
        return self._task_type()(operator.rshift, self, other)

    def __rrshift__(self, other: Any) -> Task[Any]:  # noqa: D105
        return self._task_type()(operator.rshift, other, self)

    def __neg__(self) -> Task[Any]:  # noqa: D105
        return self._task_type()(operator.neg, self)

    def __pos__(self) -> Task[Any]:  # noqa: D105
        return self._task_type()(operator.pos, self)

    def __abs__(self) -> Task[Any]:  # noqa: D105
        return self._task_type()(operator.abs, self)

    def __invert__(self) -> Task[Any]:  # noqa: D105
        return self._task_type()(operator.invert, self)

    def __getitem__(self, key: Any) -> Task[Any]:  # noqa: D105
        return self._task_type()(operator.getitem, self, key)


@meta(id=external_callable_id(getattr))
def _task_getattr(obj: Any, name: str) -> Any:
    """Wrapper for :func:`getattr` that exposes an introspectable signature to :class:`Task`."""
    return getattr(obj, name)
