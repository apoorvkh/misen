"""Tests for operator-overloading aliases on Task."""

from __future__ import annotations

import operator
from types import SimpleNamespace
from typing import Any

import pytest

from misen import Task, meta


@meta(id="test.add_xy")
def add_xy(x: int, y: int) -> int:
    return x + y


@meta(id="test.make_ns")
def make_ns(value: int) -> SimpleNamespace:
    return SimpleNamespace(value=value)


def _compute(task: Task[object]) -> object:
    return task.result(compute_if_uncached=True, compute_uncached_deps=True)


@pytest.mark.parametrize(
    ("symbol", "op"),
    [
        ("+", operator.add),
        ("-", operator.sub),
        ("*", operator.mul),
        ("/", operator.truediv),
        ("//", operator.floordiv),
        ("%", operator.mod),
        ("**", operator.pow),
        ("&", operator.and_),
        ("|", operator.or_),
        ("^", operator.xor),
        ("<<", operator.lshift),
        (">>", operator.rshift),
    ],
)
def test_binary_operators_build_tasks(symbol: str, op: object) -> None:
    a = Task(operator.add, 6, 0)
    b = Task(operator.add, 3, 0)
    built = eval(f"a {symbol} b", {"a": a, "b": b})  # noqa: S307
    assert isinstance(built, Task)
    assert built.func is op
    assert built.args == (a, b)
    assert _compute(built) == op(6, 3)


def test_binary_with_non_task_operand() -> None:
    a = Task(operator.add, 10, 0)
    result = a + 5
    assert result.func is operator.add
    assert result.args == (a, 5)
    assert _compute(result) == 15


def test_reflected_operator_puts_task_on_right() -> None:
    a = Task(operator.add, 10, 0)
    result = 5 - a
    assert result.func is operator.sub
    assert result.args == (5, a)
    assert _compute(result) == -5


def test_matmul_builds_task() -> None:
    a = Task(operator.add, 1, 0)
    b = Task(operator.add, 2, 0)
    built = a @ b
    assert built.func is operator.matmul
    assert built.args == (a, b)


@pytest.mark.parametrize(
    ("build", "expected_op", "value"),
    [
        (lambda t: -t, operator.neg, -7),
        (lambda t: +t, operator.pos, 7),
        (abs, operator.abs, 7),
        (lambda t: ~t, operator.invert, -8),
    ],
)
def test_unary_operators_build_tasks(
    build: object, expected_op: object, value: int
) -> None:
    a = Task(operator.add, 7, 0)
    built = build(a)  # ty:ignore[call-non-callable]
    assert built.func is expected_op
    assert built.args == (a,)
    assert _compute(built) == value


def test_getitem_builds_task_and_captures_dependency() -> None:
    tup = Task(operator.add, (10, 20, 30), ())  # concatenation yields (10,20,30)
    index_task = Task(operator.add, 1, 0)
    built = tup[index_task]
    assert built.func is operator.getitem
    assert built.args == (tup, index_task)
    assert index_task in built.dependencies
    assert _compute(built) == 20


def test_identity_and_hashing_preserved() -> None:
    a = Task(operator.add, 1, 2)
    b = Task(operator.add, 1, 2)
    assert a == b
    assert hash(a) == hash(b)
    assert {a, b} == {a}


def test_method_names_are_set() -> None:
    assert Task.__add__.__name__ == "__add__"
    assert Task.__radd__.__name__ == "__radd__"
    assert Task.__and__.__name__ == "__and__"
    assert Task.__rand__.__name__ == "__rand__"
    assert Task.__neg__.__name__ == "__neg__"
    assert Task.__getitem__.__name__ == "__getitem__"


def test_comparison_and_bool_are_not_overloaded() -> None:
    # __eq__ still returns a bool for DAG identity, not a Task.
    a = Task(operator.add, 1, 2)
    assert isinstance(a == a, bool)  # noqa: PLR0124
    # No <, >, <=, >= installed.
    assert not hasattr(Task, "__lt__") or Task.__lt__ is object.__lt__


def test_apply_wraps_function_over_result() -> None:
    a = Task(operator.add, 3, 4)
    built = a.apply(operator.neg)
    assert built.func is operator.neg
    assert built.args == (a,)
    assert _compute(built) == -7


def test_apply_forwards_extra_positional_args() -> None:
    a = Task(operator.add, 3, 4)
    built = a.apply(operator.add, 100)
    assert built.args == (a, 100)
    assert _compute(built) == 107


def test_apply_forwards_kwargs_into_task() -> None:
    a = Task(operator.add, 3, 4)
    kw_built = a.apply(add_xy, y=100)
    assert kw_built.args == (a,)
    assert dict(kw_built.kwargs) == {"y": 100}
    assert _compute(kw_built) == 107


def test_apply_enables_fluent_chaining() -> None:
    a = Task(operator.add, 1, 2)
    chained = a.apply(operator.neg).apply(operator.abs)
    assert _compute(chained) == 3


def test_attr_reads_attribute_of_result() -> None:
    from misen.utils.task_operators import _task_getattr

    ns = Task(make_ns, 42)
    attr_task: Task[Any] = ns.attr("value")
    assert attr_task.func is _task_getattr
    assert attr_task.args == (ns, "value")
    assert _compute(attr_task) == 42
