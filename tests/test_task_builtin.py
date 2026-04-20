"""Tests for C builtin function support in Task."""

import pytest

from misen import Task


def test_task_accepts_c_builtin_function() -> None:
    task = Task(sum, [1, 2, 3])
    assert task.meta.id == "builtins.sum"
    assert task.result(compute_if_uncached=True) == 6


def test_task_accepts_c_builtin_from_module() -> None:
    import os

    task = Task(os.urandom, 4)
    assert task.meta.id == "posix.urandom"
    assert len(task.result(compute_if_uncached=True)) == 4


def test_task_rejects_non_callable() -> None:
    with pytest.raises(TypeError, match="Python function or C builtin"):
        Task(42)  # type: ignore[arg-type]


def test_task_rejects_builtin_without_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    from misen.utils import function_introspection

    def broken_signature(_func: object) -> object:
        msg = "no signature found"
        raise ValueError(msg)

    monkeypatch.setattr(function_introspection, "signature", broken_signature)
    with pytest.raises(TypeError, match="Could not introspect signature"):
        Task(sum, [1])
