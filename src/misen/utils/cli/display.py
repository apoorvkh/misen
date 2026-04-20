"""Shared helpers for presenting Tasks in CLI and TUI output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rich.markup import escape
from rich.text import Text

if TYPE_CHECKING:
    from misen.tasks import Task

__all__ = [
    "TaskArgs",
    "format_task_line_markup",
    "format_task_line_text",
    "iter_task_arg_children",
    "task_args",
]


@dataclass(frozen=True)
class TaskArgs:
    """Bound arguments of a Task separated into scalars and task-valued children."""

    scalar_items: tuple[tuple[str, str], ...]
    """``(name, repr(value))`` pairs for arguments with no nested Task references."""

    task_children: tuple[tuple[str, Any], ...]
    """``(label, child_task)`` pairs for each Task leaf reachable from the task's arguments."""


def task_args(task: Task[Any]) -> TaskArgs:
    """Split a task's bound arguments into inline scalars and task-valued children.

    Arguments declared in ``task.meta.exclude`` are skipped. Argument values that
    contain one or more Task leaves are emitted as task children (one per leaf);
    values with no Task references are emitted as scalar repr items.
    """
    from misen.tasks import Task as _Task
    from misen.utils.nested import iter_nested_leaves

    bound = task._signature.bind_partial(*task.args, **task.kwargs)  # noqa: SLF001
    scalars: list[tuple[str, str]] = []
    children: list[tuple[str, _Task[Any]]] = []

    for name, value in bound.arguments.items():
        if name in task.meta.exclude:
            continue
        if isinstance(value, _Task):
            children.append((name, value))
            continue
        if _Task._contains_task_reference(value):  # noqa: SLF001
            task_leaves = [leaf for leaf in iter_nested_leaves(value) if isinstance(leaf, _Task)]
            if len(task_leaves) == 1:
                children.append((name, task_leaves[0]))
            else:
                for index, leaf in enumerate(task_leaves):
                    children.append((f"{name}[{index}]", leaf))
            continue
        scalars.append((name, repr(value)))

    return TaskArgs(scalar_items=tuple(scalars), task_children=tuple(children))


def iter_task_arg_children(task: Task[Any]) -> tuple[tuple[str, Any], ...]:
    """Return ``(label, child_task)`` pairs for task-valued arguments."""
    return task_args(task).task_children


def format_task_line_markup(task: Task[Any], *, prefix: str | None = None) -> str:
    """Return rich-markup string ``name([dim]arg=[/dim]value, ...)`` for a task.

    Task-valued arguments are omitted from this single-line representation; they
    are expected to be rendered as tree children by the caller.
    """
    parts = task_args(task)
    line = f"[bright_white]{escape(task.func.__name__)}[/bright_white]"
    if parts.scalar_items:
        fragments = ", ".join(
            f"[dim]{escape(name)}=[/dim]{escape(value_repr)}" for name, value_repr in parts.scalar_items
        )
        line += f"({fragments})"
    if prefix is not None:
        line = f"[dim]{escape(prefix)} = [/dim]{line}"
    return line


def format_task_line_text(
    task: Task[Any],
    *,
    prefix: str | None = None,
    name_style: str = "bold white",
    dim_style: str = "dim",
) -> Text:
    """Return a :class:`rich.text.Text` representation of a task line for Textual rendering."""
    text = Text()
    if prefix is not None:
        text.append(f"{prefix} = ", style=dim_style)
    text.append(task.func.__name__, style=name_style)
    parts = task_args(task)
    if parts.scalar_items:
        text.append("(")
        for index, (name, value_repr) in enumerate(parts.scalar_items):
            if index > 0:
                text.append(", ")
            text.append(f"{name}=", style=dim_style)
            text.append(value_repr)
        text.append(")")
    return text
