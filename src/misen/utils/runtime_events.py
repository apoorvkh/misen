"""Runtime event printing for interactive observability."""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any, Literal

from rich.console import Console
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from misen.tasks import Task
    from misen.utils.work_unit import WorkUnit

__all__ = [
    "RuntimeJobSummary",
    "runtime_activity",
    "runtime_event",
    "runtime_job_done",
    "runtime_job_failed",
    "runtime_job_pending",
    "runtime_job_running",
    "runtime_job_summary_lines",
    "runtime_progress",
    "task_label",
    "work_unit_label",
]

_FALSEY = frozenset({"0", "false", "no", "off"})
_LIVE_CONTEXT: dict[str, int] = {"depth": 0}
_LIVE_CONTEXT_LOCK = threading.Lock()
_JOB_BOARD_ENV = "MISEN_RUNTIME_JOB_BOARD"

_JobState = Literal["pending", "running", "done", "failed"]
RuntimeJobState = Literal["pending", "running", "done", "failed", "unknown"]


@dataclass
class _JobStatusLine:
    label: str
    state: _JobState = "pending"
    job_id: str | None = None
    pid: int | None = None


@dataclass(frozen=True)
class RuntimeJobSummary:
    """Final one-line summary row for a job."""

    label: str
    state: RuntimeJobState
    job_id: str | None = None
    pid: int | None = None


class _RuntimeJobBoard:
    """Live-updating local job status board."""

    __slots__ = ("_entries", "_live", "_lock")

    def __init__(self) -> None:
        self._entries: dict[int, _JobStatusLine] = {}
        self._live: Live | None = None
        self._lock = threading.RLock()

    def pending(self, job_key: int, label: str) -> None:
        """Register a pending job row."""
        with self._lock:
            if self._live is None and self._entries and self._all_terminal_locked():
                self._entries.clear()

            line = self._entries.get(job_key)
            if line is None:
                self._entries[job_key] = _JobStatusLine(label=label, state="pending")
            else:
                line.label = label
                line.state = "pending"
            self._refresh_locked()

    def running(self, job_key: int, *, job_id: str | None, pid: int | None) -> None:
        """Mark a job row as running."""
        with self._lock:
            line = self._entries.setdefault(job_key, _JobStatusLine(label=f"job-{job_key}", state="pending"))
            line.state = "running"
            if job_id is not None:
                line.job_id = job_id
            if pid is not None:
                line.pid = pid
            self._refresh_locked()

    def done(self, job_key: int) -> None:
        """Mark a job row as complete."""
        with self._lock:
            line = self._entries.setdefault(job_key, _JobStatusLine(label=f"job-{job_key}", state="pending"))
            line.state = "done"
            self._refresh_locked()

    def failed(self, job_key: int) -> None:
        """Mark a job row as failed."""
        with self._lock:
            line = self._entries.setdefault(job_key, _JobStatusLine(label=f"job-{job_key}", state="pending"))
            line.state = "failed"
            self._refresh_locked()

    def on_live_context_exit(self) -> None:
        """Refresh board after other Rich live widgets finish."""
        with self._lock:
            self._refresh_locked()

    def _refresh_locked(self) -> None:
        if not self._entries or _live_context_active():
            return

        console = _get_console()
        if console is None:
            return

        renderable = self._render_locked()
        has_active = not self._all_terminal_locked()

        if self._live is None:
            if has_active:
                self._live = Live(renderable, console=console, refresh_per_second=12, transient=False)
                self._live.start()
                return

            # If all jobs are already terminal, render a static final snapshot once.
            console.print(renderable)
            self._entries.clear()
            return

        self._live.update(renderable, refresh=True)
        if self._all_terminal_locked():
            self._live.stop()
            self._live = None
            self._entries.clear()

    def _all_terminal_locked(self) -> bool:
        return all(line.state in {"done", "failed"} for line in self._entries.values())

    def _render_locked(self) -> Table:
        table = Table.grid(padding=(0, 1))
        table.add_column(width=8, no_wrap=True)
        table.add_column()

        for line in self._entries.values():
            match line.state:
                case "pending":
                    indicator = Text("")
                case "running":
                    indicator = Spinner("dots", style="yellow")
                case "done":
                    indicator = Text("complete", style="green")
                case "failed":
                    indicator = Text("failed", style="bold red")

            detail_suffix = _job_detail_suffix(job_id=line.job_id, pid=line.pid)
            table.add_row(indicator, Text(f"{line.label}{detail_suffix}"))

        return table


_JOB_BOARD = _RuntimeJobBoard()


def runtime_event(message: str, *, style: str = "cyan") -> None:
    """Print one runtime event line.

    Output is enabled by default and can be disabled with
    ``MISEN_RUNTIME_EVENTS=0``.
    """
    if not _events_enabled():
        return

    console = _get_console()
    if console is not None:
        console.print(f"[bold blue][misen][/bold blue] {message}", style=style, highlight=False)
        return

    sys.stderr.write(f"[misen] {message}\n")
    sys.stderr.flush()


@contextlib.contextmanager
def runtime_activity(message: str, *, spinner: str = "dots", style: str = "cyan") -> Iterator[None]:
    """Show a spinner-backed runtime activity while a block executes."""
    if not _events_enabled():
        yield
        return

    console = _get_console()
    if console is None:
        runtime_event(message, style=style)
        yield
        return

    _enter_live_context()
    try:
        with console.status(
            f"[bold blue][misen][/bold blue] {message}",
            spinner=spinner,
            spinner_style=style,
        ):
            yield
    finally:
        _exit_live_context()


@contextlib.contextmanager
def runtime_progress(description: str, *, total: int) -> Iterator[Callable[[int], None]]:
    """Render a progress bar and return an ``advance(step)`` callback."""

    def noop(_: int = 1) -> None:
        return

    if not _events_enabled() or total <= 0:
        yield noop
        return

    console = _get_console()
    if console is None:
        completed = 0
        runtime_event(f"{description} (0/{total})", style="cyan")

        def advance_fallback(step: int = 1) -> None:
            nonlocal completed
            completed = min(total, completed + max(step, 0))
            runtime_event(f"{description} ({completed}/{total})", style="dim")

        yield advance_fallback
        return

    _enter_live_context()
    try:
        with Progress(
            TextColumn("[bold blue][misen][/bold blue] {task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task_id = progress.add_task(description=description, total=total)

            def advance(step: int = 1) -> None:
                progress.advance(task_id, step)

            yield advance
    finally:
        _exit_live_context()


def runtime_job_pending(job_key: int, label: str) -> None:
    """Register one pending local job in the live status board."""
    _job_board_action(_JOB_BOARD.pending, job_key, label=label)


def runtime_job_running(job_key: int, *, job_id: str | None, pid: int | None) -> None:
    """Mark one local job as running in the live status board."""
    _job_board_action(_JOB_BOARD.running, job_key, job_id=job_id, pid=pid)


def runtime_job_done(job_key: int) -> None:
    """Mark one local job as complete in the live status board."""
    _job_board_action(_JOB_BOARD.done, job_key)


def runtime_job_failed(job_key: int) -> None:
    """Mark one local job as failed in the live status board."""
    _job_board_action(_JOB_BOARD.failed, job_key)


def runtime_job_summary_lines(rows: list[RuntimeJobSummary]) -> list[str]:
    """Format final job summary rows for terminal output."""
    state_order = {
        "done": 0,
        "failed": 1,
        "running": 2,
        "pending": 3,
        "unknown": 4,
    }
    ordered_rows = sorted(rows, key=lambda row: (state_order.get(row.state, 99), row.label))

    lines: list[str] = []
    for row in ordered_rows:
        state_text = "complete" if row.state == "done" else row.state
        detail_suffix = _job_detail_suffix(job_id=row.job_id, pid=row.pid)
        lines.append(f"{state_text:<8} {row.label}{detail_suffix}")
    return lines


def task_label(
    task: Task[Any],
    *,
    include_hash: bool = True,
    include_arguments: bool = False,
    include_dependent_arguments: bool = False,
) -> str:
    """Return a compact human-readable label for a task.

    Args:
        task: Task to format.
        include_hash: Include task hash suffix.
        include_arguments: Include formatted function arguments.
        include_dependent_arguments: Kept for compatibility. ``Task.__repr__``
            excludes dependent-task arguments by design, so this has no effect.
    """
    _ = include_dependent_arguments
    label = _task_repr_label(repr(task))
    label_without_hash, hash_suffix = _split_hash_suffix(label)
    label_core = label_without_hash

    if not include_arguments and label_core.endswith(")") and "(" in label_core:
        label_core = label_core[: label_core.index("(")]

    if include_hash and hash_suffix is not None:
        return f"{label_core} [{hash_suffix}]"

    return label_core


def work_unit_label(work_unit: WorkUnit) -> str:
    """Return a compact human-readable label for a work unit root task."""
    return _work_unit_repr_label(repr(work_unit))


def _task_repr_label(task_repr: str) -> str:
    """Extract inner label from ``Task.__repr__`` output."""
    task_repr = task_repr.removesuffix(" [C]")
    if task_repr.startswith("Task(") and task_repr.endswith(")"):
        return task_repr[len("Task(") : -1]
    return task_repr


def _work_unit_repr_label(work_unit_repr: str) -> str:
    """Extract inner label from ``WorkUnit.__repr__`` output."""
    if work_unit_repr.startswith("WorkUnit(") and work_unit_repr.endswith(")"):
        return work_unit_repr[len("WorkUnit(") : -1]
    return work_unit_repr


def _split_hash_suffix(label: str) -> tuple[str, str | None]:
    """Split trailing `` [HASH]`` suffix from a task label."""
    if not label.endswith("]"):
        return label, None

    prefix, separator, remainder = label.rpartition(" [")
    if not separator or not remainder.endswith("]"):
        return label, None

    hash_suffix = remainder[:-1]
    return (prefix, hash_suffix) if hash_suffix else (label, None)


def _events_enabled() -> bool:
    return _env_toggle_enabled("MISEN_RUNTIME_EVENTS")


def _job_board_enabled() -> bool:
    return _env_toggle_enabled(_JOB_BOARD_ENV)


def _env_toggle_enabled(env_name: str, default: str = "1") -> bool:
    value = os.getenv(env_name, default).strip().lower()
    return value not in _FALSEY


def _job_board_action(
    action: Callable[..., None],
    /,
    *args: Any,
    **kwargs: Any,
) -> None:
    if not _events_enabled() or not _job_board_enabled():
        return
    action(*args, **kwargs)


def _job_detail_suffix(*, job_id: str | None, pid: int | None) -> str:
    details: list[str] = []
    if job_id is not None:
        details.append(f"job_id={job_id}")
    if pid is not None:
        details.append(f"pid={pid}")
    return f" ({', '.join(details)})" if details else ""


def _live_context_active() -> bool:
    with _LIVE_CONTEXT_LOCK:
        return _LIVE_CONTEXT["depth"] > 0


def _enter_live_context() -> None:
    with _LIVE_CONTEXT_LOCK:
        _LIVE_CONTEXT["depth"] += 1


def _exit_live_context() -> None:
    with _LIVE_CONTEXT_LOCK:
        _LIVE_CONTEXT["depth"] = max(0, _LIVE_CONTEXT["depth"] - 1)
    _JOB_BOARD.on_live_context_exit()


@cache
def _get_console() -> Console | None:
    return Console(stderr=True, soft_wrap=True)
