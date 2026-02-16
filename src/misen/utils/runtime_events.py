"""Runtime event printing for interactive observability."""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from rich.console import Console
    from rich.live import Live

    from misen.tasks import Task
    from misen.utils.work_unit import WorkUnit

__all__ = [
    "runtime_activity",
    "runtime_event",
    "runtime_job_done",
    "runtime_job_failed",
    "runtime_job_pending",
    "runtime_job_running",
    "runtime_progress",
    "task_label",
    "work_unit_label",
]

_FALSEY = frozenset({"0", "false", "no", "off"})
_LIVE_CONTEXT: dict[str, int] = {"depth": 0}
_LIVE_CONTEXT_LOCK = threading.Lock()

_JobState = Literal["pending", "running", "done", "failed"]


@dataclass
class _JobStatusLine:
    label: str
    state: _JobState = "pending"
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
                from rich.live import Live as RichLive

                self._live = RichLive(renderable, console=console, refresh_per_second=12, transient=False)
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

    def _render_locked(self) -> object:
        from rich.spinner import Spinner
        from rich.table import Table
        from rich.text import Text

        table = Table.grid(padding=(0, 1))
        table.add_column(width=8, no_wrap=True)
        table.add_column()

        for line in self._entries.values():
            match line.state:
                case "pending":
                    indicator: object = Text("")
                case "running":
                    indicator = Spinner("dots", style="yellow")
                case "done":
                    indicator = Text("complete", style="green")
                case "failed":
                    indicator = Text("failed", style="bold red")

            details = []
            if line.job_id is not None:
                details.append(f"job_id={line.job_id}")
            if line.pid is not None:
                details.append(f"pid={line.pid}")

            detail_suffix = f" ({', '.join(details)})" if details else ""
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

    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
    )

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
    if not _events_enabled():
        return
    _JOB_BOARD.pending(job_key, label=label)


def runtime_job_running(job_key: int, *, job_id: str | None, pid: int | None) -> None:
    """Mark one local job as running in the live status board."""
    if not _events_enabled():
        return
    _JOB_BOARD.running(job_key, job_id=job_id, pid=pid)


def runtime_job_done(job_key: int) -> None:
    """Mark one local job as complete in the live status board."""
    if not _events_enabled():
        return
    _JOB_BOARD.done(job_key)


def runtime_job_failed(job_key: int) -> None:
    """Mark one local job as failed in the live status board."""
    if not _events_enabled():
        return
    _JOB_BOARD.failed(job_key)


def task_label(task: Task[Any]) -> str:
    """Return a compact human-readable label for a task."""
    return f"{task.properties.id} ({task.task_hash().short_b32()})"


def work_unit_label(work_unit: WorkUnit) -> str:
    """Return a compact human-readable label for a work unit root task."""
    return task_label(work_unit.root)


def _events_enabled() -> bool:
    value = os.getenv("MISEN_RUNTIME_EVENTS", "1").strip().lower()
    return value not in _FALSEY


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
    try:
        from rich.console import Console as RichConsole
    except ImportError:
        return None

    return RichConsole(stderr=True, soft_wrap=True)
