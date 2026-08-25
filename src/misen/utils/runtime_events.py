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
    from rich.table import Table

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
_JOB_BOARD_ENV = "MISEN_RUNTIME_JOB_BOARD"

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

    def update(
        self,
        job_key: int,
        state: _JobState,
        *,
        label: str | None = None,
        job_id: str | None = None,
        pid: int | None = None,
    ) -> None:
        """Create or update a job row."""
        with self._lock:
            if state == "pending" and self._live is None and self._entries and self._all_terminal_locked():
                self._entries.clear()

            line = self._entries.get(job_key)
            if line is None:
                line = self._entries[job_key] = _JobStatusLine(label=label if label is not None else f"job-{job_key}")
            if label is not None:
                line.label = label
            line.state = state
            if job_id is not None:
                line.job_id = job_id
            if pid is not None:
                line.pid = pid
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
        all_terminal = self._all_terminal_locked()

        if all_terminal:
            if self._live is None:
                console.print(renderable)
            else:
                self._live.update(renderable, refresh=True)
                self._live.stop()
                self._live = None
            self._entries.clear()
            return

        if self._live is None:
            from rich.live import Live

            self._live = Live(renderable, console=console, refresh_per_second=12, transient=False)
            self._live.start()
        else:
            self._live.update(renderable, refresh=True)

    def _all_terminal_locked(self) -> bool:
        return all(line.state in {"done", "failed"} for line in self._entries.values())

    def _render_locked(self) -> Table:
        from rich.spinner import Spinner
        from rich.table import Table
        from rich.text import Text

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

            table.add_row(indicator, Text(line.label))

        return table


_JOB_BOARD = _RuntimeJobBoard()


def _ui_value(action: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    with contextlib.suppress(Exception, SystemExit):
        return action(*args, **kwargs)
    return None


@contextlib.contextmanager
def _live_widget(widget: Any) -> Iterator[None]:
    _enter_live_context()
    with contextlib.suppress(Exception, SystemExit):
        widget.start()
    try:
        yield
    finally:
        with contextlib.suppress(Exception, SystemExit):
            widget.stop()
        _exit_live_context()


def runtime_event(message: str, *, style: str = "cyan") -> None:
    """Print one runtime event line.

    Output is enabled by default and can be disabled with
    ``MISEN_RUNTIME_EVENTS=0``.
    """
    if not _events_enabled():
        return

    with contextlib.suppress(Exception, SystemExit):
        if console := _get_console():
            console.print(f"[bold blue][misen][/bold blue] {message}", style=style, highlight=False)
        else:
            sys.stderr.write(f"[misen] {message}\n")
            sys.stderr.flush()


@contextlib.contextmanager
def runtime_activity(message: str, *, spinner: str = "dots", style: str = "cyan") -> Iterator[None]:
    """Show a spinner-backed runtime activity while a block executes."""
    if not _events_enabled():
        yield
        return

    console = _ui_value(_get_console)
    if console is None:
        runtime_event(message, style=style)
        yield
        return

    status = _ui_value(
        console.status,
        f"[bold blue][misen][/bold blue] {message}",
        spinner=spinner,
        spinner_style=style,
    )
    if status is None:
        yield
        return
    with _live_widget(status):
        yield


@contextlib.contextmanager
def runtime_progress(description: str, *, total: int) -> Iterator[Callable[[int], None]]:
    """Render a progress bar and return an ``advance(step)`` callback."""

    def noop(_: int = 1) -> None:
        return

    if not _events_enabled() or total <= 0:
        yield noop
        return

    console = _ui_value(_get_console)
    if console is None:
        completed = 0
        runtime_event(f"{description} (0/{total})", style="cyan")

        def advance_fallback(step: int = 1) -> None:
            nonlocal completed
            completed = min(total, completed + max(step, 0))
            runtime_event(f"{description} ({completed}/{total})", style="dim")

        yield advance_fallback
        return

    progress = None
    with contextlib.suppress(Exception, SystemExit):
        from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

        progress = Progress(
            TextColumn("[bold blue][misen][/bold blue] {task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )
    if progress is None:
        yield noop
        return
    with _live_widget(progress):
        task_id = _ui_value(progress.add_task, description, total=total)

        def advance(step: int = 1) -> None:
            if task_id is not None:
                with contextlib.suppress(Exception, SystemExit):
                    progress.advance(task_id, step)

        yield advance


def runtime_job_pending(job_key: int, label: str) -> None:
    """Register one pending local job in the live status board."""
    _job_board_action(_JOB_BOARD.update, job_key, "pending", label=label)


def runtime_job_running(job_key: int, *, job_id: str | None, pid: int | None) -> None:
    """Mark one local job as running in the live status board."""
    _job_board_action(_JOB_BOARD.update, job_key, "running", job_id=job_id, pid=pid)


def runtime_job_done(job_key: int) -> None:
    """Mark one local job as complete in the live status board."""
    _job_board_action(_JOB_BOARD.update, job_key, "done")


def runtime_job_failed(job_key: int) -> None:
    """Mark one local job as failed in the live status board."""
    _job_board_action(_JOB_BOARD.update, job_key, "failed")


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
        include_dependent_arguments: Kept for compatibility. Dependent-task
            arguments are excluded by design, so this has no effect.
    """
    _ = include_dependent_arguments
    label_core = task.func.__name__
    if include_arguments:
        argument_items = task._repr_argument_items()  # noqa: SLF001
        if argument_items:
            label_core = f"{label_core}({', '.join(argument_items)})"

    return f"{label_core} [{task.task_hash().short_b32()}]" if include_hash else label_core


def work_unit_label(work_unit: WorkUnit) -> str:
    """Return a compact human-readable label for a work unit root task."""
    return _work_unit_repr_label(repr(work_unit))


def _work_unit_repr_label(work_unit_repr: str) -> str:
    """Extract inner label from ``WorkUnit.__repr__`` output."""
    wrapped = work_unit_repr.startswith("WorkUnit(") and work_unit_repr.endswith(")")
    return work_unit_repr[len("WorkUnit(") : -1] if wrapped else work_unit_repr


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
    with contextlib.suppress(Exception, SystemExit):
        action(*args, **kwargs)


def _live_context_active() -> bool:
    with _LIVE_CONTEXT_LOCK:
        return _LIVE_CONTEXT["depth"] > 0


def _enter_live_context() -> None:
    with _LIVE_CONTEXT_LOCK:
        _LIVE_CONTEXT["depth"] += 1


def _exit_live_context() -> None:
    with _LIVE_CONTEXT_LOCK:
        _LIVE_CONTEXT["depth"] = max(0, _LIVE_CONTEXT["depth"] - 1)
    _job_board_action(_JOB_BOARD.on_live_context_exit)


@cache
def _get_console() -> Console | None:
    from rich.console import Console

    return Console(stderr=True, soft_wrap=True)
