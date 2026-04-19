"""Terminal UI for monitoring submitted task graphs."""

from __future__ import annotations

import contextlib
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from rich.console import Console
from rich.text import Text

from misen.utils.cli.display import format_task_line_text, iter_task_arg_children
from misen.utils.runtime_events import RuntimeJobSummary, runtime_job_summary_lines, task_label

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from misen.executor import Job
    from misen.tasks import Task
    from misen.utils.graph import DependencyGraph
    from misen.utils.work_unit import WorkUnit
    from misen.workspace import Workspace

__all__ = [
    "JobState",
    "Mode",
    "submit_and_watch_jobs",
    "watch_tasks",
]

JobState = Literal["pending", "running", "done", "failed", "unknown"]
Mode = Literal["task", "job"]
_TERMINAL_STATES = frozenset({"done", "failed"})
_STATE_STYLES: dict[JobState, str] = {
    "pending": "yellow",
    "running": "cyan",
    "done": "green",
    "failed": "bold red",
    "unknown": "magenta",
}
_STATE_ICONS: dict[JobState, str] = {
    "pending": "○",
    "running": "◐",
    "done": "●",
    "failed": "✗",
    "unknown": "?",
}
_TASK_EMPHASIS_STYLE = "underline"
_JOB_EMPHASIS_STYLE = "on grey35"


def submit_and_watch_jobs(*, experiment: Any, executor: Any, workspace: Any) -> None:
    """Submit experiment tasks and monitor resulting jobs via the TUI."""
    named_tasks = experiment.tasks()
    tasks = set(named_tasks.values())
    with _runtime_job_board_suppressed():
        job_graph, snapshot = executor.submit(tasks=tasks, workspace=workspace, blocking=False)
        with _runtime_events_suppressed():
            watch_tasks(named_tasks=named_tasks, job_graph=job_graph, workspace=workspace)
    jobs = list(job_graph.nodes())
    if all(_safe_job_state(job) in _TERMINAL_STATES for job in jobs):
        executor.cleanup_snapshot(snapshot)
    _print_final_summary(jobs)


def watch_tasks(
    *,
    named_tasks: Mapping[str, Task[Any]],
    job_graph: DependencyGraph[Job],
    workspace: Workspace,
    poll_interval_s: float = 0.2,
) -> None:
    """Render the task dependency tree and stream logs until users exit."""
    console = Console(stderr=True, soft_wrap=True)
    if not job_graph.nodes():
        console.print("[bold blue][misen][/bold blue] No jobs were submitted.", style="dim")
        return
    _run_textual_task_monitor(
        named_tasks=named_tasks,
        job_graph=job_graph,
        workspace=workspace,
        poll_interval_s=poll_interval_s,
    )


@dataclass
class _TaskTreeNode:
    """Bookkeeping for a rendered tree node backed by a ``Task``."""

    task: Task[Any]
    tree_node: Any
    arg_prefix: str | None
    named_as: str | None
    work_unit: WorkUnit | None = None
    state: JobState = "unknown"


@dataclass
class _JobStateIndex:
    """Map work units and cacheable root tasks to the backing ``Job``."""

    wu_to_job: dict[WorkUnit, Job] = field(default_factory=dict)
    wu_by_root: dict[Task[Any], WorkUnit] = field(default_factory=dict)

    @classmethod
    def build(cls, job_graph: DependencyGraph[Job]) -> _JobStateIndex:
        index = cls()
        for job in job_graph.nodes():
            wu = job.work_unit
            index.wu_to_job[wu] = job
            index.wu_by_root[wu.root] = wu
        return index

    def job_for_work_unit(self, wu: WorkUnit | None) -> Job | None:
        return self.wu_to_job.get(wu) if wu is not None else None

    def work_unit_of_root(self, task: Task[Any]) -> WorkUnit | None:
        return self.wu_by_root.get(task)


def _safe_job_state(job: Job) -> JobState:
    try:
        state = job.state()
    except (FileNotFoundError, OSError, RuntimeError):
        return "unknown"
    return state if state in _STATE_STYLES else "unknown"


def _render_node_label(entry: _TaskTreeNode, *, emphasis_style: str | None = None) -> Text:
    """Return the styled Textual label for a task tree node."""
    text = Text()
    text.append(_STATE_ICONS[entry.state], style=_STATE_STYLES[entry.state])
    text.append(" ")
    text.append_text(format_task_line_text(entry.task, prefix=entry.arg_prefix))
    if emphasis_style is not None:
        text.stylize(emphasis_style)
    return text


def _render_summary(jobs: list[Job], states: list[JobState]) -> Text:
    counts = Counter(states)
    summary = Text("Jobs: ")
    summary.append(str(len(jobs)), style="bold")
    summary.append("  ")
    for state in ("pending", "running", "done", "failed", "unknown"):
        summary.append(f"{state}=", style="dim")
        summary.append(str(counts.get(state, 0)), style=_STATE_STYLES[state])
        summary.append("  ")
    return summary


def _run_textual_task_monitor(
    *,
    named_tasks: Mapping[str, Task[Any]],
    job_graph: DependencyGraph[Job],
    workspace: Workspace,
    poll_interval_s: float,
) -> None:
    try:
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.containers import Horizontal, Vertical
        from textual.widgets import Footer, Header, RichLog, Static, Tree
    except ModuleNotFoundError as e:
        msg = (
            "Textual is required for the run TUI but is not installed. "
            "Install dependencies (for example: `uv sync`) and retry."
        )
        raise RuntimeError(msg) from e

    index = _JobStateIndex.build(job_graph)

    class _FilteredTree(Tree):
        """Tree whose arrow-key cursor snaps to an explicit list of stop lines."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.scrollable_lines: list[int] = []

        def action_cursor_up(self) -> None:
            self._move_cursor(-1)

        def action_cursor_down(self) -> None:
            self._move_cursor(+1)

        def _move_cursor(self, direction: int) -> None:
            if not self.scrollable_lines:
                return
            current = self.cursor_line
            if direction > 0:
                later = [line for line in self.scrollable_lines if line > current]
                target = min(later) if later else max(self.scrollable_lines)
            else:
                earlier = [line for line in self.scrollable_lines if line < current]
                target = max(earlier) if earlier else min(self.scrollable_lines)
            self.cursor_line = target

    class _TaskMonitorApp(App[None]):
        TITLE = "misen run"
        ENABLE_COMMAND_PALETTE = False
        BINDINGS = (
            Binding("escape", "quit", "Quit"),
            Binding("tab", "toggle_mode", "Toggle task/job log", priority=True),
        )
        CSS = """
        Screen { layout: vertical; }
        #summary { height: 1; padding: 0 1; }
        #main-content { height: 1fr; }
        #task-tree { width: 1fr; min-width: 32; padding: 0 1; }
        #log-panel { width: 1fr; border-left: solid green; }
        #log-title { height: 1; padding: 0 1; color: $accent; text-style: bold; }
        #log-viewer { height: 1fr; padding: 0 1; }
        """

        def __init__(self) -> None:
            super().__init__()
            self._mode: Mode = "task"
            self._entries: list[_TaskTreeNode] = []
            self._entry_by_node_id: dict[int, _TaskTreeNode] = {}
            self._wu_canonical: dict[WorkUnit, _TaskTreeNode] = {}
            self._cursor_entry: _TaskTreeNode | None = None
            self._log_offset: int = 0
            self._log_key: tuple[Any, ...] | None = None

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            yield Static(id="summary")
            with Horizontal(id="main-content"):
                yield _FilteredTree("Experiment", id="task-tree")
                with Vertical(id="log-panel"):
                    yield Static("Task log", id="log-title")
                    yield RichLog(id="log-viewer", highlight=True, markup=False, wrap=True)
            yield Footer()

        def on_mount(self) -> None:
            tree = self.query_one("#task-tree", _FilteredTree)
            tree.show_root = False
            tree.root.expand()
            self._build_tree(tree)
            self._refresh_states()
            self._repaint_tree()
            self._render_summary_widget()
            self._update_log_title()
            self._stream_log_chunk()
            self.call_after_refresh(self._post_mount_focus)
            self.set_interval(max(0.1, poll_interval_s), self._tick)

        def _post_mount_focus(self) -> None:
            tree = self.query_one("#task-tree", _FilteredTree)
            tree.focus()
            self._recompute_scrollable_lines()
            self._snap_cursor_to_mode_stop()

        def _build_tree(self, tree: Any) -> None:
            rendered: set[Task[Any]] = set()

            def add_subtree(
                parent_node: Any,
                task: Task[Any],
                arg_prefix: str | None,
                named_as: str | None,
                ancestry: set[Task[Any]],
                parent_wu: WorkUnit | None,
            ) -> None:
                if task in rendered or task in ancestry:
                    return
                node_wu = index.work_unit_of_root(task) or parent_wu
                entry = _TaskTreeNode(
                    task=task,
                    tree_node=None,
                    arg_prefix=arg_prefix,
                    named_as=named_as,
                    work_unit=node_wu,
                )
                node = parent_node.add(_render_node_label(entry), data=entry, expand=True)
                entry.tree_node = node
                self._entries.append(entry)
                self._entry_by_node_id[id(node)] = entry
                if node_wu is not None and node_wu not in self._wu_canonical:
                    self._wu_canonical[node_wu] = entry

                rendered.add(task)
                next_ancestry = ancestry | {task}
                for child_prefix, child_task in iter_task_arg_children(task):
                    add_subtree(node, child_task, child_prefix, None, next_ancestry, node_wu)

            for alias, task in sorted(named_tasks.items()):
                add_subtree(tree.root, task, None, alias, set(), None)

            if self._entries and self._cursor_entry is None:
                self._cursor_entry = self._entries[0]

        def _refresh_states(self) -> None:
            for entry in self._entries:
                job = index.job_for_work_unit(entry.work_unit)
                entry.state = _safe_job_state(job) if job is not None else "unknown"

        def _repaint_tree(self) -> None:
            cursor_entry = self._cursor_entry
            emphasized_deps: set[Task[Any]] = set()
            emphasized_wu_tasks: set[Task[Any]] = set()
            if cursor_entry is not None:
                if self._mode == "task":
                    emphasized_deps = set(cursor_entry.task.dependencies)
                elif cursor_entry.work_unit is not None:
                    emphasized_wu_tasks = set(cursor_entry.work_unit.graph.nodes())
            for entry in self._entries:
                if entry is cursor_entry:
                    style: str | None = None
                elif self._mode == "task":
                    style = _TASK_EMPHASIS_STYLE if entry.task in emphasized_deps else None
                else:
                    style = _JOB_EMPHASIS_STYLE if entry.task in emphasized_wu_tasks else None
                entry.tree_node.set_label(_render_node_label(entry, emphasis_style=style))

        def _render_summary_widget(self) -> None:
            summary_widget = self.query_one("#summary", Static)
            jobs = list(job_graph.nodes())
            states = [_safe_job_state(job) for job in jobs]
            summary_widget.update(_render_summary(jobs, states))

        def _update_log_title(self) -> None:
            title = self.query_one("#log-title", Static)
            title.update("Task log" if self._mode == "task" else "Job log")

        def _tick(self) -> None:
            if not self.is_mounted:
                return
            self._refresh_states()
            self._repaint_tree()
            self._render_summary_widget()
            self._stream_log_chunk()

        def action_toggle_mode(self) -> None:
            self._mode = "job" if self._mode == "task" else "task"
            self._update_log_title()
            self._recompute_scrollable_lines()
            self._snap_cursor_to_mode_stop()
            self._reset_log()
            self._repaint_tree()
            self._stream_log_chunk()

        def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
            entry = self._entry_by_node_id.get(id(event.node))
            if entry is None or entry is self._cursor_entry:
                return
            self._cursor_entry = entry
            self._reset_log()
            self._repaint_tree()
            self._stream_log_chunk()

        def _recompute_scrollable_lines(self) -> None:
            tree = self.query_one("#task-tree", _FilteredTree)
            entries = self._entries if self._mode == "task" else list(self._wu_canonical.values())
            tree.scrollable_lines = sorted(
                {e.tree_node.line for e in entries if e.tree_node.line >= 0}
            )

        def _snap_cursor_to_mode_stop(self) -> None:
            if self._cursor_entry is None:
                return
            if self._mode == "task":
                target: _TaskTreeNode | None = self._cursor_entry
            else:
                target = self._wu_canonical.get(self._cursor_entry.work_unit)
                if target is None:
                    target = next(iter(self._wu_canonical.values()), None)
            if target is not None and target is not self._cursor_entry:
                self._cursor_entry = target
            if target is not None and target.tree_node.line >= 0:
                tree = self.query_one("#task-tree", _FilteredTree)
                tree.cursor_line = target.tree_node.line

        def _reset_log(self) -> None:
            self._log_key = None
            self._log_offset = 0
            log_viewer = self.query_one("#log-viewer", RichLog)
            log_viewer.clear()

        def _stream_log_chunk(self) -> None:
            entry = self._cursor_entry
            if entry is None:
                return
            log_viewer = self.query_one("#log-viewer", RichLog)

            key, opener, placeholder = self._resolve_log_source(entry)
            is_new_source = key != self._log_key
            if is_new_source:
                self._log_key = key
                self._log_offset = 0
                log_viewer.clear()

            if opener is None:
                if is_new_source:
                    log_viewer.write(Text(placeholder, style="dim italic"))
                return

            try:
                with opener() as f:
                    f.seek(self._log_offset)
                    chunk = f.read()
                    self._log_offset = f.tell()
            except FileNotFoundError:
                if is_new_source:
                    log_viewer.write(Text(placeholder, style="dim italic"))
                return
            except Exception as exc:  # noqa: BLE001
                if is_new_source:
                    log_viewer.write(
                        Text(f"(log unavailable: {exc.__class__.__name__})", style="dim italic")
                    )
                return

            if chunk:
                log_viewer.write(Text.from_ansi(chunk.rstrip("\n")))

        def _resolve_log_source(
            self, entry: _TaskTreeNode
        ) -> tuple[tuple[Any, ...], Any, str]:
            """Return ``(key, opener, placeholder)`` for the log matching the current mode.

            ``opener`` is either a zero-arg callable returning a readable text file or
            ``None`` when no log is resolvable. ``placeholder`` is the message shown
            when there's nothing to stream (either resolution failed or the file is missing).
            """
            if self._mode == "task":
                key: tuple[Any, ...] = ("task", id(entry.task))
                return key, lambda: workspace.open_task_log(entry.task, mode="r"), "(no task log yet)"
            wu = entry.work_unit
            if wu is None:
                return ("job", None), None, "(no job assigned)"
            log_path = self._resolve_job_log_path(wu)
            if log_path is None:
                return ("job", id(wu)), None, "(no job log yet)"
            return (
                ("job", str(log_path)),
                lambda: log_path.open("r", encoding="utf-8", errors="replace"),
                "(no job log yet)",
            )

        def _resolve_job_log_path(self, wu: WorkUnit):
            """Return the freshest job log path for a work unit, if one exists.

            Prefers the live ``Job.log_path`` when the file is present; otherwise
            falls back to the most recently modified archived log discoverable via
            ``workspace.job_log_iter(wu)``.
            """
            candidates: list[Any] = []
            job = index.job_for_work_unit(wu)
            if job is not None and job.log_path is not None:
                candidates.append(job.log_path)
            try:
                candidates.extend(workspace.job_log_iter(wu))
            except (AttributeError, OSError, FileNotFoundError):
                pass
            existing = [p for p in candidates if p.exists()]
            if not existing:
                return None
            return max(existing, key=lambda p: p.stat().st_mtime)

    app = _TaskMonitorApp()
    app.run()


@contextlib.contextmanager
def _env_var(name: str, value: str) -> Iterator[None]:
    """Temporarily set environment variable ``name`` to ``value``."""
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _runtime_events_suppressed() -> Any:
    """Disable runtime event widgets while the Textual app is running."""
    return _env_var("MISEN_RUNTIME_EVENTS", "0")


def _runtime_job_board_suppressed() -> Any:
    """Disable the runtime job-board rows while keeping regular runtime events."""
    return _env_var("MISEN_RUNTIME_JOB_BOARD", "0")


def _print_final_summary(jobs: list[Job]) -> None:
    rows = [
        RuntimeJobSummary(
            label=task_label(job.root, include_hash=False, include_arguments=True),
            state=_safe_job_state(job),
        )
        for job in jobs
    ]
    lines = runtime_job_summary_lines(rows)
    if not lines:
        return

    console = Console(stderr=True, soft_wrap=True)
    for line in lines:
        console.print(line)
