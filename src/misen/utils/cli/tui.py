"""Terminal UI for monitoring submitted task graphs."""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from rich.console import Console
from rich.text import Text
from rich.tree import Tree as RichTree

from misen.exceptions import CacheError
from misen.executor import CompletedJob, JobState, bulk_job_states
from misen.utils.cli.display import format_task_line_markup, format_task_line_text, iter_task_arg_children
from misen.utils.runtime_events import task_label

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    from textual import events

    from misen.executor import Job
    from misen.tasks import Task
    from misen.utils.graph import DependencyGraph
    from misen.utils.work_unit import WorkUnit
    from misen.workspace import Workspace

__all__ = [
    "JobState",
    "Mode",
    "run_without_tui",
    "submit_and_watch_jobs",
    "watch_tasks",
]

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
    named_tasks = experiment.normalized_tasks()
    tasks = set(named_tasks.values())
    with _runtime_job_board_suppressed():
        job_graph, snapshot = executor.submit(tasks=tasks, workspace=workspace, blocking=False)
        with _runtime_events_suppressed():
            watch_tasks(named_tasks=named_tasks, job_graph=job_graph, workspace=workspace)
    jobs = list(job_graph.nodes())
    final_states = bulk_job_states(jobs)
    if all(final_states.get(job, "unknown") in _TERMINAL_STATES for job in jobs):
        executor.cleanup_snapshot(snapshot)
    _print_final_tree(named_tasks=named_tasks, job_graph=job_graph, states=final_states)


def run_without_tui(*, experiment: Any, executor: Any, workspace: Any) -> None:
    """Submit experiment tasks without the full TUI, rendering a dependency tree.

    When stderr is a terminal the tree re-renders in place while jobs run. When
    stderr is piped (CI/logs), one line is emitted per job state transition and
    a final tree is printed at the end so the output still has structure.
    """
    named_tasks = experiment.normalized_tasks()
    tasks = set(named_tasks.values())
    console = Console(stderr=True, soft_wrap=True)
    final_states: dict[Job, JobState] = {}
    with _runtime_job_board_suppressed():
        job_graph, snapshot = executor.submit(tasks=tasks, workspace=workspace, blocking=False)
        if not job_graph.nodes():
            console.print("[bold blue][misen][/bold blue] No jobs were submitted.", style="dim")
            return
        try:
            if console.is_terminal:
                _watch_live_tree(named_tasks=named_tasks, job_graph=job_graph, console=console)
            else:
                _watch_line_events(job_graph=job_graph, console=console)
        finally:
            jobs = list(job_graph.nodes())
            final_states = bulk_job_states(jobs)
            if all(final_states.get(job, "unknown") in _TERMINAL_STATES for job in jobs):
                executor.cleanup_snapshot(snapshot)
    if not console.is_terminal:
        _print_final_tree(named_tasks=named_tasks, job_graph=job_graph, console=console, states=final_states)


def watch_tasks(
    *,
    named_tasks: Mapping[str, Task[Any]],
    job_graph: DependencyGraph[Job],
    workspace: Workspace,
    poll_interval_s: float = 0.2,
    state_poll_interval_s: float = 2.0,
) -> None:
    """Render the task dependency tree and stream logs until users exit.

    Args:
        named_tasks: User-named root tasks to render.
        job_graph: Dependency graph of dispatched job handles.
        workspace: Workspace whose log files the UI will tail.
        poll_interval_s: How often the UI re-reads log files. The default is
            chosen to feel "live" without burning CPU.
        state_poll_interval_s: How often backend job state is polled. State
            queries can be expensive on busy SLURM controllers, so they
            run on a slower cadence than log streaming and execute on a
            background thread to avoid blocking the UI.
    """
    console = Console(stderr=True, soft_wrap=True)
    if not job_graph.nodes():
        console.print("[bold blue][misen][/bold blue] No jobs were submitted.", style="dim")
        return
    _run_textual_task_monitor(
        named_tasks=named_tasks,
        job_graph=job_graph,
        workspace=workspace,
        poll_interval_s=poll_interval_s,
        state_poll_interval_s=state_poll_interval_s,
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


def _canonical_parent_edges(
    named_tasks: Mapping[str, Task[Any]],
) -> tuple[list[tuple[str, Task[Any]]], dict[Task[Any], tuple[Task[Any], str] | None]]:
    """Pick a single (parent, arg-label) for each task that places it at maximum depth.

    The result is a deepest-placement spanning tree of the task DAG: every task
    reachable from a top-level named root is rendered exactly once, under the
    parent edge whose path from a root is longest. Ties are broken by
    topological visit order, which mirrors argument iteration order from each
    parent.

    Returns a ``(roots, canonical)`` pair. ``roots`` is the alphabetically
    sorted list of ``(alias, task)`` pairs that aren't reachable as a
    descendant of any other named task. ``canonical[task]`` is ``None`` for
    those roots and ``(parent, label)`` for every other reachable task.
    """
    all_descendants: set[Task[Any]] = set()

    def collect_descendants(task: Task[Any], seen: set[Task[Any]]) -> None:
        if task in seen:
            return
        seen.add(task)
        for _, child in iter_task_arg_children(task):
            all_descendants.add(child)
            collect_descendants(child, seen)

    for named_task in named_tasks.values():
        collect_descendants(named_task, set())

    roots: list[tuple[str, Task[Any]]] = [
        (alias, task) for alias, task in sorted(named_tasks.items()) if task not in all_descendants
    ]
    root_tasks = {task for _, task in roots}

    in_count: dict[Task[Any], int] = {}

    def explore(task: Task[Any], seen: set[Task[Any]]) -> None:
        if task in seen:
            return
        seen.add(task)
        in_count.setdefault(task, 0)
        for _, child in iter_task_arg_children(task):
            in_count[child] = in_count.get(child, 0) + 1
            explore(child, seen)

    explored: set[Task[Any]] = set()
    for task in root_tasks:
        explore(task, explored)

    depth: dict[Task[Any], int] = {task: 0 for task in root_tasks}
    canonical: dict[Task[Any], tuple[Task[Any], str] | None] = {task: None for task in root_tasks}
    remaining = dict(in_count)
    for task in root_tasks:
        remaining[task] = 0

    queue: list[Task[Any]] = [task for _, task in roots]
    head = 0
    while head < len(queue):
        parent = queue[head]
        head += 1
        for label, child in iter_task_arg_children(parent):
            candidate = depth[parent] + 1
            if child not in depth or candidate > depth[child]:
                depth[child] = candidate
                canonical[child] = (parent, label)
            if child in remaining:
                remaining[child] -= 1
                if remaining[child] == 0:
                    queue.append(child)

    return roots, canonical


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
    state_poll_interval_s: float,
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
    all_jobs: list[Job] = list(job_graph.nodes())

    class _FilteredTree(Tree):
        """Tree whose arrow-key cursor snaps to an explicit list of stop lines."""

        # Shadow the base Tree's hidden ``space`` binding with a visible one so
        # the app footer advertises expand/collapse. Mouse-based toggling is
        # disabled via ``_on_click`` — expand/collapse is keyboard-only.
        BINDINGS: ClassVar[list[Any]] = [Binding("space", "toggle_node", "Expand/collapse")]

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            # ``auto_expand`` would re-toggle nodes inside
            # ``on_tree_node_selected``; leaving it on would reintroduce
            # click-based expand via the select path even though we suppress
            # the chevron branch in ``_on_click``.
            self.auto_expand = False
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

        async def _on_click(self, event: Any) -> None:
            # Textual dispatches ``_on_click`` along the entire MRO, so the
            # stock ``Tree._on_click`` would still toggle nodes on chevron
            # clicks unless we call ``prevent_default``. We handle cursor
            # movement ourselves and suppress the base handler entirely;
            # expand/collapse is keyboard-only.
            event.prevent_default()
            async with self.lock:
                meta = event.style.meta
                if "line" in meta and not meta.get("toggle", False):
                    self.cursor_line = meta["line"]

    class _LogPane(RichLog):
        """RichLog with smart auto-scroll and click-drag text selection.

        Two behaviors added on top of stock RichLog:

        1. Writes only auto-scroll to the end when the viewport is already
           pinned there, so manual scrolling and active selections aren't
           yanked around by streaming log updates.
        2. Selection rendering and text extraction. Stock RichLog renders
           via the line API and overrides ``render()`` to return a Panel,
           which makes ``Widget.get_selection`` return ``None`` and leaves
           the visual selection unshaded. We override both sides so
           click-drag highlights and Ctrl+C / Cmd+C copy work on logs.
        """

        def write(  # type: ignore[override]
            self,
            content: Any,
            width: int | None = None,
            expand: bool = False,  # noqa: FBT001, FBT002
            shrink: bool = True,  # noqa: FBT001, FBT002
            scroll_end: bool | None = None,  # noqa: FBT001
            animate: bool = False,  # noqa: FBT001, FBT002
        ) -> Any:
            if scroll_end is None:
                scroll_end = self.is_vertical_scroll_end
            return super().write(
                content,
                width=width,
                expand=expand,
                shrink=shrink,
                scroll_end=scroll_end,
                animate=animate,
            )

        def get_selection(self, selection: Any) -> tuple[str, str] | None:  # type: ignore[override]
            if not self.lines:
                return None
            text = "\n".join(strip.text for strip in self.lines)
            extracted = selection.extract(text)
            if not extracted:
                return None
            return extracted, "\n"

        def selection_updated(self, selection: Any) -> None:  # type: ignore[override]  # noqa: ARG002
            # Invalidate the line cache so selection style changes repaint.
            self._line_cache.clear()
            self.refresh()

        def render_line(self, y: int) -> Any:  # type: ignore[override]
            from rich.segment import Segment
            from textual.strip import Strip

            strip = super().render_line(y)
            scroll_x, scroll_y = self.scroll_offset
            absolute_y = scroll_y + y
            # Stock RichLog renders via the line API but never calls
            # ``Strip.apply_offsets``, which is what attaches the ``offset``
            # meta the compositor reads to resolve a click position into a
            # character offset. Without that meta the Screen sees
            # ``select_offset is None`` and refuses to start a selection —
            # which is why click-drag in the log pane appeared dead.
            strip = strip.apply_offsets(scroll_x, absolute_y)
            selection = self.text_selection
            if selection is None:
                return strip
            span = selection.get_span(absolute_y)
            if span is None:
                return strip
            start_char, end_char = span
            cell_length = strip.cell_length
            # ``selection.get_span`` returns character offsets; ``Strip.divide``
            # takes cell positions. For pure ASCII these are identical, but as
            # soon as a rendered segment has ``len(text) != cell_length`` (Rich
            # can group characters into wider segments — and any wide char,
            # emoji, or combining char also breaks the equivalence) the cuts
            # land in the wrong place and the highlight stops short of (or
            # overshoots) the cursor.
            if end_char == -1:
                end_cell = cell_length
            else:
                end_cell = strip.index_to_cell_position(
                    max(0, end_char - scroll_x)
                )
                end_cell = min(cell_length, end_cell)
            start_cell = strip.index_to_cell_position(
                max(0, start_char - scroll_x)
            )
            start_cell = min(cell_length, start_cell)
            if end_cell <= start_cell:
                return strip
            selection_style = self.screen.get_component_rich_style(
                "screen--selection"
            )
            cuts: list[int] = []
            if start_cell > 0:
                cuts.append(start_cell)
            cuts.append(end_cell)
            if end_cell < cell_length:
                cuts.append(cell_length)
            parts = list(strip.divide(cuts))
            if start_cell > 0:
                pre: Any = parts[0]
                selected_part = parts[1]
                rest = parts[2:]
            else:
                pre = None
                selected_part = parts[0]
                rest = parts[1:]
            # Use post_style so the selection background overrides the
            # segment's existing background (``Strip.apply_style`` prepends
            # the style, which leaves the original bg color intact).
            styled_segments = list(
                Segment.apply_style(iter(selected_part), post_style=selection_style)
            )
            styled_selected = Strip(styled_segments, selected_part.cell_length)
            return Strip.join([pre, styled_selected, *rest])

    class _CopyButton(Static):
        """Footer-style button that copies the current text selection.

        A naive click handler doesn't work: Textual's Screen clears the active
        selection when MouseDown and MouseUp land on the same widget (the
        deselect-on-click affordance). By the time the Click event reaches the
        button, the selection is gone. We work around this by snapshotting the
        selections on MouseDown — before the Screen processes MouseUp — and
        restoring them just before invoking ``screen.copy_text``.
        """

        DEFAULT_CSS = """
        _CopyButton {
            dock: right;
            width: auto;
            height: 1;
            padding: 0 2;
            background: $accent;
            color: $background;
            text-style: bold;
        }
        _CopyButton:hover {
            background: $accent-lighten-1;
        }
        """

        def __init__(self, **kwargs: Any) -> None:
            super().__init__("Copy", **kwargs)
            self._captured_selection: dict[Any, Any] | None = None

        def on_mouse_down(self, event: events.MouseDown) -> None:  # noqa: ARG002
            self._captured_selection = dict(self.screen.selections)

        def on_click(self, event: events.Click) -> None:  # noqa: ARG002
            captured = self._captured_selection
            self._captured_selection = None
            if not captured:
                return
            self.screen.selections = captured
            with contextlib.suppress(Exception):
                self.screen.action_copy_text()

    class _TaskMonitorApp(App[None]):
        TITLE = "misen run"
        ENABLE_COMMAND_PALETTE = False
        # Quit bindings stay declared but are gated by ``check_action`` so they
        # only light up after every job reaches a terminal state. During a run
        # Ctrl+C is the only way out — it raises KeyboardInterrupt in the
        # parent Python process so callers can decide how to handle it.
        # Copy is hidden from the global footer (``show=False``); the log pane
        # carries its own ``Copy`` label so the affordance lives where the
        # selectable text does.
        BINDINGS = (
            Binding("escape", "quit_when_done", "Quit"),
            Binding("q", "quit_when_done", "Quit"),
            Binding("tab", "toggle_mode", "Toggle task/job log", priority=True),
            # ``priority=True`` so this beats Textual's defaults
            # (Screen's ``ctrl+c,super+c`` → screen.copy_text and App's
            # ``ctrl+c`` → help_quit) and Ctrl+C always interrupts.
            Binding("ctrl+c", "interrupt", "Interrupt", priority=True, show=False),
            Binding("super+c,ctrl+shift+c", "screen.copy_text", "Copy", show=False),
            # Suppress Textual's default Ctrl+Q quit binding — exiting mid-run
            # would orphan the submitted Jobs. Ctrl+C remains as the deliberate
            # interrupt path.
            Binding("ctrl+q", "noop", show=False, priority=True),
        )
        POST_RUN_IDLE_TIMEOUT_S = 60.0
        CSS = """
        Screen { layout: vertical; }
        #summary { height: 1; padding: 0 1; }
        #main-content { height: 1fr; }
        #task-tree { width: 1fr; min-width: 32; padding: 0 1; }
        #log-panel { width: 1fr; border-left: solid green; }
        #log-title { height: 1; padding: 0 1; color: $accent; text-style: bold; }
        #log-viewer { height: 1fr; padding: 0 1; }
        #footer-bar { height: 1; }
        #footer-bar Footer { width: 1fr; }
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
            self._all_done: bool = False
            self._last_activity_at: float = time.monotonic()
            self._last_scroll_offsets: tuple[float, float, float, float] | None = None
            """Sampled per tick as a fallback for wheel activity (widgets stop
            scroll events before they bubble to app-level handlers)."""
            self._user_interrupted: bool = False
            """Set by :meth:`action_interrupt` so the runner can re-raise
            ``KeyboardInterrupt`` once the TUI has shut down cleanly."""
            self._job_states: dict[Job, JobState] = dict.fromkeys(all_jobs, "unknown")
            """Single source of truth for per-job state. Refreshed by the
            slow state tick; every UI consumer reads from this cache so a
            slow ``squeue``/``sacct`` call doesn't run N times per render."""
            self._state_poll_pending: bool = False
            """Set while a backgrounded state poll is in flight to prevent
            two overlapping polls when the controller is slow."""

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            yield Static(id="summary")
            with Horizontal(id="main-content"):
                yield _FilteredTree("Experiment", id="task-tree")
                with Vertical(id="log-panel"):
                    yield Static("Task log", id="log-title")
                    yield _LogPane(id="log-viewer", highlight=True, markup=False, wrap=True)
            with Horizontal(id="footer-bar"):
                yield Footer()
                yield _CopyButton(id="copy-button")

        def on_mount(self) -> None:
            tree = self.query_one("#task-tree", _FilteredTree)
            tree.show_root = False
            tree.root.expand()
            self._build_tree(tree)
            # Initial state poll runs synchronously so the first paint shows
            # accurate states; subsequent polls go through the background
            # path so the UI never blocks on SLURM CLI latency.
            self._apply_states(bulk_job_states(all_jobs))
            self._repaint_tree()
            self._render_summary_widget()
            self._update_log_title()
            self._stream_log_chunk()
            self.call_after_refresh(self._post_mount_focus)
            self.set_interval(max(0.05, poll_interval_s), self._log_tick)
            self.set_interval(max(0.5, state_poll_interval_s), self._state_tick)

        def _post_mount_focus(self) -> None:
            tree = self.query_one("#task-tree", _FilteredTree)
            tree.focus()
            self._recompute_scrollable_lines()
            self._snap_cursor_to_mode_stop()

        def _build_tree(self, tree: Any) -> None:
            # Render each task once, under the parent that places it at the
            # deepest position reachable from any top-level named root. Tasks
            # whose only paths run through another named task are rendered
            # under that parent, not at the top level.
            roots, canonical = _canonical_parent_edges(named_tasks)

            def add_subtree(
                parent_node: Any,
                task: Task[Any],
                arg_prefix: str | None,
                named_as: str | None,
                parent_wu: WorkUnit | None,
            ) -> None:
                node_wu = index.work_unit_of_root(task) or parent_wu
                entry = _TaskTreeNode(
                    task=task,
                    tree_node=None,
                    arg_prefix=arg_prefix,
                    named_as=named_as,
                    work_unit=node_wu,
                )
                children = [
                    (label, child)
                    for label, child in iter_task_arg_children(task)
                    if canonical.get(child) == (task, label)
                ]
                has_children = bool(children)
                node = parent_node.add(
                    _render_node_label(entry),
                    data=entry,
                    expand=has_children,
                    allow_expand=has_children,
                )
                entry.tree_node = node
                self._entries.append(entry)
                self._entry_by_node_id[id(node)] = entry
                if node_wu is not None and node_wu not in self._wu_canonical:
                    self._wu_canonical[node_wu] = entry
                for child_label, child_task in children:
                    add_subtree(node, child_task, child_label, None, node_wu)

            for alias, task in roots:
                add_subtree(tree.root, task, None, alias, None)

            if self._entries and self._cursor_entry is None:
                self._cursor_entry = self._entries[0]

        def _apply_states(self, states: dict[Job, JobState]) -> None:
            """Replace the cached state map and propagate it to tree entries."""
            self._job_states = states
            for entry in self._entries:
                job = index.job_for_work_unit(entry.work_unit)
                entry.state = states.get(job, "unknown") if job is not None else "unknown"

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
            states = [self._job_states.get(job, "unknown") for job in jobs]
            summary_widget.update(_render_summary(jobs, states))

        def _update_log_title(self) -> None:
            title = self.query_one("#log-title", Static)
            title.update("Task log" if self._mode == "task" else "Job log")

        def _log_tick(self) -> None:
            """Fast tick: stream log chunks and observe user scroll activity.

            Re-painting the tree here picks up the cursor-emphasis change as
            soon as the user moves the cursor, without waiting on the slower
            state tick.
            """
            if not self.is_mounted:
                return
            self._stream_log_chunk()
            self._poll_scroll_activity()

        async def _state_tick(self) -> None:
            """Slow tick: refresh job states off the UI thread.

            Backend state queries (especially ``squeue``/``sacct`` on a busy
            SLURM controller) can take seconds. Running them on the event
            loop would freeze log streaming for that whole window, so we
            offload to a worker thread via :func:`asyncio.to_thread`. The
            ``_state_poll_pending`` flag guards against firing a second
            poll while the previous one is still in flight.
            """
            if not self.is_mounted or self._state_poll_pending:
                return
            self._state_poll_pending = True
            try:
                states = await asyncio.to_thread(bulk_job_states, all_jobs)
            finally:
                self._state_poll_pending = False
            if not self.is_mounted:
                return
            self._apply_states(states)
            self._repaint_tree()
            self._render_summary_widget()
            self._update_completion_state()

        def _poll_scroll_activity(self) -> None:
            # Widget-level scroll handlers stop mouse-wheel events before they
            # bubble to the app, so sample the scroll offsets each tick and
            # treat any change as user activity for the idle timer.
            try:
                tree = self.query_one("#task-tree", _FilteredTree)
                log_viewer = self.query_one("#log-viewer", _LogPane)
            except Exception:  # noqa: BLE001 — widgets may not be mounted yet
                return
            offsets = (tree.scroll_y, tree.scroll_x, log_viewer.scroll_y, log_viewer.scroll_x)
            if self._last_scroll_offsets is not None and offsets != self._last_scroll_offsets:
                self._mark_activity()
            self._last_scroll_offsets = offsets

        def _all_jobs_terminal(self) -> bool:
            return all(
                self._job_states.get(job, "unknown") in _TERMINAL_STATES for job in job_graph.nodes()
            )

        def _update_completion_state(self) -> None:
            all_done = self._all_jobs_terminal()
            if all_done and not self._all_done:
                self._all_done = True
                # Reset the activity clock at the moment of completion so the
                # idle countdown starts fresh regardless of prior scrolling.
                self._last_activity_at = time.monotonic()
                # Re-evaluate bindings: the gated quit actions just turned on.
                self.refresh_bindings()
            if self._all_done and (
                time.monotonic() - self._last_activity_at >= self.POST_RUN_IDLE_TIMEOUT_S
            ):
                self.exit()

        def _mark_activity(self) -> None:
            self._last_activity_at = time.monotonic()

        def check_action(
            self,
            action: str,
            parameters: tuple[object, ...],  # noqa: ARG002
        ) -> bool | None:
            if action == "quit_when_done":
                # Hide the binding in the footer and block its action while
                # jobs are still running.
                return self._all_done
            return True

        def action_quit_when_done(self) -> None:
            if self._all_done:
                self.exit()

        def action_interrupt(self) -> None:
            """Tear down the TUI and signal a user interrupt to the caller."""
            self._user_interrupted = True
            self.exit()

        def action_noop(self) -> None:
            """No-op action used to swallow keys we want to disable."""

        def on_key(self, event: events.Key) -> None:  # noqa: ARG002
            self._mark_activity()

        def on_mouse_down(self, event: events.MouseDown) -> None:  # noqa: ARG002
            self._mark_activity()

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
            tree.scrollable_lines = sorted({e.tree_node.line for e in entries if e.tree_node.line >= 0})

        def _snap_cursor_to_mode_stop(self) -> None:
            if self._cursor_entry is None:
                return
            if self._mode == "task":
                target: _TaskTreeNode | None = self._cursor_entry
            else:
                wu = self._cursor_entry.work_unit
                target = self._wu_canonical.get(wu) if wu is not None else None
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
            log_viewer = self.query_one("#log-viewer", _LogPane)
            log_viewer.clear()

        def _stream_log_chunk(self) -> None:
            entry = self._cursor_entry
            if entry is None:
                return
            log_viewer = self.query_one("#log-viewer", _LogPane)

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
            except (FileNotFoundError, CacheError):
                # CacheError fires when the task log path can't be resolved
                # yet — its directory is keyed by ``resolved_hash``, which
                # requires every dependency's result hash, which doesn't
                # exist until the deps complete. Treat that the same as
                # "log file not produced yet" and show the placeholder.
                if is_new_source:
                    log_viewer.write(Text(placeholder, style="dim italic"))
                return
            except Exception as exc:  # noqa: BLE001
                if is_new_source:
                    log_viewer.write(Text(f"(log unavailable: {exc.__class__.__name__})", style="dim italic"))
                return

            if chunk:
                log_viewer.write(Text.from_ansi(chunk.rstrip("\n")))

        def _resolve_log_source(self, entry: _TaskTreeNode) -> tuple[tuple[Any, ...], Any, str]:
            """Return ``(key, opener, placeholder)`` for the log matching the current mode.

            ``opener`` is either a zero-arg callable returning a readable text file or
            ``None`` when no log is resolvable. ``placeholder`` is the message shown
            when there's nothing to stream (either resolution failed or the file is missing).
            """
            if self._mode == "task":
                job = index.job_for_work_unit(entry.work_unit)
                if job is not None and not isinstance(job, CompletedJob):
                    # Real current-session job runs this task — pin the task
                    # log lookup to its job_id so we never fall back to an
                    # archived log from a prior session.
                    job_id = job.job_id
                    if job_id is None:
                        return ("task", id(entry.task), "pending"), None, "(no task log yet)"
                    return (
                        ("task", id(entry.task), job_id),
                        lambda: workspace.read_task_log(entry.task, job_id=job_id),
                        "(no task log yet)",
                    )
                # No current-session job, or a cached CompletedJob — fall back
                # to whichever task log was most recently written.
                return (
                    ("task", id(entry.task)),
                    lambda: workspace.read_task_log(entry.task),
                    "(no task log yet)",
                )
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

        def _resolve_job_log_path(self, wu: WorkUnit) -> Path | None:
            """Return the job log path to display for a work unit, if one exists.

            If a real (non-cached) job for this work unit is part of the current
            session, only its live ``Job.log_path`` is returned — never an
            archived log from a prior run, even if the live file has not been
            written yet. For cached work units (``CompletedJob``) or work units
            without a current-session job, falls back to the most recently
            modified archived log from ``workspace.job_log_iter(wu)``.
            """
            job = index.job_for_work_unit(wu)
            if job is not None and not isinstance(job, CompletedJob):
                if job.log_path is not None and job.log_path.exists():
                    return job.log_path
                return None
            candidates: list[Path] = []
            with contextlib.suppress(AttributeError, OSError, FileNotFoundError):
                candidates.extend(workspace.job_log_iter(wu))
            existing = [p for p in candidates if p.exists()]
            if not existing:
                return None
            return max(existing, key=lambda p: p.stat().st_mtime)

    app = _TaskMonitorApp()
    app.run()
    # Propagate Ctrl+C as a real KeyboardInterrupt — Textual swallows the
    # signal and turns it into a key event, so we re-raise here once the TUI
    # has finished tearing itself down.
    if app._user_interrupted:  # noqa: SLF001
        raise KeyboardInterrupt


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


def _build_session_rich_tree(
    named_tasks: Mapping[str, Task[Any]],
    job_graph: DependencyGraph[Job],
    states: Mapping[Job, JobState] | None = None,
) -> RichTree:
    """Build a Rich dependency tree labelled with per-task job states.

    The hierarchy mirrors :meth:`_TaskMonitorApp._build_tree`: each task is
    rendered exactly once, under the parent edge that places it at maximum
    depth from any top-level named root. Other paths to the same task are
    omitted entirely (no back-references).

    ``states`` lets callers pass a pre-computed state map so the tree
    builder doesn't re-poll the backend (one bulk call upstream is much
    cheaper than per-job calls here).
    """
    index = _JobStateIndex.build(job_graph)
    resolved_states: Mapping[Job, JobState] = states if states is not None else bulk_job_states(job_graph.nodes())

    def state_for(task: Task[Any], parent_wu: WorkUnit | None) -> tuple[JobState, WorkUnit | None]:
        wu = index.work_unit_of_root(task) or parent_wu
        job = index.job_for_work_unit(wu)
        state: JobState = resolved_states.get(job, "unknown") if job is not None else "unknown"
        return state, wu

    def render_label(task: Task[Any], state: JobState, arg_prefix: str | None) -> str:
        style = _STATE_STYLES[state]
        icon = _STATE_ICONS[state]
        body = format_task_line_markup(task, prefix=arg_prefix)
        return f"[{style}]{icon}[/{style}] {body}"

    roots, canonical = _canonical_parent_edges(named_tasks)
    root = RichTree("[bold cyan]Tasks[/bold cyan]")

    def add_subtree(
        parent_branch: RichTree,
        task: Task[Any],
        arg_prefix: str | None,
        parent_wu: WorkUnit | None,
    ) -> None:
        state, wu = state_for(task, parent_wu)
        branch = parent_branch.add(render_label(task, state, arg_prefix))
        for label, child in iter_task_arg_children(task):
            if canonical.get(child) == (task, label):
                add_subtree(branch, child, label, wu)

    for _alias, task in roots:
        add_subtree(root, task, None, None)

    return root


def _print_final_tree(
    *,
    named_tasks: Mapping[str, Task[Any]],
    job_graph: DependencyGraph[Job],
    console: Console | None = None,
    states: Mapping[Job, JobState] | None = None,
) -> None:
    """Print a dependency tree with final job states to stderr."""
    if not named_tasks:
        return
    target = console if console is not None else Console(stderr=True, soft_wrap=True)
    target.print(_build_session_rich_tree(named_tasks, job_graph, states))


def _watch_live_tree(
    *,
    named_tasks: Mapping[str, Task[Any]],
    job_graph: DependencyGraph[Job],
    console: Console,
    poll_interval_s: float = 0.25,
) -> None:
    """Re-render the dependency tree in place via Rich ``Live`` until all jobs terminate."""
    from rich.live import Live

    jobs = list(job_graph.nodes())
    states = bulk_job_states(jobs)
    with Live(
        _build_session_rich_tree(named_tasks, job_graph, states),
        console=console,
        refresh_per_second=4,
        transient=False,
    ) as live:
        while not all(states.get(job, "unknown") in _TERMINAL_STATES for job in jobs):
            time.sleep(poll_interval_s)
            states = bulk_job_states(jobs)
            live.update(_build_session_rich_tree(named_tasks, job_graph, states))
        live.update(_build_session_rich_tree(named_tasks, job_graph, states))


def _watch_line_events(
    *,
    job_graph: DependencyGraph[Job],
    console: Console,
    poll_interval_s: float = 0.5,
) -> None:
    """Poll job states and emit one line per transition for non-TTY consumers."""
    jobs = list(job_graph.nodes())
    last: dict[int, JobState] = {}
    while True:
        states = bulk_job_states(jobs)
        all_terminal = True
        for job in jobs:
            state = states.get(job, "unknown")
            if state not in _TERMINAL_STATES:
                all_terminal = False
            if last.get(id(job)) != state:
                last[id(job)] = state
                label = task_label(job.root, include_hash=False, include_arguments=True)
                display = "complete" if state == "done" else state
                console.print(f"{display:<8} {label}")
        if all_terminal:
            return
        time.sleep(poll_interval_s)
