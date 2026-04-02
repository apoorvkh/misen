"""Terminal UI helpers for monitoring submitted job graphs."""

from __future__ import annotations

import contextlib
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TextIO

from rich.console import Console
from rich.text import Text

from misen.utils.runtime_events import RuntimeJobSummary, runtime_job_summary_lines, task_label

if TYPE_CHECKING:
    from collections.abc import Iterator

    from misen.executor import Job
    from misen.utils.graph import DependencyGraph

__all__ = [
    "JobSnapshot",
    "snapshot_jobs",
    "stream_job_graph_logs",
    "submit_and_stream_job_logs",
    "submit_and_watch_jobs",
    "watch_job_graph",
]

JobState = Literal["pending", "running", "done", "failed", "unknown"]
_TERMINAL_STATES = frozenset({"done", "failed"})
_STATE_STYLES: dict[JobState, str] = {
    "pending": "yellow",
    "running": "cyan",
    "done": "green",
    "failed": "bold red",
    "unknown": "magenta",
}


@dataclass(frozen=True)
class JobSnapshot:
    """One polled view of a submitted job."""

    index: int
    label: str
    summary_label: str
    state: JobState
    job_id: str
    pid: int | None
    log_file: str
    dependencies: tuple[str, ...]


def submit_and_watch_jobs(*, experiment: Any, executor: Any, workspace: Any) -> None:
    """Submit experiment tasks and monitor resulting jobs with the TUI."""
    tasks = set(experiment.tasks().values())
    # Keep submit-time runtime events visible (e.g. snapshot/submission messages),
    # but hide the runtime job board rows because Textual will render job states.
    with _runtime_job_board_suppressed():
        job_graph = executor.submit(tasks=tasks, workspace=workspace, blocking=False)
        with _runtime_events_suppressed():
            snapshots = watch_job_graph(job_graph=job_graph)
    _print_final_summary(snapshots)


def submit_and_stream_job_logs(
    *,
    tasks: set[Any],
    executor: Any,
    workspace: Any,
    poll_interval_s: float = 0.2,
    output: TextIO | None = None,
) -> list[JobSnapshot]:
    """Submit tasks without the TUI and stream appended job logs."""
    with _runtime_job_board_suppressed():
        job_graph = executor.submit(tasks=tasks, workspace=workspace, blocking=False)
        return stream_job_graph_logs(job_graph=job_graph, poll_interval_s=poll_interval_s, output=output)


def watch_job_graph(job_graph: DependencyGraph[Job], poll_interval_s: float = 0.2) -> list[JobSnapshot]:
    """Render the submitted job graph in a Textual app."""
    console = Console(stderr=True, soft_wrap=True)
    if not job_graph.nodes():
        console.print("[bold blue][misen][/bold blue] No work units were submitted.", style="dim")
        return []

    return _run_textual_job_monitor(job_graph=job_graph, poll_interval_s=poll_interval_s)


def stream_job_graph_logs(
    job_graph: DependencyGraph[Job],
    poll_interval_s: float = 0.2,
    output: TextIO | None = None,
) -> list[JobSnapshot]:
    """Poll the job graph and print newly appended job-log content."""
    console = Console(stderr=True, soft_wrap=True)
    if not job_graph.nodes():
        console.print("[bold blue][misen][/bold blue] No work units were submitted.", style="dim")
        return []

    output = sys.stdout if output is None else output
    offsets: dict[int, int] = {}

    while True:
        snapshots, _, done = snapshot_jobs(job_graph)
        for snapshot in snapshots:
            _drain_job_log(
                job_key=snapshot.index,
                job=job_graph[snapshot.index],
                offsets=offsets,
                output=output,
            )
        if done:
            return snapshots
        time.sleep(max(0.05, poll_interval_s))


def snapshot_jobs(
    job_graph: DependencyGraph[Job],
) -> tuple[list[JobSnapshot], dict[int, list[int]], bool]:
    """Collect one state snapshot for each job node."""
    node_indices = job_graph.node_indices()
    labels = {idx: _job_label(job_graph[idx]) for idx in node_indices}
    index_by_job = {job_graph[idx]: idx for idx in node_indices}

    dependency_indices = {
        idx: sorted((index_by_job[dep] for dep in job_graph.successors(idx)), key=lambda dep_idx: labels[dep_idx])
        for idx in node_indices
    }

    snapshots: list[JobSnapshot] = []
    done = True
    for idx in sorted(node_indices, key=lambda node_idx: labels[node_idx]):
        job = job_graph[idx]
        state = _safe_job_state(job)
        if state not in _TERMINAL_STATES:
            done = False
        snapshots.append(
            JobSnapshot(
                index=idx,
                label=labels[idx],
                summary_label=_job_summary_label(job),
                state=state,
                job_id=job.job_id or "-",
                pid=_job_pid(job),
                log_file=str(job.log_path) if job.log_path is not None else "-",
                dependencies=tuple(labels[dep_idx] for dep_idx in dependency_indices[idx]),
            )
        )
    return snapshots, dependency_indices, done


def _run_textual_job_monitor(job_graph: DependencyGraph[Job], poll_interval_s: float) -> list[JobSnapshot]:
    try:
        from textual.app import App, ComposeResult
        from textual.widgets import DataTable, Footer, Header, Static, Tree
    except ModuleNotFoundError as e:
        msg = (
            "Textual is required for the run TUI but is not installed. "
            "Install dependencies (for example: `uv sync`) and retry."
        )
        raise RuntimeError(msg) from e

    class _JobMonitorApp(App[None]):
        TITLE = "misen run"
        BINDINGS = (("q", "quit", "Quit"),)
        CSS = """
        Screen {
            layout: vertical;
        }

        #summary {
            height: auto;
            padding: 0 1;
        }

        #jobs {
            height: 1fr;
            min-height: 8;
        }

        #dependency-graph {
            height: 1fr;
            min-height: 8;
        }
        """

        def __init__(self) -> None:
            super().__init__()
            self.snapshots: list[JobSnapshot] = []
            self._exit_scheduled = False

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            yield Static(id="summary")
            yield DataTable(id="jobs")
            yield Tree("Dependency Graph", id="dependency-graph")
            yield Footer()

        def on_mount(self) -> None:
            table = self.query_one("#jobs", DataTable)
            table.cursor_type = "row"
            table.add_columns("Job", "State", "Job ID", "Log File", "Depends On")

            tree = self.query_one("#dependency-graph", Tree)
            tree.root.expand()

            self._refresh()
            self.set_interval(max(0.05, poll_interval_s), self._refresh)

        def _refresh(self) -> None:
            snapshots, dependency_indices, done = snapshot_jobs(job_graph)
            self.snapshots = snapshots
            self._render_summary(snapshots)
            self._render_jobs_table(snapshots)
            self._render_dependency_tree(snapshots=snapshots, dependency_indices=dependency_indices)
            if done and not self._exit_scheduled:
                self._exit_scheduled = True
                self.call_later(self.exit)

        def _render_summary(self, snapshots: list[JobSnapshot]) -> None:
            summary_widget = self.query_one("#summary", Static)
            summary_widget.update(_render_summary(snapshots))

        def _render_jobs_table(self, snapshots: list[JobSnapshot]) -> None:
            table = self.query_one("#jobs", DataTable)
            table.clear(columns=False)
            for snapshot in snapshots:
                table.add_row(
                    snapshot.label,
                    snapshot.state,
                    snapshot.job_id,
                    snapshot.log_file,
                    ", ".join(snapshot.dependencies) if snapshot.dependencies else "-",
                )

        def _render_dependency_tree(
            self,
            *,
            snapshots: list[JobSnapshot],
            dependency_indices: dict[int, list[int]],
        ) -> None:
            tree = self.query_one("#dependency-graph", Tree)
            root = tree.root
            root.label = Text("Dependency Graph")
            root.remove_children()

            snapshots_by_index = {snapshot.index: snapshot for snapshot in snapshots}
            roots = [idx for idx in job_graph.node_indices() if job_graph.is_root(idx)]
            for idx in sorted(roots, key=lambda root_idx: snapshots_by_index[root_idx].label):
                self._add_dependency_tree_node(
                    parent=root,
                    node_idx=idx,
                    snapshots_by_index=snapshots_by_index,
                    dependency_indices=dependency_indices,
                    stack=set(),
                )
            root.expand()

        def _add_dependency_tree_node(
            self,
            *,
            parent: Any,
            node_idx: int,
            snapshots_by_index: dict[int, JobSnapshot],
            dependency_indices: dict[int, list[int]],
            stack: set[int],
        ) -> None:
            snapshot = snapshots_by_index[node_idx]
            branch = parent.add(_tree_node_label(snapshot))
            branch.expand()

            next_stack = set(stack)
            next_stack.add(node_idx)
            for dep_idx in dependency_indices.get(node_idx, []):
                if dep_idx in stack:
                    dep_snapshot = snapshots_by_index[dep_idx]
                    cycle_text = _tree_node_label(dep_snapshot)
                    cycle_text.append(" (cycle)", style="bold red")
                    branch.add(cycle_text)
                    continue
                self._add_dependency_tree_node(
                    parent=branch,
                    node_idx=dep_idx,
                    snapshots_by_index=snapshots_by_index,
                    dependency_indices=dependency_indices,
                    stack=next_stack,
                )

    app = _JobMonitorApp()
    app.run()
    return app.snapshots


def _render_summary(snapshots: list[JobSnapshot]) -> Text:
    """Return the summary line shown above the jobs table."""
    counts = Counter(snapshot.state for snapshot in snapshots)
    summary = Text("Jobs: ")
    summary.append(str(len(snapshots)), style="bold")
    summary.append("  ")
    for state in ("pending", "running", "done", "failed", "unknown"):
        summary.append(f"{state}=", style="dim")
        summary.append(str(counts.get(state, 0)), style=_STATE_STYLES[state])
        summary.append("  ")
    return summary


def _tree_node_label(snapshot: JobSnapshot) -> Text:
    """Format a dependency tree label for one job snapshot."""
    text = Text(snapshot.label)
    text.append(" [")
    text.append(snapshot.state, style=_STATE_STYLES[snapshot.state])
    text.append("]")
    return text


def _job_label(job: Job) -> str:
    task = job.work_unit.root
    return f"{task.properties.id} ({task.task_hash().short_b32()})"


def _job_summary_label(job: Job) -> str:
    return task_label(job.work_unit.root, include_hash=True, include_arguments=True)


def _job_pid(job: Job) -> int | None:
    pid = getattr(job, "pid", None)
    if isinstance(pid, int):
        return pid

    process = getattr(job, "_process", None)
    if process is None:
        return None
    process_pid = getattr(process, "pid", None)
    if isinstance(process_pid, int):
        return process_pid
    return None


def _safe_job_state(job: Job) -> JobState:
    """Poll one job state, mapping backend query errors to ``unknown``."""
    try:
        state = job.state()
    except (FileNotFoundError, OSError, RuntimeError):
        return "unknown"
    return state if state in _STATE_STYLES else "unknown"


def _drain_job_log(*, job_key: int, job: Job, offsets: dict[int, int], output: TextIO) -> None:
    """Write any new bytes from one job log to the output stream."""
    if job.log_path is None:
        return

    offset = offsets.get(job_key, 0)
    try:
        with job.log_path.open("r", encoding="utf-8", errors="replace") as log_file:
            log_file.seek(offset)
            chunk = log_file.read()
            offsets[job_key] = log_file.tell()
    except FileNotFoundError:
        return

    if not chunk:
        return
    output.write(chunk)
    output.flush()


@contextlib.contextmanager
def _runtime_events_suppressed() -> Iterator[None]:
    """Disable runtime event widgets while the Textual app is running."""
    previous = os.environ.get("MISEN_RUNTIME_EVENTS")
    os.environ["MISEN_RUNTIME_EVENTS"] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("MISEN_RUNTIME_EVENTS", None)
        else:
            os.environ["MISEN_RUNTIME_EVENTS"] = previous


@contextlib.contextmanager
def _runtime_job_board_suppressed() -> Iterator[None]:
    """Disable runtime job-board rows while keeping regular runtime events."""
    previous = os.environ.get("MISEN_RUNTIME_JOB_BOARD")
    os.environ["MISEN_RUNTIME_JOB_BOARD"] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("MISEN_RUNTIME_JOB_BOARD", None)
        else:
            os.environ["MISEN_RUNTIME_JOB_BOARD"] = previous


def _print_final_summary(snapshots: list[JobSnapshot]) -> None:
    lines = runtime_job_summary_lines(
        [
            RuntimeJobSummary(
                label=snapshot.summary_label,
                state=snapshot.state,
                job_id=None if snapshot.job_id == "-" else snapshot.job_id,
                pid=snapshot.pid,
            )
            for snapshot in snapshots
        ]
    )
    if not lines:
        return

    console = Console(stderr=True, soft_wrap=True)
    for line in lines:
        console.print(line)
