from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import tyro

from misen import Experiment, Task, task
from misen.executor import Executor
from misen.executor import Job
import misen.utils.experiment_cli as experiment_cli_module
from misen.utils.graph import DependencyGraph
import misen.utils.tui as tui_module
from misen.utils.work_unit import WorkUnit
from misen.workspace import Workspace


@task(id="cli_task", cache=False)
def cli_task(x: int) -> int:
    return x


class CliExperiment(Experiment):
    value: int = 1

    def tasks(self) -> dict[str, Task[int]]:
        return {"task": Task(cli_task, x=self.value)}


def test_experiment_cli_count_command(monkeypatch, capsys, tmp_path) -> None:
    settings_file = tmp_path / "misen.toml"
    first_args = SimpleNamespace(
        command="count",
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
    )
    second_args = SimpleNamespace(
        command="count",
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
        experiment=CliExperiment(),
    )

    def fake_cli(*_args: object, **kwargs: object) -> object:
        if kwargs.get("return_unknown_args"):
            return first_args, []
        return second_args

    monkeypatch.setattr(tyro, "cli", fake_cli)
    monkeypatch.setattr(Executor, "auto", classmethod(lambda _cls, settings=None: object()))
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: object()))

    experiment_cli_module.experiment_cli(CliExperiment)

    assert capsys.readouterr().out.strip() == "1"


class FakeJob(Job):
    """Simple state-sequence job for TUI tests."""

    def __init__(
        self,
        *,
        work_unit: WorkUnit,
        states: list[Literal["pending", "running", "done", "failed", "unknown"]],
        job_id: str | None = None,
        log_path: Path | None = None,
    ) -> None:
        super().__init__(work_unit=work_unit, job_id=job_id, log_path=log_path)
        self._states = states
        self.state_calls = 0

    def state(self) -> Literal["pending", "running", "done", "failed", "unknown"]:
        idx = min(self.state_calls, len(self._states) - 1)
        self.state_calls += 1
        return self._states[idx]


class FakeExecutor:
    def __init__(self, graph: DependencyGraph[Job]) -> None:
        self.graph = graph
        self.submitted_tasks: set[Task] | None = None
        self.submitted_workspace: object | None = None

    def submit(self, tasks: set[Task], workspace: object) -> DependencyGraph[Job]:
        self.submitted_tasks = tasks
        self.submitted_workspace = workspace
        return self.graph


def test_experiment_cli_run_command_uses_submit_graph(monkeypatch, tmp_path) -> None:
    settings_file = tmp_path / "misen.toml"
    first_args = SimpleNamespace(
        command="run",
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
    )
    second_args = SimpleNamespace(
        command="run",
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
        experiment=CliExperiment(),
    )

    def fake_cli(*_args: object, **kwargs: object) -> object:
        if kwargs.get("return_unknown_args"):
            return first_args, []
        return second_args

    graph: DependencyGraph[Job] = DependencyGraph()
    work_unit = WorkUnit(root=Task(cli_task, x=1), dependencies=set())
    graph.add_node(FakeJob(work_unit=work_unit, states=["done"]))

    fake_executor = FakeExecutor(graph=graph)
    fake_workspace = object()
    watched_graph: dict[str, DependencyGraph[Job]] = {}

    def fake_watch(job_graph: DependencyGraph[Job], poll_interval_s: float = 0.2) -> list[object]:
        _ = poll_interval_s
        watched_graph["graph"] = job_graph
        return []

    monkeypatch.setattr(tyro, "cli", fake_cli)
    monkeypatch.setattr(Executor, "auto", classmethod(lambda _cls, settings=None: fake_executor))
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: fake_workspace))
    monkeypatch.setattr(tui_module, "watch_job_graph", fake_watch)

    experiment_cli_module.experiment_cli(CliExperiment)

    assert watched_graph["graph"] is graph
    assert fake_executor.submitted_workspace is fake_workspace
    assert fake_executor.submitted_tasks is not None
    assert len(fake_executor.submitted_tasks) == 1


def test_snapshot_jobs_captures_dependencies_and_log_path() -> None:
    dep_wu = WorkUnit(root=Task(cli_task, x=1), dependencies=set())
    parent_wu = WorkUnit(root=Task(cli_task, x=2), dependencies={dep_wu})
    dep_job = FakeJob(work_unit=dep_wu, states=["running"], job_id="dep-1", log_path=Path("/tmp/dep.log"))
    parent_job = FakeJob(work_unit=parent_wu, states=["pending"], job_id=None, log_path=None)

    graph: DependencyGraph[Job] = DependencyGraph()
    parent_idx = graph.add_node(parent_job)
    dep_idx = graph.add_node(dep_job)
    graph.add_edge(parent_idx, dep_idx, None)

    snapshots, dependency_indices, done = tui_module.snapshot_jobs(graph)
    snapshots_by_index = {snapshot.index: snapshot for snapshot in snapshots}

    assert done is False
    assert dependency_indices[parent_idx] == [dep_idx]
    assert snapshots_by_index[parent_idx].dependencies == (snapshots_by_index[dep_idx].label,)
    assert snapshots_by_index[dep_idx].log_file == "/tmp/dep.log"


def test_watch_job_graph_uses_textual_runner(monkeypatch) -> None:
    graph: DependencyGraph[Job] = DependencyGraph()
    work_unit = WorkUnit(root=Task(cli_task, x=3), dependencies=set())
    job = FakeJob(work_unit=work_unit, states=["done"])
    graph.add_node(job)

    seen: dict[str, object] = {}

    def fake_run_textual(job_graph: DependencyGraph[Job], poll_interval_s: float) -> list[tui_module.JobSnapshot]:
        seen["job_graph"] = job_graph
        seen["poll_interval_s"] = poll_interval_s
        snapshots, _, _ = tui_module.snapshot_jobs(job_graph)
        return snapshots

    monkeypatch.setattr(tui_module, "_run_textual_job_monitor", fake_run_textual)

    snapshots = tui_module.watch_job_graph(graph, poll_interval_s=0.0)

    assert snapshots
    assert snapshots[0].state == "done"
    assert seen["job_graph"] is graph
    assert seen["poll_interval_s"] == 0.0
