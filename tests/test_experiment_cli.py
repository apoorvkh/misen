import sys
from types import SimpleNamespace

import pytest
import tyro

from misen import Experiment, Task, task
from misen.executor import Executor
from misen.utils.experiment_cli import _resolve_command, experiment_cli
from misen.utils.runtime_events import task_label, work_unit_label
from misen.utils.work_unit import WorkUnit
from misen.workspace import Workspace


@task(id="source", cache=False)
def source(x: int) -> int:
    return x


@task(id="sink", cache=False)
def sink(x: int) -> int:
    return x


@task(id="left", cache=False)
def left(x: int) -> int:
    return x


@task(id="right", cache=False)
def right(x: int) -> int:
    return x


@task(id="with_exclude", cache=False, exclude={"hidden"})
def with_exclude(hidden: int, visible: int) -> int:
    return hidden + visible


@task(id="cached_only", cache=True)
def cached_only(x: int) -> int:
    return x


class CliExperiment(Experiment):
    value: int = 1

    def tasks(self) -> dict[str, Task[int]]:
        upstream = Task(source, x=self.value)
        return {"task": Task(sink, x=upstream.T)}


class SharedDepsExperiment(Experiment):
    value: int = 1

    def tasks(self) -> dict[str, Task[int]]:
        shared = Task(source, x=self.value)
        return {"left": Task(left, x=shared.T), "right": Task(right, x=shared.T)}


class CachedExperiment(Experiment):
    value: int = 1

    def tasks(self) -> dict[str, Task[int]]:
        return {"cached": Task(cached_only, x=self.value)}


class _StatusWorkspace:
    def __init__(self, *, done_ids: set[str]) -> None:
        self.done_ids = done_ids

    def get_result_hash(self, task: Task[int]) -> object:
        if task.properties.id in self.done_ids:
            return object()
        msg = "Task not complete"
        raise RuntimeError(msg)


def _mock_cli(monkeypatch: pytest.MonkeyPatch, first_args: SimpleNamespace, second_args: SimpleNamespace) -> None:
    def fake_cli(*_args: object, **kwargs: object) -> object:
        if kwargs.get("return_unknown_args"):
            return first_args, []
        return second_args

    monkeypatch.setattr(tyro, "cli", fake_cli)


def test_resolve_command_uses_explicit_token() -> None:
    assert _resolve_command(command_token="count", unknown_args=["run"]) == "count"


def test_resolve_command_scans_unknown_args_for_positional_command() -> None:
    assert _resolve_command(command_token=None, unknown_args=["--value", "2", "tree"]) == "tree"


def test_resolve_command_defaults_to_run() -> None:
    assert _resolve_command(command_token=None, unknown_args=["--value", "2"]) == "run"


def test_resolve_command_rejects_invalid_command() -> None:
    with pytest.raises(ValueError, match=r"Unknown command"):
        _resolve_command(command_token="bad", unknown_args=[])


def test_resolve_command_accepts_incomplete_alias() -> None:
    assert _resolve_command(command_token="incomplete", unknown_args=[]) == "incomplete"


def test_task_label_includes_arguments_without_dependent_tasks() -> None:
    upstream = Task(source, x=7)
    downstream = Task(sink, x=upstream.T)

    upstream_label = task_label(upstream, include_arguments=True)
    downstream_label = task_label(downstream, include_arguments=True)

    assert "x=7" in upstream_label
    assert upstream_label.startswith("source(")
    assert "]" in upstream_label
    assert "x=" not in downstream_label


def test_task_repr_includes_arguments_without_dependent_tasks() -> None:
    upstream = Task(source, x=7)
    downstream = Task(sink, x=upstream.T)

    upstream_repr = repr(upstream)
    downstream_repr = repr(downstream)

    assert "x=7" in upstream_repr
    assert "x=" not in downstream_repr
    assert upstream_repr == f"Task({task_label(upstream, include_arguments=True)})"
    assert downstream_repr == f"Task({task_label(downstream, include_arguments=True)})"


def test_task_label_excludes_arguments_configured_in_properties() -> None:
    label = task_label(Task(with_exclude, hidden=99, visible=1), include_arguments=True)
    assert "hidden=" not in label
    assert "visible=1" in label


def test_work_unit_repr_uses_work_unit_label() -> None:
    work_unit = WorkUnit(root=Task(source, x=7), dependencies=set())
    assert repr(work_unit) == f"WorkUnit({work_unit_label(work_unit)})"


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

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids={"source"})))

    experiment_cli(CliExperiment)

    assert capsys.readouterr().out.strip() == "Completed 1 of 2 tasks."


def test_experiment_cli_tree_command_prints_tree(monkeypatch, capsys, tmp_path) -> None:
    settings_file = tmp_path / "misen.toml"
    first_args = SimpleNamespace(
        command="tree",
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
    )
    second_args = SimpleNamespace(
        command="tree",
        tree_all=False,
        tree_max_depth=None,
        tree_cacheable_only=False,
        tree_incomplete=False,
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
        experiment=CliExperiment(),
    )

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids={"source"})))

    experiment_cli(CliExperiment)

    output = capsys.readouterr().out
    assert "Tasks" in output
    assert "task:" in output
    assert "source" in output
    assert "sink" in output
    assert "source(x=1)" in output
    assert "[NC]" not in output
    assert "✓" in output
    assert "○" in output


def test_experiment_cli_tree_command_max_depth(monkeypatch, capsys, tmp_path) -> None:
    settings_file = tmp_path / "misen.toml"
    first_args = SimpleNamespace(
        command="tree",
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
    )
    second_args = SimpleNamespace(
        command="tree",
        tree_all=False,
        tree_max_depth=0,
        tree_cacheable_only=False,
        tree_incomplete=False,
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
        experiment=CliExperiment(),
    )

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids={"source"})))

    experiment_cli(CliExperiment)

    output = capsys.readouterr().out
    assert "task:" in output
    assert "sink" in output
    assert "source" not in output


def test_experiment_cli_tree_command_cacheable_only(monkeypatch, capsys, tmp_path) -> None:
    settings_file = tmp_path / "misen.toml"
    first_args = SimpleNamespace(
        command="tree",
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
    )
    second_args = SimpleNamespace(
        command="tree",
        tree_all=False,
        tree_max_depth=None,
        tree_cacheable_only=True,
        tree_incomplete=False,
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
        experiment=CliExperiment(),
    )

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids={"source"})))

    experiment_cli(CliExperiment)

    assert "No tasks matched filters." in capsys.readouterr().out


def test_experiment_cli_tree_command_all_expands_shared_dependencies(monkeypatch, capsys, tmp_path) -> None:
    settings_file = tmp_path / "misen.toml"
    first_args = SimpleNamespace(
        command="tree",
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
    )
    second_args = SimpleNamespace(
        command="tree",
        tree_all=True,
        tree_max_depth=None,
        tree_cacheable_only=False,
        tree_incomplete=False,
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
        experiment=SharedDepsExperiment(),
    )

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids={"source"})))

    experiment_cli(SharedDepsExperiment)

    output = capsys.readouterr().out
    assert "(*)" not in output
    assert "Dependencies already listed." not in output
    assert output.count("source") == 2


def test_experiment_cli_tree_command_collapses_shared_dependencies_by_default(monkeypatch, capsys, tmp_path) -> None:
    settings_file = tmp_path / "misen.toml"
    first_args = SimpleNamespace(
        command="tree",
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
    )
    second_args = SimpleNamespace(
        command="tree",
        tree_all=False,
        tree_max_depth=None,
        tree_cacheable_only=False,
        tree_incomplete=False,
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
        experiment=SharedDepsExperiment(),
    )

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids={"source"})))

    experiment_cli(SharedDepsExperiment)

    output = capsys.readouterr().out
    assert "(*)" in output
    assert "Dependencies already listed." in output
    assert output.count("source") == 1


def test_experiment_cli_list_command_prints_flat_tasks(monkeypatch, capsys, tmp_path) -> None:
    settings_file = tmp_path / "misen.toml"
    first_args = SimpleNamespace(
        command="list",
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
    )
    second_args = SimpleNamespace(
        command="list",
        list_cacheable_only=False,
        list_incomplete=False,
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
        experiment=CliExperiment(),
    )

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids={"source"})))

    experiment_cli(CliExperiment)

    output = capsys.readouterr().out
    assert "Tasks" in output
    assert "Status" not in output
    assert "Cache" not in output
    assert "source" in output
    assert "sink" in output
    assert "source(x=1)" in output
    assert "[NC]" not in output
    assert "○" in output or "✓" in output
    assert "┏" not in output
    assert "task:" not in output


def test_experiment_cli_list_command_places_cache_marker_after_task_name(monkeypatch, capsys, tmp_path) -> None:
    settings_file = tmp_path / "misen.toml"
    first_args = SimpleNamespace(
        command="list",
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
    )
    second_args = SimpleNamespace(
        command="list",
        list_cacheable_only=False,
        list_incomplete=False,
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
        experiment=CachedExperiment(),
    )

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids=set())))

    experiment_cli(CachedExperiment)

    output = capsys.readouterr().out
    assert "cached_only(x=1) [C] [" in output
    assert "[NC]" not in output


def test_experiment_cli_list_command_cacheable_only(monkeypatch, capsys, tmp_path) -> None:
    settings_file = tmp_path / "misen.toml"
    first_args = SimpleNamespace(
        command="list",
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
    )
    second_args = SimpleNamespace(
        command="list",
        list_cacheable_only=True,
        list_incomplete=False,
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
        experiment=CliExperiment(),
    )

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids={"source"})))

    experiment_cli(CliExperiment)

    assert "No tasks matched filters." in capsys.readouterr().out


def test_experiment_cli_list_command_incomplete_filter(monkeypatch, capsys, tmp_path) -> None:
    settings_file = tmp_path / "misen.toml"
    first_args = SimpleNamespace(
        command="list",
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
    )
    second_args = SimpleNamespace(
        command="list",
        list_cacheable_only=False,
        list_incomplete=True,
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
        experiment=CliExperiment(),
    )

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids={"source"})))

    experiment_cli(CliExperiment)

    output = capsys.readouterr().out
    assert "Incomplete Tasks" in output
    assert "sink" in output
    assert "source" not in output


def test_experiment_cli_tree_command_incomplete_filter_prints_missing_tasks(monkeypatch, capsys, tmp_path) -> None:
    settings_file = tmp_path / "misen.toml"
    first_args = SimpleNamespace(
        command="tree",
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
    )
    second_args = SimpleNamespace(
        command="tree",
        tree_all=False,
        tree_max_depth=None,
        tree_cacheable_only=False,
        tree_incomplete=True,
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
        experiment=CliExperiment(),
    )

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids={"source"})))

    experiment_cli(CliExperiment)

    output = capsys.readouterr().out
    assert "Incomplete" in output
    assert "Tasks" in output
    assert "sink" in output
    assert "source" not in output


def test_experiment_cli_incomplete_command_alias_prints_missing_tasks(monkeypatch, capsys, tmp_path) -> None:
    settings_file = tmp_path / "misen.toml"
    first_args = SimpleNamespace(
        command="incomplete",
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
    )
    second_args = SimpleNamespace(
        command="incomplete",
        tree_all=False,
        tree_max_depth=None,
        tree_cacheable_only=False,
        tree_incomplete=True,
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
        experiment=CliExperiment(),
    )

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids={"source"})))

    experiment_cli(CliExperiment)

    output = capsys.readouterr().out
    assert "Incomplete Tasks" in output
    assert "sink" in output
    assert "source" not in output


def test_experiment_cli_tree_command_incomplete_filter_handles_fully_complete_graph(
    monkeypatch, capsys, tmp_path
) -> None:
    settings_file = tmp_path / "misen.toml"
    first_args = SimpleNamespace(
        command="tree",
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
    )
    second_args = SimpleNamespace(
        command="tree",
        tree_all=False,
        tree_max_depth=None,
        tree_cacheable_only=False,
        tree_incomplete=True,
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
        experiment=CliExperiment(),
    )

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(
        Workspace,
        "auto",
        classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids={"source", "sink"})),
    )

    experiment_cli(CliExperiment)

    assert "No incomplete tasks." in capsys.readouterr().out


def test_experiment_cli_run_command_runs_full_experiment(monkeypatch, tmp_path) -> None:
    settings_file = tmp_path / "misen.toml"
    executor = object()
    workspace = object()
    calls: dict[str, object] = {}

    def fake_run(*, executor: object, workspace: object) -> None:
        calls["executor"] = executor
        calls["workspace"] = workspace

    first_args = SimpleNamespace(
        command="run",
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
    )
    second_args = SimpleNamespace(
        command="run",
        run_task=None,
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
        experiment=SimpleNamespace(run=fake_run),
    )

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(Executor, "auto", classmethod(lambda _cls, settings=None: executor))
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: workspace))

    experiment_cli(CliExperiment)

    assert calls == {"executor": executor, "workspace": workspace}


def test_experiment_cli_run_command_with_task_name(monkeypatch, tmp_path) -> None:
    settings_file = tmp_path / "misen.toml"
    executor = object()
    workspace = object()
    submit_calls: list[tuple[object, object]] = []

    class StubTask:
        def submit(self, *, executor: object, workspace: object) -> None:
            submit_calls.append((executor, workspace))

    class StubExperiment:
        def __init__(self, task: StubTask) -> None:
            self._task = task

        def __getitem__(self, key: str) -> StubTask:
            assert key == "task"
            return self._task

    first_args = SimpleNamespace(
        command="run",
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
    )
    second_args = SimpleNamespace(
        command="run",
        run_task="task",
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
        experiment=StubExperiment(task=StubTask()),
    )

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(Executor, "auto", classmethod(lambda _cls, settings=None: executor))
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: workspace))

    experiment_cli(CliExperiment)

    assert submit_calls == [(executor, workspace)]


def test_experiment_cli_parses_positional_run_command(monkeypatch) -> None:
    submit_calls: list[tuple[set[object], object]] = []
    workspace = object()

    class StubExecutor:
        def submit(self, *, tasks: set[object], workspace: object) -> None:
            submit_calls.append((tasks, workspace))

    monkeypatch.setattr(Executor, "auto", classmethod(lambda _cls, settings=None: StubExecutor()))
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: workspace))
    monkeypatch.setattr(sys, "argv", ["prog", "run", "task"])

    experiment_cli(CliExperiment)

    assert len(submit_calls) == 1
    assert len(submit_calls[0][0]) == 1
    assert submit_calls[0][1] is workspace


def test_experiment_cli_parses_tree_short_depth_flag(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "tree", "-L", "0", "--cacheable-only"])
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids=set())))

    experiment_cli(CliExperiment)

    assert "No tasks matched filters." in capsys.readouterr().out


def test_experiment_cli_parses_tree_incomplete_flag(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "tree", "--incomplete"])
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids={"source"})))

    experiment_cli(CliExperiment)

    output = capsys.readouterr().out
    assert "Incomplete Tasks" in output
    assert "sink" in output
    assert "source" not in output


def test_experiment_cli_parses_positional_incomplete_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "incomplete"])
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids={"source"})))

    experiment_cli(CliExperiment)

    output = capsys.readouterr().out
    assert "Incomplete Tasks" in output
    assert "sink" in output
    assert "source" not in output


def test_experiment_cli_parses_list_incomplete_flag(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "list", "--incomplete"])
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids={"source"})))

    experiment_cli(CliExperiment)

    output = capsys.readouterr().out
    assert "Incomplete Tasks" in output
    assert "sink" in output
    assert "source" not in output


def test_experiment_cli_result_command(monkeypatch, capsys, tmp_path) -> None:
    settings_file = tmp_path / "misen.toml"
    workspace = object()

    def fake_result(key: str, workspace: object) -> str:
        assert key == "task"
        assert workspace is workspace_obj
        return "value"

    workspace_obj = workspace
    experiment = SimpleNamespace(result=fake_result)
    first_args = SimpleNamespace(
        command="result",
        result_task=None,
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
    )
    second_args = SimpleNamespace(
        command="result",
        result_task="task",
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
        experiment=experiment,
    )

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: workspace_obj))

    experiment_cli(CliExperiment)

    assert "value" in capsys.readouterr().out


def test_experiment_cli_result_command_requires_task_selector(monkeypatch, tmp_path) -> None:
    settings_file = tmp_path / "misen.toml"
    first_args = SimpleNamespace(
        command="result",
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
    )
    second_args = SimpleNamespace(
        command="result",
        result_task=None,
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
        experiment=SimpleNamespace(result=lambda _key, workspace: workspace),
    )

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: object()))

    with pytest.raises(ValueError, match=r"requires a task name"):
        experiment_cli(CliExperiment)
