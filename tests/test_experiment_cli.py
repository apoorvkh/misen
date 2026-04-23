import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import pytest
import tyro

import misen.cli as misen_cli
import misen.utils.cli.experiment as experiment_module
import misen.utils.cli.tui as tui_module
from misen import CacheError, Experiment, Task, meta
from misen.executor import Executor, Job
from misen.utils.cli.experiment import _resolve_command, experiment, experiment_cli
from misen.utils.graph import DependencyGraph
from misen.utils.runtime_events import RuntimeJobSummary, runtime_job_summary_lines, task_label, work_unit_label
from misen.utils.work_unit import WorkUnit
from misen.workspace import Workspace


@meta(id="source", cache=False)
def source(x: int) -> int:
    return x


@meta(id="sink", cache=False)
def sink(x: int) -> int:
    return x


@meta(id="left", cache=False)
def left(x: int) -> int:
    return x


@meta(id="right", cache=False)
def right(x: int) -> int:
    return x


@meta(id="with_exclude", cache=False, exclude={"hidden"})
def with_exclude(hidden: int, visible: int) -> int:
    return hidden + visible


@meta(id="cached_only", cache=True)
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


def test_experiment_command_resolves_reference_and_forwards_argv(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_experiment_cli(experiment_cls: type[object], argv: list[str] | tuple[str, ...] | None = None) -> None:
        seen["experiment_cls"] = experiment_cls
        seen["argv"] = argv

    monkeypatch.setattr(experiment_module, "experiment_cli", fake_experiment_cli)
    exit_code = experiment(["tests.test_experiment_cli:CliExperiment", "tree", "--max-depth", "1"])

    assert exit_code == 0
    assert seen["experiment_cls"] is CliExperiment
    assert seen["argv"] == ["tree", "--max-depth", "1"]


def test_experiment_command_accepts_src_file_reference(monkeypatch, tmp_path) -> None:
    project_src = tmp_path / "src" / "demo_pkg"
    project_src.mkdir(parents=True)
    (project_src / "__init__.py").write_text("", encoding="utf-8")
    (project_src / "demo.py").write_text(
        (
            "from misen import Experiment, Task, meta\n\n"
            "@meta(id='demo_task', cache=False)\n"
            "def demo_task(x: int) -> int:\n"
            "    return x\n\n"
            "class DemoExperiment(Experiment):\n"
            "    marker = 'local'\n\n"
            "    def tasks(self):\n"
            "        return {'task': Task(demo_task, x=1)}\n"
        ),
        encoding="utf-8",
    )

    seen: dict[str, object] = {}

    def fake_experiment_cli(experiment_cls: type[object], argv: list[str] | tuple[str, ...] | None = None) -> None:
        seen["experiment_cls"] = experiment_cls
        seen["argv"] = argv

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment_module, "experiment_cli", fake_experiment_cli)
    sys.modules.pop("demo_pkg", None)
    sys.modules.pop("demo_pkg.demo", None)

    exit_code = experiment(["src/demo_pkg/demo.py:DemoExperiment", "list"])

    assert exit_code == 0
    assert getattr(cast("object", seen["experiment_cls"]), "marker", None) == "local"
    assert seen["argv"] == ["list"]


def test_experiment_command_reports_invalid_reference(capsys) -> None:
    exit_code = experiment(["bad-reference"])

    assert exit_code == 2
    assert "Invalid experiment reference." in capsys.readouterr().err


def test_resolve_experiment_class_prefers_local_src_module(monkeypatch, tmp_path) -> None:
    project_src = tmp_path / "src" / "demo_pkg"
    project_src.mkdir(parents=True)
    (project_src / "__init__.py").write_text("", encoding="utf-8")
    (project_src / "demo.py").write_text(
        (
            "from misen import Experiment, Task, meta\n\n"
            "@meta(id='demo_task', cache=False)\n"
            "def demo_task(x: int) -> int:\n"
            "    return x\n\n"
            "class DemoExperiment(Experiment):\n"
            "    marker = 'local'\n\n"
            "    def tasks(self):\n"
            "        return {'task': Task(demo_task, x=1)}\n"
        ),
        encoding="utf-8",
    )

    installed_pkg = tmp_path / "site" / "demo_pkg"
    installed_pkg.mkdir(parents=True)
    (installed_pkg / "__init__.py").write_text("", encoding="utf-8")
    (installed_pkg / "demo.py").write_text(
        (
            "from misen import Experiment, Task, meta\n\n"
            "@meta(id='demo_task', cache=False)\n"
            "def demo_task(x: int) -> int:\n"
            "    return x\n\n"
            "class DemoExperiment(Experiment):\n"
            "    marker = 'site'\n\n"
            "    def tasks(self):\n"
            "        return {'task': Task(demo_task, x=2)}\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("demo_pkg", None)
    sys.modules.pop("demo_pkg.demo", None)

    resolved = experiment_module.resolve_experiment_class("demo_pkg.demo:DemoExperiment")

    assert getattr(resolved, "marker") == "local"
    assert Path(sys.modules["demo_pkg.demo"].__file__).resolve() == (project_src / "demo.py").resolve()


def test_resolve_experiment_class_accepts_src_file_reference(monkeypatch, tmp_path) -> None:
    project_src = tmp_path / "src" / "demo_pkg"
    project_src.mkdir(parents=True)
    (project_src / "__init__.py").write_text("", encoding="utf-8")
    (project_src / "demo.py").write_text(
        (
            "from misen import Experiment, Task, meta\n\n"
            "@meta(id='demo_task', cache=False)\n"
            "def demo_task(x: int) -> int:\n"
            "    return x\n\n"
            "class DemoExperiment(Experiment):\n"
            "    marker = 'local'\n\n"
            "    def tasks(self):\n"
            "        return {'task': Task(demo_task, x=1)}\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    sys.modules.pop("demo_pkg", None)
    sys.modules.pop("demo_pkg.demo", None)

    resolved = experiment_module.resolve_experiment_class("src/demo_pkg/demo.py:DemoExperiment")

    assert getattr(resolved, "marker") == "local"
    assert Path(sys.modules["demo_pkg.demo"].__file__).resolve() == (project_src / "demo.py").resolve()


def test_resolve_experiment_class_registers_module_for_value_pickling(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_register(module: object) -> None:
        seen["module"] = module

    monkeypatch.setattr(experiment_module, "_register_module_pickle_by_value", fake_register)
    resolved = experiment_module.resolve_experiment_class("tests.test_experiment_cli:CliExperiment")

    assert resolved is CliExperiment
    assert getattr(cast("object", seen["module"]), "__name__", None) == "tests.test_experiment_cli"


@pytest.fixture
def captured_args(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    captured: dict[str, object] = {}

    def fake_execute(*, args: object, console: object) -> None:  # noqa: ARG001
        captured["args"] = args

    monkeypatch.setattr(experiment_module, "_execute_command", fake_execute)
    return captured


def test_experiment_cli_instance_seeds_defaults_from_bound_fields(captured_args) -> None:
    experiment_cli(CliExperiment(value=42), argv=["count"])
    exp = cast("CliExperiment", getattr(captured_args["args"], "experiment"))
    assert isinstance(exp, CliExperiment)
    assert exp.value == 42


def test_experiment_cli_instance_defaults_still_overridable_by_flags(captured_args) -> None:
    experiment_cli(CliExperiment(value=42), argv=["--value", "99", "count"])
    exp = cast("CliExperiment", getattr(captured_args["args"], "experiment"))
    assert exp.value == 99


def test_resolve_experiment_reference_accepts_instance(monkeypatch, tmp_path) -> None:
    project_src = tmp_path / "src" / "config_pkg"
    project_src.mkdir(parents=True)
    (project_src / "__init__.py").write_text("", encoding="utf-8")
    (project_src / "config.py").write_text(
        (
            "from misen import Experiment, Task, meta\n\n"
            "@meta(id='inst_task', cache=False)\n"
            "def inst_task(x: int) -> int:\n"
            "    return x\n\n"
            "class MyExp(Experiment):\n"
            "    value: int = 1\n\n"
            "    def tasks(self):\n"
            "        return {'t': Task(inst_task, x=self.value)}\n\n"
            "__config__ = MyExp(value=7)\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    sys.modules.pop("config_pkg", None)
    sys.modules.pop("config_pkg.config", None)

    resolved = experiment_module.resolve_experiment_reference("config_pkg.config:__config__")

    assert not isinstance(resolved, type)
    assert getattr(resolved, "value") == 7


def test_misen_main_dispatches_experiment(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_experiment(*, argv: list[str] | None = None) -> int:
        seen["argv"] = argv
        return 7

    monkeypatch.setattr(misen_cli, "experiment", fake_experiment)

    assert misen_cli.main(["experiment", "tests.test_experiment_cli:CliExperiment", "list"]) == 7
    assert seen["argv"] == ["tests.test_experiment_cli:CliExperiment", "list"]


class _StatusWorkspace:
    def __init__(self, *, done_ids: set[str]) -> None:
        self.done_ids = done_ids

    def get_result_hash(self, task: Task[int]) -> object:
        if task.meta.id in self.done_ids:
            return object()
        msg = "Task not complete"
        raise CacheError(msg)


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


def test_task_repr_uses_function_qualname_and_bound_arguments() -> None:
    upstream = Task(source, x=7)
    downstream = Task(sink, x=upstream.T)

    upstream_repr = repr(upstream)
    downstream_repr = repr(downstream)

    assert upstream_repr == f"Task({source.__module__}.{source.__qualname__}, x=7)"
    assert downstream_repr == f"Task({sink.__module__}.{sink.__qualname__}, x={upstream_repr})"


def test_task_label_excludes_arguments_configured_in_meta() -> None:
    label = task_label(Task(with_exclude, hidden=99, visible=1), include_arguments=True)
    assert "hidden=" not in label
    assert "visible=1" in label


def test_work_unit_repr_uses_work_unit_label() -> None:
    work_unit = WorkUnit(root=Task(source, x=7), dependencies=set())
    assert repr(work_unit) == f"WorkUnit({work_unit_label(work_unit)})"


def test_experiment_cli_count_command(monkeypatch, capsys, tmp_path) -> None:
    config_file = tmp_path / ".misen.toml"
    first_args = SimpleNamespace(
        command="count",
        config=config_file,
        executor="auto",
        workspace="auto",
    )
    second_args = SimpleNamespace(
        command="count",
        config=config_file,
        executor="auto",
        workspace="auto",
        experiment=CliExperiment(),
    )

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(
        Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids={"source"}))
    )

    experiment_cli(CliExperiment)

    assert capsys.readouterr().out.strip() == "Completed 1 of 2 tasks."


def test_experiment_cli_tree_command_prints_tree(monkeypatch, capsys, tmp_path) -> None:
    config_file = tmp_path / ".misen.toml"
    first_args = SimpleNamespace(
        command="tree",
        config=config_file,
        executor="auto",
        workspace="auto",
    )
    second_args = SimpleNamespace(
        command="tree",
        tree_all=False,
        tree_max_depth=None,
        tree_cacheable_only=False,
        tree_incomplete=False,
        config=config_file,
        executor="auto",
        workspace="auto",
        experiment=CliExperiment(),
    )

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(
        Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids={"source"}))
    )

    experiment_cli(CliExperiment)

    output = capsys.readouterr().out
    assert "Tasks" in output
    assert "task:" not in output
    assert "source" in output
    assert "sink" in output
    assert "source(x=1)" in output
    assert "[NC]" not in output
    assert "✓" in output
    assert "○" in output


def test_experiment_cli_tree_command_max_depth(monkeypatch, capsys, tmp_path) -> None:
    config_file = tmp_path / ".misen.toml"
    first_args = SimpleNamespace(
        command="tree",
        config=config_file,
        executor="auto",
        workspace="auto",
    )
    second_args = SimpleNamespace(
        command="tree",
        tree_all=False,
        tree_max_depth=0,
        tree_cacheable_only=False,
        tree_incomplete=False,
        config=config_file,
        executor="auto",
        workspace="auto",
        experiment=CliExperiment(),
    )

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(
        Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids={"source"}))
    )

    experiment_cli(CliExperiment)

    output = capsys.readouterr().out
    assert "task:" not in output
    assert "sink" in output
    assert "source" not in output


def test_experiment_cli_tree_command_cacheable_only(monkeypatch, capsys, tmp_path) -> None:
    config_file = tmp_path / ".misen.toml"
    first_args = SimpleNamespace(
        command="tree",
        config=config_file,
        executor="auto",
        workspace="auto",
    )
    second_args = SimpleNamespace(
        command="tree",
        tree_all=False,
        tree_max_depth=None,
        tree_cacheable_only=True,
        tree_incomplete=False,
        config=config_file,
        executor="auto",
        workspace="auto",
        experiment=CliExperiment(),
    )

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(
        Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids={"source"}))
    )

    experiment_cli(CliExperiment)

    assert "No tasks matched filters." in capsys.readouterr().out


def test_experiment_cli_tree_command_all_expands_shared_dependencies(monkeypatch, capsys, tmp_path) -> None:
    config_file = tmp_path / ".misen.toml"
    first_args = SimpleNamespace(
        command="tree",
        config=config_file,
        executor="auto",
        workspace="auto",
    )
    second_args = SimpleNamespace(
        command="tree",
        tree_all=True,
        tree_max_depth=None,
        tree_cacheable_only=False,
        tree_incomplete=False,
        config=config_file,
        executor="auto",
        workspace="auto",
        experiment=SharedDepsExperiment(),
    )

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(
        Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids={"source"}))
    )

    experiment_cli(SharedDepsExperiment)

    output = capsys.readouterr().out
    assert "(*)" not in output
    assert "Dependencies already listed." not in output
    assert output.count("source") == 2


def test_experiment_cli_tree_command_collapses_shared_dependencies_by_default(monkeypatch, capsys, tmp_path) -> None:
    config_file = tmp_path / ".misen.toml"
    first_args = SimpleNamespace(
        command="tree",
        config=config_file,
        executor="auto",
        workspace="auto",
    )
    second_args = SimpleNamespace(
        command="tree",
        tree_all=False,
        tree_max_depth=None,
        tree_cacheable_only=False,
        tree_incomplete=False,
        config=config_file,
        executor="auto",
        workspace="auto",
        experiment=SharedDepsExperiment(),
    )

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(
        Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids={"source"}))
    )

    experiment_cli(SharedDepsExperiment)

    output = capsys.readouterr().out
    assert "(*)" not in output
    assert "Dependencies already listed." not in output
    assert output.count("source") == 1


def test_experiment_cli_list_command_prints_flat_tasks(monkeypatch, capsys, tmp_path) -> None:
    config_file = tmp_path / ".misen.toml"
    first_args = SimpleNamespace(
        command="list",
        config=config_file,
        executor="auto",
        workspace="auto",
    )
    second_args = SimpleNamespace(
        command="list",
        list_cacheable_only=False,
        list_incomplete=False,
        config=config_file,
        executor="auto",
        workspace="auto",
        experiment=CliExperiment(),
    )

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(
        Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids={"source"}))
    )

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


def test_experiment_cli_list_command_omits_cache_marker(monkeypatch, capsys, tmp_path) -> None:
    config_file = tmp_path / ".misen.toml"
    first_args = SimpleNamespace(
        command="list",
        config=config_file,
        executor="auto",
        workspace="auto",
    )
    second_args = SimpleNamespace(
        command="list",
        list_cacheable_only=False,
        list_incomplete=False,
        config=config_file,
        executor="auto",
        workspace="auto",
        experiment=CachedExperiment(),
    )

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids=set())))

    experiment_cli(CachedExperiment)

    output = capsys.readouterr().out
    assert "cached_only(x=1)" in output
    assert "[C]" not in output
    assert "[NC]" not in output


def test_experiment_cli_list_command_cacheable_only(monkeypatch, capsys, tmp_path) -> None:
    config_file = tmp_path / ".misen.toml"
    first_args = SimpleNamespace(
        command="list",
        config=config_file,
        executor="auto",
        workspace="auto",
    )
    second_args = SimpleNamespace(
        command="list",
        list_cacheable_only=True,
        list_incomplete=False,
        config=config_file,
        executor="auto",
        workspace="auto",
        experiment=CliExperiment(),
    )

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(
        Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids={"source"}))
    )

    experiment_cli(CliExperiment)

    assert "No tasks matched filters." in capsys.readouterr().out


def test_experiment_cli_list_command_incomplete_filter(monkeypatch, capsys, tmp_path) -> None:
    config_file = tmp_path / ".misen.toml"
    first_args = SimpleNamespace(
        command="list",
        config=config_file,
        executor="auto",
        workspace="auto",
    )
    second_args = SimpleNamespace(
        command="list",
        list_cacheable_only=False,
        list_incomplete=True,
        config=config_file,
        executor="auto",
        workspace="auto",
        experiment=CliExperiment(),
    )

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(
        Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids={"source"}))
    )

    experiment_cli(CliExperiment)

    output = capsys.readouterr().out
    assert "Incomplete Tasks" in output
    assert "sink" in output
    assert "source" not in output


def test_experiment_cli_tree_command_incomplete_filter_prints_missing_tasks(monkeypatch, capsys, tmp_path) -> None:
    config_file = tmp_path / ".misen.toml"
    first_args = SimpleNamespace(
        command="tree",
        config=config_file,
        executor="auto",
        workspace="auto",
    )
    second_args = SimpleNamespace(
        command="tree",
        tree_all=False,
        tree_max_depth=None,
        tree_cacheable_only=False,
        tree_incomplete=True,
        config=config_file,
        executor="auto",
        workspace="auto",
        experiment=CliExperiment(),
    )

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(
        Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids={"source"}))
    )

    experiment_cli(CliExperiment)

    output = capsys.readouterr().out
    assert "Incomplete" in output
    assert "Tasks" in output
    assert "sink" in output
    assert "source" not in output


def test_experiment_cli_incomplete_command_alias_prints_missing_tasks(monkeypatch, capsys, tmp_path) -> None:
    config_file = tmp_path / ".misen.toml"
    first_args = SimpleNamespace(
        command="incomplete",
        config=config_file,
        executor="auto",
        workspace="auto",
    )
    second_args = SimpleNamespace(
        command="incomplete",
        tree_all=False,
        tree_max_depth=None,
        tree_cacheable_only=False,
        tree_incomplete=True,
        config=config_file,
        executor="auto",
        workspace="auto",
        experiment=CliExperiment(),
    )

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(
        Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids={"source"}))
    )

    experiment_cli(CliExperiment)

    output = capsys.readouterr().out
    assert "Incomplete Tasks" in output
    assert "sink" in output
    assert "source" not in output


def test_experiment_cli_tree_command_incomplete_filter_handles_fully_complete_graph(
    monkeypatch, capsys, tmp_path
) -> None:
    config_file = tmp_path / ".misen.toml"
    first_args = SimpleNamespace(
        command="tree",
        config=config_file,
        executor="auto",
        workspace="auto",
    )
    second_args = SimpleNamespace(
        command="tree",
        tree_all=False,
        tree_max_depth=None,
        tree_cacheable_only=False,
        tree_incomplete=True,
        config=config_file,
        executor="auto",
        workspace="auto",
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
    config_file = tmp_path / ".misen.toml"
    executor = object()
    workspace = object()
    calls: dict[str, object] = {}

    def fake_submit_and_watch_jobs(*, experiment: object, executor: object, workspace: object) -> None:
        calls["experiment"] = experiment
        calls["executor"] = executor
        calls["workspace"] = workspace

    first_args = SimpleNamespace(
        command="run",
        config=config_file,
        executor="auto",
        workspace="auto",
    )
    second_args = SimpleNamespace(
        command="run",
        run_task=None,
        run_tui=True,
        config=config_file,
        executor="auto",
        workspace="auto",
        experiment=SimpleNamespace(tasks=lambda: {}),
    )

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(tui_module, "submit_and_watch_jobs", fake_submit_and_watch_jobs)
    monkeypatch.setattr(Executor, "auto", classmethod(lambda _cls, settings=None: executor))
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: workspace))

    experiment_cli(CliExperiment)

    assert calls["experiment"] is second_args.experiment
    assert calls["executor"] is executor
    assert calls["workspace"] is workspace


def test_experiment_cli_run_command_with_task_name(monkeypatch, tmp_path) -> None:
    config_file = tmp_path / ".misen.toml"
    executor = object()
    workspace = object()
    submit_calls: list[tuple[object, object, bool]] = []

    class StubTask:
        def submit(self, *, executor: object, workspace: object, blocking: bool = False) -> None:
            submit_calls.append((executor, workspace, blocking))

    class StubExperiment:
        def __init__(self, task: StubTask) -> None:
            self._task = task

        def __getitem__(self, key: str) -> StubTask:
            assert key == "task"
            return self._task

    first_args = SimpleNamespace(
        command="run",
        config=config_file,
        executor="auto",
        workspace="auto",
    )
    second_args = SimpleNamespace(
        command="run",
        run_task="task",
        run_tui=True,
        config=config_file,
        executor="auto",
        workspace="auto",
        experiment=StubExperiment(task=StubTask()),
    )

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(Executor, "auto", classmethod(lambda _cls, settings=None: executor))
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: workspace))

    experiment_cli(CliExperiment)

    assert submit_calls == [(executor, workspace, True)]


def test_experiment_cli_run_command_without_tui_submits_blocking(monkeypatch, tmp_path) -> None:
    config_file = tmp_path / ".misen.toml"
    workspace = object()
    submit_calls: list[tuple[set[Task[int]], object, bool]] = []

    class StubExecutor:
        def submit(self, *, tasks: set[Task[int]], workspace: object, blocking: bool = False) -> tuple[None, None]:
            submit_calls.append((tasks, workspace, blocking))
            return None, None

    class StubExperiment:
        def tasks(self) -> dict[str, Task[int]]:
            return {"task": Task(source, x=1)}

        def normalized_tasks(self) -> dict[str, Task[int]]:
            return self.tasks()

    first_args = SimpleNamespace(
        command="run",
        config=config_file,
        executor="auto",
        workspace="auto",
    )
    second_args = SimpleNamespace(
        command="run",
        run_task=None,
        run_tui=False,
        config=config_file,
        executor="auto",
        workspace="auto",
        experiment=StubExperiment(),
    )

    def fail_tui(**_kwargs: object) -> None:
        msg = "TUI should be bypassed with --no-tui"
        raise AssertionError(msg)

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(Executor, "auto", classmethod(lambda _cls, settings=None: StubExecutor()))
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: workspace))
    monkeypatch.setattr(tui_module, "submit_and_watch_jobs", fail_tui)

    experiment_cli(CliExperiment)

    assert len(submit_calls) == 1
    assert len(submit_calls[0][0]) == 1
    assert submit_calls[0][1] is workspace
    assert submit_calls[0][2] is True


def test_experiment_cli_parses_positional_run_command(monkeypatch) -> None:
    submit_calls: list[tuple[set[object], object, bool]] = []
    workspace = object()

    class StubExecutor:
        def submit(self, *, tasks: set[object], workspace: object, blocking: bool = False) -> tuple[None, None]:
            submit_calls.append((tasks, workspace, blocking))
            return None, None

    monkeypatch.setattr(Executor, "auto", classmethod(lambda _cls, settings=None: StubExecutor()))
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: workspace))
    monkeypatch.setattr(sys, "argv", ["prog", "run", "task"])

    experiment_cli(CliExperiment)

    assert len(submit_calls) == 1
    assert len(submit_calls[0][0]) == 1
    assert submit_calls[0][1] is workspace
    assert submit_calls[0][2] is True


def test_experiment_cli_parses_run_no_tui_flag(monkeypatch) -> None:
    submit_calls: list[tuple[set[object], object, bool]] = []
    workspace = object()

    class StubExecutor:
        def submit(self, *, tasks: set[object], workspace: object, blocking: bool = False) -> tuple[None, None]:
            submit_calls.append((tasks, workspace, blocking))
            return None, None

    def fail_tui(**_kwargs: object) -> None:
        msg = "TUI should be bypassed with --no-tui"
        raise AssertionError(msg)

    monkeypatch.setattr(Executor, "auto", classmethod(lambda _cls, settings=None: StubExecutor()))
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: workspace))
    monkeypatch.setattr(tui_module, "submit_and_watch_jobs", fail_tui)
    monkeypatch.setattr(sys, "argv", ["prog", "run", "--no-tui"])

    experiment_cli(CliExperiment)

    assert len(submit_calls) == 1
    assert len(submit_calls[0][0]) == 1
    assert submit_calls[0][1] is workspace
    assert submit_calls[0][2] is True


def test_experiment_cli_tree_command_with_task_positional(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "tree", "task"])
    monkeypatch.setattr(
        Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids=set()))
    )

    experiment_cli(CliExperiment)

    output = capsys.readouterr().out
    assert "task:" not in output
    assert "sink" in output
    assert "source(x=1)" in output


def test_experiment_cli_tree_command_rejects_unknown_task(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "tree", "missing"])
    monkeypatch.setattr(
        Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids=set()))
    )

    with pytest.raises(ValueError, match=r"no task named 'missing'"):
        experiment_cli(CliExperiment)


def test_experiment_cli_parses_tree_short_depth_flag(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "tree", "-L", "0", "--cacheable-only"])
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids=set())))

    experiment_cli(CliExperiment)

    assert "No tasks matched filters." in capsys.readouterr().out


def test_experiment_cli_parses_tree_incomplete_flag(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "tree", "--incomplete"])
    monkeypatch.setattr(
        Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids={"source"}))
    )

    experiment_cli(CliExperiment)

    output = capsys.readouterr().out
    assert "Incomplete Tasks" in output
    assert "sink" in output
    assert "source" not in output


def test_experiment_cli_parses_positional_incomplete_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "incomplete"])
    monkeypatch.setattr(
        Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids={"source"}))
    )

    experiment_cli(CliExperiment)

    output = capsys.readouterr().out
    assert "Incomplete Tasks" in output
    assert "sink" in output
    assert "source" not in output


def test_experiment_cli_parses_list_incomplete_flag(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "list", "--incomplete"])
    monkeypatch.setattr(
        Workspace, "auto", classmethod(lambda _cls, settings=None: _StatusWorkspace(done_ids={"source"}))
    )

    experiment_cli(CliExperiment)

    output = capsys.readouterr().out
    assert "Incomplete Tasks" in output
    assert "sink" in output
    assert "source" not in output


def test_experiment_cli_result_command(monkeypatch, capsys, tmp_path) -> None:
    config_file = tmp_path / ".misen.toml"
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
        config=config_file,
        executor="auto",
        workspace="auto",
    )
    second_args = SimpleNamespace(
        command="result",
        result_task="task",
        config=config_file,
        executor="auto",
        workspace="auto",
        experiment=experiment,
    )

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: workspace_obj))

    experiment_cli(CliExperiment)

    assert "value" in capsys.readouterr().out


def test_experiment_cli_result_command_requires_task_selector(monkeypatch, tmp_path) -> None:
    config_file = tmp_path / ".misen.toml"
    first_args = SimpleNamespace(
        command="result",
        config=config_file,
        executor="auto",
        workspace="auto",
    )
    second_args = SimpleNamespace(
        command="result",
        result_task=None,
        config=config_file,
        executor="auto",
        workspace="auto",
        experiment=SimpleNamespace(result=lambda _key, workspace: workspace),
    )

    _mock_cli(monkeypatch, first_args, second_args)
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: object()))

    with pytest.raises(ValueError, match=r"requires a task name"):
        experiment_cli(CliExperiment)


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


def test_submit_and_watch_jobs_calls_submit_without_blocking(monkeypatch) -> None:
    graph: DependencyGraph[Job] = DependencyGraph()
    graph.add_node(FakeJob(work_unit=WorkUnit(root=Task(source, x=1), dependencies=set()), states=["done"]))
    submit_args: dict[str, object] = {}

    class StubExecutor:
        def submit(
            self,
            tasks: set[Task[int]],
            workspace: object,
            *,
            blocking: bool = False,
        ) -> tuple[DependencyGraph[Job], None]:
            submit_args["tasks"] = tasks
            submit_args["workspace"] = workspace
            submit_args["blocking"] = blocking
            return graph, None

        def cleanup_snapshot(self, snapshot: object) -> None:
            pass

    class StubExperiment:
        def tasks(self) -> dict[str, Task[int]]:
            return {"task": Task(source, x=1)}

        def normalized_tasks(self) -> dict[str, Task[int]]:
            return self.tasks()

    seen: dict[str, object] = {}

    def fake_watch(
        *,
        named_tasks: dict[str, Task[int]],
        job_graph: DependencyGraph[Job],
        workspace: object,
        poll_interval_s: float = 0.2,
    ) -> None:
        seen["named_tasks"] = named_tasks
        seen["job_graph"] = job_graph
        seen["workspace"] = workspace
        seen["poll_interval_s"] = poll_interval_s

    workspace = object()
    monkeypatch.setattr(tui_module, "watch_tasks", fake_watch)
    tui_module.submit_and_watch_jobs(experiment=StubExperiment(), executor=StubExecutor(), workspace=workspace)

    assert submit_args["blocking"] is False
    assert submit_args["workspace"] is workspace
    assert len(cast("set[Task[int]]", submit_args["tasks"])) == 1
    assert seen["job_graph"] is graph
    assert seen["workspace"] is workspace
    assert set(cast("dict[str, Task[int]]", seen["named_tasks"]).keys()) == {"task"}


def test_submit_and_watch_jobs_suppresses_runtime_events_only_during_watch(monkeypatch) -> None:
    graph: DependencyGraph[Job] = DependencyGraph()
    graph.add_node(FakeJob(work_unit=WorkUnit(root=Task(source, x=1), dependencies=set()), states=["done"]))
    seen: dict[str, str | None] = {}

    class StubExecutor:
        def submit(
            self,
            tasks: set[Task[int]],
            workspace: object,
            *,
            blocking: bool = False,
        ) -> tuple[DependencyGraph[Job], None]:
            _ = tasks, workspace, blocking
            seen["during_submit"] = os.environ.get("MISEN_RUNTIME_EVENTS")
            seen["during_submit_job_board"] = os.environ.get("MISEN_RUNTIME_JOB_BOARD")
            return graph, None

        def cleanup_snapshot(self, snapshot: object) -> None:
            pass

    class StubExperiment:
        def tasks(self) -> dict[str, Task[int]]:
            return {"task": Task(source, x=1)}

        def normalized_tasks(self) -> dict[str, Task[int]]:
            return self.tasks()

    def fake_watch(
        *,
        named_tasks: dict[str, Task[int]],
        job_graph: DependencyGraph[Job],
        workspace: object,
        poll_interval_s: float = 0.2,
    ) -> None:
        _ = named_tasks, job_graph, workspace, poll_interval_s
        seen["during_watch"] = os.environ.get("MISEN_RUNTIME_EVENTS")
        seen["during_watch_job_board"] = os.environ.get("MISEN_RUNTIME_JOB_BOARD")

    monkeypatch.delenv("MISEN_RUNTIME_EVENTS", raising=False)
    monkeypatch.delenv("MISEN_RUNTIME_JOB_BOARD", raising=False)
    monkeypatch.setattr(tui_module, "watch_tasks", fake_watch)
    tui_module.submit_and_watch_jobs(experiment=StubExperiment(), executor=StubExecutor(), workspace=object())

    assert seen["during_submit"] is None
    assert seen["during_submit_job_board"] == "0"
    assert seen["during_watch"] == "0"
    assert seen["during_watch_job_board"] == "0"
    assert os.environ.get("MISEN_RUNTIME_EVENTS") is None
    assert os.environ.get("MISEN_RUNTIME_JOB_BOARD") is None


def test_job_state_index_maps_roots_and_jobs() -> None:
    dep_wu = WorkUnit(root=Task(source, x=1), dependencies=set())
    parent_wu = WorkUnit(root=Task(sink, x=2), dependencies={dep_wu})
    dep_job = FakeJob(work_unit=dep_wu, states=["running"], job_id="dep-1", log_path=Path("/tmp/dep.log"))
    parent_job = FakeJob(work_unit=parent_wu, states=["pending"], job_id=None, log_path=None)

    graph: DependencyGraph[Job] = DependencyGraph()
    graph.add_node(parent_job)
    graph.add_node(dep_job)

    index = tui_module._JobStateIndex.build(graph)

    assert index.work_unit_of_root(dep_wu.root) is dep_wu
    assert index.work_unit_of_root(parent_wu.root) is parent_wu
    assert index.job_for_work_unit(dep_wu) is dep_job
    assert index.job_for_work_unit(parent_wu) is parent_job
    assert index.job_for_work_unit(None) is None


def test_final_summary_lines_omit_hash_and_job_id(capsys) -> None:
    done_job = FakeJob(
        work_unit=WorkUnit(root=Task(source, x=1), dependencies=set()),
        states=["done"],
        job_id="DONE1",
    )
    failed_job = FakeJob(
        work_unit=WorkUnit(root=Task(sink, x=2), dependencies=set()),
        states=["failed"],
        job_id="FAIL1",
    )
    done_job.pid = 12345
    failed_job.pid = 67890

    tui_module._print_final_summary([done_job, failed_job])

    output = capsys.readouterr().err
    lines = [line for line in output.splitlines() if line.strip()]
    assert lines
    assert lines[0].startswith("complete ")
    assert "source(x=1)" in lines[0]
    assert any(line.startswith("failed") and "sink(x=2)" in line for line in lines)
    joined = "\n".join(lines)
    assert "job_id" not in joined
    assert "pid=" not in joined
    assert "[" not in joined  # no hash bracket suffix


def test_watch_tasks_uses_textual_runner(monkeypatch) -> None:
    graph: DependencyGraph[Job] = DependencyGraph()
    work_unit = WorkUnit(root=Task(source, x=3), dependencies=set())
    graph.add_node(FakeJob(work_unit=work_unit, states=["done"]))
    named_tasks = {"task": work_unit.root}

    seen: dict[str, object] = {}

    def fake_run_textual(
        *,
        named_tasks: dict[str, Task[int]],
        job_graph: DependencyGraph[Job],
        workspace: object,
        poll_interval_s: float,
    ) -> None:
        seen["named_tasks"] = named_tasks
        seen["job_graph"] = job_graph
        seen["workspace"] = workspace
        seen["poll_interval_s"] = poll_interval_s

    monkeypatch.setattr(tui_module, "_run_textual_task_monitor", fake_run_textual)

    workspace = object()
    tui_module.watch_tasks(
        named_tasks=named_tasks,
        job_graph=graph,
        workspace=workspace,
        poll_interval_s=0.0,
    )

    assert seen["job_graph"] is graph
    assert seen["workspace"] is workspace
    assert seen["named_tasks"] is named_tasks
    assert seen["poll_interval_s"] == 0.0
