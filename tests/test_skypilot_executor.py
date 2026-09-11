"""Reusable-worker executor configuration and command contracts."""

# ruff: noqa: ANN001, ANN201, D103, S101, SLF001
from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

import misen.executors.skypilot as skypilot_module
from misen import Task, meta
from misen.exceptions import ConfigError
from misen.executors.skypilot import SkyPilotExecutor
from misen.task_metadata import aggregate_resources
from misen.utils.graph import DependencyGraph
from misen.utils.work_unit import WorkUnit
from misen.workspace import Workspace


@meta(id="skypilot_chain_task", cache=False)
def _chain_task(value: int) -> int:
    return value


@meta(
    id="skypilot_cpu_task",
    cache=False,
    resources={"cpus": 4, "memory": 32, "time": 17},
)
def _cpu_task() -> None:
    return None


def _remote_workspace() -> MagicMock:
    workspace = MagicMock(spec=Workspace)
    workspace.bootstrap_transport.return_value = "fetch-from-object-store"
    workspace.get_temp_dir.return_value = Path(".cache/misen/test-workspace")
    workspace.read_job_file.side_effect = FileNotFoundError
    return workspace


def _work_unit(task: Task[Any]) -> WorkUnit:
    return WorkUnit(root=task, dependencies=set())


def _diamond_graph() -> tuple[DependencyGraph[WorkUnit], tuple[WorkUnit, WorkUnit, WorkUnit, WorkUnit]]:
    base = _work_unit(Task(_chain_task, value=1))
    left = WorkUnit(root=Task(_chain_task, value=2), dependencies={base})
    right = WorkUnit(root=Task(_chain_task, value=3), dependencies={base})
    root = WorkUnit(root=Task(_chain_task, value=4), dependencies={left, right})
    graph: DependencyGraph[WorkUnit] = DependencyGraph()
    base_index = graph.add_node(base)
    left_index = graph.add_node(left)
    right_index = graph.add_node(right)
    root_index = graph.add_node(root)
    graph.add_edge(left_index, base_index)
    graph.add_edge(right_index, base_index)
    graph.add_edge(root_index, left_index)
    graph.add_edge(root_index, right_index)
    return graph, (base, left, right, root)


@pytest.mark.parametrize("timeout", [True, 0, -1, 1.5])
def test_skypilot_validates_dask_startup_timeout_eagerly(timeout: Any) -> None:
    with pytest.raises(ValueError, match="dask_startup_timeout"):
        SkyPilotExecutor(workers=[{"cpus": 8, "memory": 64}], dask_startup_timeout=timeout)


@pytest.mark.parametrize("port", [True, 1023, 65536, 1.5])
def test_skypilot_validates_dask_scheduler_port_eagerly(port: Any) -> None:
    with pytest.raises(ValueError, match="dask_scheduler_port"):
        SkyPilotExecutor(workers=[{"cpus": 8, "memory": 64}], dask_scheduler_port=port)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"snapshot": False}, "requires snapshot=True"),
        ({"prewarm_envs": True}, "prewarm_envs=False"),
    ],
)
def test_remote_only_snapshot_modes_are_rejected_at_configuration(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        SkyPilotExecutor(workers=[{"cpus": 8, "memory": 64}], **kwargs)


@pytest.mark.parametrize(
    ("transport", "temp_dir", "message"),
    [
        (None, Path(".cache/misen"), "remotely fetchable workspace transport"),
        ("fetch", Path("/submitter-only/cache"), "relative workspace cache_dir"),
    ],
)
def test_local_only_workspace_shapes_fail_before_sdk_load(transport, temp_dir, message, monkeypatch) -> None:
    work_unit = _work_unit(Task(_cpu_task))
    graph: DependencyGraph[WorkUnit] = DependencyGraph()
    graph.add_node(work_unit)
    workspace = _remote_workspace()
    workspace.bootstrap_transport.return_value = transport
    workspace.get_temp_dir.return_value = temp_dir
    sdk_attempted = False

    def unexpected_sdk() -> object:
        nonlocal sdk_attempted
        sdk_attempted = True
        return object()

    monkeypatch.setattr(skypilot_module, "_load_skypilot", unexpected_sdk)

    with pytest.raises(ConfigError, match=message):
        SkyPilotExecutor(workers=[{"cpus": 8, "memory": 64}])._validate_submission(
            work_graph=graph,
            pending_work_units=[work_unit],
            workspace=cast("Workspace", workspace),
        )

    assert not sdk_attempted


def test_workspace_without_coordination_reads_fails_before_sdk_load(monkeypatch) -> None:
    work_unit = _work_unit(Task(_cpu_task))
    graph: DependencyGraph[WorkUnit] = DependencyGraph()
    graph.add_node(work_unit)
    workspace = _remote_workspace()
    workspace.supports_job_file_reads.return_value = False
    sdk_attempted = False

    def unexpected_sdk() -> object:
        nonlocal sdk_attempted
        sdk_attempted = True
        return object()

    monkeypatch.setattr(skypilot_module, "_load_skypilot", unexpected_sdk)

    with pytest.raises(ConfigError, match="submission-file coordination reads"):
        SkyPilotExecutor(workers=[{"cpus": 8, "memory": 64}])._validate_submission(
            work_graph=graph,
            pending_work_units=[work_unit],
            workspace=cast("Workspace", workspace),
        )

    assert not sdk_attempted


def test_missing_skypilot_sdk_has_actionable_lazy_import_error(monkeypatch) -> None:
    def missing_sky(module_name: str) -> object:
        assert module_name == "sky"
        msg = "No module named 'sky'"
        raise ModuleNotFoundError(msg, name="sky")

    monkeypatch.setattr(skypilot_module.importlib, "import_module", missing_sky)

    with pytest.raises(ConfigError, match="SkyPilotExecutor requires skypilot-nightly") as exc_info:
        skypilot_module._load_skypilot()

    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)
    assert "misen[skypilot]" in str(exc_info.value)


def test_skypilot_transitive_import_error_is_not_hidden(monkeypatch) -> None:
    error = ModuleNotFoundError("No module named 'sky_dependency'", name="sky_dependency")

    def broken_sky_import(_module_name: str) -> object:
        raise error

    monkeypatch.setattr(skypilot_module.importlib, "import_module", broken_sky_import)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        skypilot_module._load_skypilot()

    assert exc_info.value is error


def test_legacy_executor_options_are_rejected():
    for field in ("infra", "use_spot", "accelerators", "job_recovery"):
        with pytest.raises(TypeError):
            SkyPilotExecutor(workers=[{"cpus": 4, "memory": 32}], **{field: "legacy"})
    with pytest.raises(ValueError, match="at least one worker type"):
        SkyPilotExecutor()


@pytest.mark.parametrize("dask", [False, True])
def test_multinode_commands_and_shell_syntax(tmp_path, dask):
    executor = SkyPilotExecutor(workers=[{"cpus": 4, "memory": 12}], dask_startup_timeout=45, dask_scheduler_port=18786)
    command = executor._run_command(
        ["python", "worker.py", "spaces and 'quotes'"],
        {"EXAMPLE": "value with spaces"},
        tmp_path / "job.log",
        aggregate_resources([{"time": 17, "nodes": 3, "cpus": 4, "memory": 12}]),
        uses_dask_client=dask,
    )
    subprocess.run(["bash", "-n"], input=command, text=True, check=True)
    assert "17m" in command
    assert "SKYPILOT_NODE_RANK" in command
    if dask:
        for expected in (
            "SKYPILOT_NODE_IPS",
            "MISEN_DASK_ROLE=scheduler",
            "MISEN_DASK_ROLE=worker",
            "MISEN_DASK_EXPECTED_WORKERS=3",
            "MISEN_DASK_STARTUP_TIMEOUT=45",
            "MISEN_DASK_CPUS=4",
            "MISEN_DASK_MEMORY_GIB=12",
            "18786",
            "trap cleanup EXIT",
        ):
            assert expected in command
    else:
        assert '"${SKYPILOT_NODE_RANK:-0}" != "0"' in command


@pytest.mark.parametrize(
    "dask, prepare, rank", [(False, False, 0), (False, False, 1), (False, True, 1), (True, True, 1)]
)
def test_worker_command_preserves_arguments_environment_and_rank(tmp_path, dask, prepare, rank):
    executor = SkyPilotExecutor(workers=[{"cpus": 4, "memory": 12}])
    argument = "spaces, 'quotes', $expansion, `commands`, and\nnewlines"
    log_path = tmp_path / "log directory" / "job's log"
    script = "import json, os, sys; print(json.dumps([sys.argv[1], os.environ['EXAMPLE']]))"
    command = executor._run_command(
        [sys.executable, "-c", script, argument],
        {"EXAMPLE": argument},
        log_path,
        aggregate_resources([{"nodes": 2}]),
        uses_dask_client=dask,
    )
    result = subprocess.run(
        ["bash", "-c", command],
        env=os.environ | {"SKYPILOT_NODE_RANK": str(rank), "MISEN_PREPARE_ONLY": "1" if prepare else ""},
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    if prepare or rank == 0:
        assert json.loads(result.stdout) == [argument, argument]
    else:
        assert result.stdout == ""
    assert log_path.exists() == (not prepare and rank == 0)


def test_worker_command_propagates_payload_failure_through_logging(tmp_path):
    executor = SkyPilotExecutor(workers=[{"cpus": 1, "memory": 1}])
    log_path = tmp_path / "job.log"
    command = executor._run_command(
        [sys.executable, "-c", "import sys; print('task failed'); sys.exit(7)"],
        {},
        log_path,
        aggregate_resources([{}]),
        uses_dask_client=False,
    )
    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=10)
    assert result.returncode == 7
    assert log_path.read_text() == "task failed\n"


def test_installed_sdk_supports_owned_local_api():
    sky = pytest.importorskip("sky")
    inspect.signature(sky.api_start).bind(foreground=True, port=48123)
    assert callable(sky.server.common.get_local_api_server_port)
