"""Executor-owned Dask scheduler and worker roles."""
# ruff: noqa: D103, S101, S603

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Self

import distributed
import pytest
from distributed import Client

from misen.utils import dask_runtime
from misen.utils.dask_runtime import (
    DASK_CPUS_ENV,
    DASK_EXPECTED_WORKERS_ENV,
    DASK_MEMORY_GIB_ENV,
    DASK_ROLE_ENV,
    DASK_SCHEDULER_ADDRESS_ENV,
    DASK_SCHEDULER_FILE_ENV,
    DASK_SCHEDULER_HOST_ENV,
    DASK_SCHEDULER_PORT_ENV,
    DASK_STARTUP_TIMEOUT_ENV,
    managed_cluster_script,
    managed_ranked_cluster_script,
    run_role_from_env,
)

if TYPE_CHECKING:
    from pathlib import Path

_ROLE_COMMAND = [
    sys.executable,
    "-c",
    "from misen.utils.dask_runtime import run_role_from_env; run_role_from_env()",
]

_RANKED_ROLE_SOURCE = """
import os

from distributed import Client
from misen.utils.dask_runtime import run_role_from_env

if not run_role_from_env():
    with Client(os.environ["MISEN_DASK_SCHEDULER_ADDRESS"], set_as_default=False) as client:
        client.wait_for_workers(int(os.environ["MISEN_DASK_EXPECTED_WORKERS"]), timeout=10)
        assert client.submit(pow, 2, 5).result() == 32
"""

_FAKE_ROLE_SOURCE = """
import os
import signal
import sys
import time
from pathlib import Path

role = os.environ.get("MISEN_DASK_ROLE")
if role == "scheduler":
    scheduler_file = Path(os.environ["MISEN_DASK_SCHEDULER_FILE"])
    shutdown_file = scheduler_file.with_suffix(".shutdown")
    signal.signal(signal.SIGTERM, lambda *_: shutdown_file.touch())
    scheduler_file.write_text(str(shutdown_file))
    while not shutdown_file.exists():
        time.sleep(0.05)
elif role == "worker":
    exit_code = int(os.environ.get("MISEN_TEST_WORKER_EXIT", "0"))
    if exit_code:
        sys.exit(exit_code)
    while not Path(os.environ["MISEN_DASK_SCHEDULER_ADDRESS"]).exists():
        time.sleep(0.05)
elif role == "preflight":
    pass
else:
    time.sleep(float(os.environ.get("MISEN_TEST_COORDINATOR_DELAY", "0")))
    sys.exit(int(os.environ.get("MISEN_TEST_COORDINATOR_EXIT", "0")))
"""


def _stop(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    process.wait(timeout=10)


def _bash() -> str:
    bash = shutil.which("bash")
    assert bash is not None
    return bash


def _managed_script(*environment: str, workers: int = 2) -> str:
    return managed_cluster_script(
        [sys.executable, "-c", _FAKE_ROLE_SOURCE],
        [_bash(), "-c", 'exec "$@"', "_"],
        environment=dict(item.split("=", 1) for item in environment),
        workers=workers,
        cpus=1,
        memory_gib=1,
        startup_timeout=5,
    )


def _ranked_script(*environment: str, workers: int = 2) -> str:
    return managed_ranked_cluster_script(
        [sys.executable, "-c", _FAKE_ROLE_SOURCE],
        environment=dict(item.split("=", 1) for item in environment),
        workers=workers,
        cpus=1,
        memory_gib=1,
        startup_timeout=5,
        node_rank_env="MISEN_TEST_NODE_RANK",
        node_ips_env="MISEN_TEST_NODE_IPS",
        scheduler_port=18786,
    )


def _ranked_env(rank: int, *, tmp_path: Path) -> dict[str, str]:
    return os.environ | {
        "MISEN_TEST_NODE_RANK": str(rank),
        "MISEN_TEST_NODE_IPS": "127.0.0.1\n127.0.0.2",
        "TMPDIR": str(tmp_path),
    }


def test_scheduler_and_two_worker_roles_form_a_private_cluster(tmp_path: Path) -> None:
    address_file = tmp_path / "scheduler-address"
    scheduler = subprocess.Popen(
        _ROLE_COMMAND,
        env=os.environ
        | {
            DASK_ROLE_ENV: "scheduler",
            DASK_SCHEDULER_FILE_ENV: str(address_file),
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    workers: list[subprocess.Popen[bytes]] = []
    try:
        deadline = time.monotonic() + 10
        while not address_file.is_file():
            assert scheduler.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.05)

        address = address_file.read_text()
        workers.extend(
            subprocess.Popen(
                _ROLE_COMMAND,
                env=os.environ
                | {
                    DASK_ROLE_ENV: "worker",
                    DASK_SCHEDULER_ADDRESS_ENV: address,
                    DASK_CPUS_ENV: "1",
                    DASK_MEMORY_GIB_ENV: "1",
                },
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for _ in range(2)
        )
        with Client(address) as client:
            client.wait_for_workers(2, timeout=10)
            scheduler_info = client.scheduler_info()
            metadata = list(scheduler_info["workers"].values())
            expected = pow(2, 5)
            assert scheduler_info["services"] == {}
            assert len(metadata) == len(workers)
            assert all(worker["memory_limit"] == 1024**3 for worker in metadata)
            assert all(worker["services"] == {} for worker in metadata)
            assert client.submit(pow, 2, 5).result() == expected

        scheduler.terminate()
        assert scheduler.wait(timeout=10) == 0
        assert all(worker.wait(timeout=10) == 0 for worker in workers)
    finally:
        for worker in workers:
            _stop(worker)
        _stop(scheduler)


def test_runtime_dashboards_are_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, tuple[tuple[object, ...], dict[str, object]]] = {}
    http_started: list[str] = []

    class FakeServer:
        address = "tcp://127.0.0.1:1234"

        def __init__(self, role: str, *args: object, **kwargs: object) -> None:
            self.role = role
            captured[role] = (args, kwargs)
            self.start_http_server()

        def start_http_server(self) -> None:
            http_started.append(self.role)

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def finished(self) -> None:
            return None

    class FakeScheduler(FakeServer):
        def __init__(self, **kwargs: object) -> None:
            super().__init__("scheduler", **kwargs)

    class FakeWorker(FakeServer):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__("worker", *args, **kwargs)

    monkeypatch.setattr(distributed, "Scheduler", FakeScheduler)
    monkeypatch.setattr(distributed, "Worker", FakeWorker)

    asyncio.run(dask_runtime._run_scheduler(tmp_path / "scheduler-address"))  # noqa: SLF001
    asyncio.run(dask_runtime._run_worker("tcp://scheduler", nthreads=1, memory_gib=1))  # noqa: SLF001

    for _, kwargs in captured.values():
        assert kwargs["dashboard"] is False
        assert kwargs["dashboard_address"] == "127.0.0.1:0"
    assert not http_started
    assert captured["scheduler"][1]["local_directory"] == str(tmp_path)


def test_no_dask_role_leaves_normal_worker_execution_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DASK_ROLE_ENV, raising=False)

    assert not run_role_from_env()


def test_scheduler_role_binds_executor_selected_host_and_port(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scheduler_file = tmp_path / "scheduler-address"
    captured: dict[str, object] = {}

    async def fake_scheduler(path: Path, *, host: str | None, port: int) -> None:
        captured.update(path=path, host=host, port=port)

    monkeypatch.setenv(DASK_ROLE_ENV, "scheduler")
    monkeypatch.setenv(DASK_SCHEDULER_FILE_ENV, str(scheduler_file))
    monkeypatch.setenv(DASK_SCHEDULER_HOST_ENV, "10.2.3.4")
    monkeypatch.setenv(DASK_SCHEDULER_PORT_ENV, "18786")
    monkeypatch.setattr(dask_runtime, "_run_scheduler", fake_scheduler)

    assert run_role_from_env()
    assert captured == {"path": scheduler_file, "host": "10.2.3.4", "port": 18786}


def test_preflight_role_waits_for_the_complete_worker_group(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    class FakeClient:
        def __init__(self, address: str, **kwargs: object) -> None:
            observed.update(address=address, kwargs=kwargs)

        def wait_for_workers(self, workers: int, *, timeout: int) -> None:
            observed.update(workers=workers, timeout=timeout)

        def scheduler_info(self) -> dict[str, object]:
            return {"workers": {f"worker-{index}": {} for index in range(3)}}

        def close(self) -> None:
            observed["closed"] = True

    monkeypatch.setattr(distributed, "Client", FakeClient)
    monkeypatch.setenv(DASK_ROLE_ENV, "preflight")
    monkeypatch.setenv(DASK_SCHEDULER_ADDRESS_ENV, "tcp://10.2.3.4:18786")
    monkeypatch.setenv(DASK_EXPECTED_WORKERS_ENV, "3")
    monkeypatch.setenv(DASK_STARTUP_TIMEOUT_ENV, "45")

    assert run_role_from_env()
    assert observed == {
        "address": "tcp://10.2.3.4:18786",
        "kwargs": {"set_as_default": False, "timeout": 45},
        "workers": 3,
        "timeout": 45,
        "closed": True,
    }


def test_dask_role_preserves_unexpected_runtime_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail_run(coroutine: object) -> None:
        coroutine.close()  # type: ignore[attr-defined]
        msg = "runtime bug"
        raise AssertionError(msg)

    monkeypatch.setenv(DASK_ROLE_ENV, "scheduler")
    monkeypatch.setenv(DASK_SCHEDULER_FILE_ENV, str(tmp_path / "scheduler-address"))
    monkeypatch.setattr(asyncio, "run", fail_run)

    with pytest.raises(AssertionError, match="runtime bug"):
        run_role_from_env()


def test_managed_cluster_script_safely_embeds_commands() -> None:
    script = managed_cluster_script(
        ["bash", "-c", "printf '%s\\n' \"$VALUE\""],
        ["worker launcher", "--flag=spaces and 'quotes'"],
        environment={"VALUE": "spaces and 'quotes'"},
        workers=2,
        cpus=4,
        memory_gib=8,
        startup_timeout=30,
    )

    subprocess.run([_bash(), "-n"], input=script, text=True, check=True)
    assert "MISEN_DASK_MEMORY_GIB=8" in script
    assert "MISEN_DASK_ROLE=coordinator" not in script
    assert not any(line.lstrip().startswith("env ") for line in script.splitlines())


def test_managed_cluster_script_rejects_a_single_worker() -> None:
    with pytest.raises(ValueError, match="requires at least 2"):
        _managed_script(workers=1)


def test_managed_cluster_script_propagates_coordinator_status() -> None:
    expected_status = 7
    script = _managed_script(f"MISEN_TEST_COORDINATOR_EXIT={expected_status}")

    result = subprocess.run([_bash(), "-c", script], timeout=10, check=False)
    assert result.returncode == expected_status


def test_managed_cluster_script_fails_when_a_worker_exits() -> None:
    script = _managed_script("MISEN_TEST_COORDINATOR_DELAY=60", "MISEN_TEST_WORKER_EXIT=9")

    result = subprocess.run([_bash(), "-c", script], timeout=10, check=False)
    assert result.returncode == 1


def test_ranked_cluster_script_assigns_head_and_worker_roles_safely() -> None:
    script = managed_ranked_cluster_script(
        ["bash", "-c", "printf '%s\\n' \"$VALUE\""],
        environment={"VALUE": "spaces and 'quotes'"},
        workers=3,
        cpus=4,
        memory_gib=8,
        startup_timeout=30,
        node_rank_env="SKYPILOT_NODE_RANK",
        node_ips_env="SKYPILOT_NODE_IPS",
        scheduler_port=18786,
    )

    subprocess.run([_bash(), "-n"], input=script, text=True, check=True)
    assert "SKYPILOT_NODE_RANK" in script
    assert "SKYPILOT_NODE_IPS" in script
    assert "MISEN_DASK_ROLE=scheduler" in script
    assert "MISEN_DASK_ROLE=worker" in script
    assert "MISEN_DASK_ROLE=preflight" in script
    assert "MISEN_DASK_ROLE=coordinator" not in script
    assert "MISEN_DASK_EXPECTED_WORKERS=3" in script
    assert "MISEN_DASK_CPUS=4" in script
    assert "MISEN_DASK_MEMORY_GIB=8" in script
    assert "MISEN_DASK_STARTUP_TIMEOUT=30" in script
    assert "18786" in script


def test_ranked_cluster_script_propagates_non_head_worker_failure(tmp_path: Path) -> None:
    expected_status = 9
    script = _ranked_script(f"MISEN_TEST_WORKER_EXIT={expected_status}")

    result = subprocess.run(
        [_bash(), "-c", script],
        env=_ranked_env(1, tmp_path=tmp_path),
        timeout=10,
        check=False,
    )

    assert result.returncode == expected_status


def test_ranked_cluster_script_cleans_up_and_propagates_coordinator_failure(tmp_path: Path) -> None:
    expected_status = 7
    script = _ranked_script(f"MISEN_TEST_COORDINATOR_EXIT={expected_status}")

    result = subprocess.run(
        [_bash(), "-c", script],
        env=_ranked_env(0, tmp_path=tmp_path),
        timeout=10,
        check=False,
    )

    assert result.returncode == expected_status
    assert not list(tmp_path.glob("misen-dask.*"))


def test_ranked_cluster_script_forms_a_real_two_rank_cluster(tmp_path: Path) -> None:
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        scheduler_port = reservation.getsockname()[1]
    script = managed_ranked_cluster_script(
        [sys.executable, "-c", _RANKED_ROLE_SOURCE],
        environment={},
        workers=2,
        cpus=1,
        memory_gib=1,
        startup_timeout=10,
        node_rank_env="MISEN_TEST_NODE_RANK",
        node_ips_env="MISEN_TEST_NODE_IPS",
        scheduler_port=scheduler_port,
    )
    head: subprocess.Popen[bytes] | None = None
    worker: subprocess.Popen[bytes] | None = None
    try:
        head = subprocess.Popen(
            [_bash(), "-c", script],
            env=_ranked_env(0, tmp_path=tmp_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        worker = subprocess.Popen(
            [_bash(), "-c", script],
            env=_ranked_env(1, tmp_path=tmp_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        head_stdout, head_stderr = head.communicate(timeout=20)
        worker_stdout, worker_stderr = worker.communicate(timeout=20)

        assert head.returncode == 0, (head_stdout, head_stderr)
        assert worker.returncode == 0, (worker_stdout, worker_stderr)
        assert not list(tmp_path.glob("misen-dask.*"))
    finally:
        _stop(head)
        _stop(worker)
