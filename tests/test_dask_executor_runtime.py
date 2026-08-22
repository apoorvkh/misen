"""Executor-owned Dask scheduler and worker roles."""
# ruff: noqa: D103, S101, S603

from __future__ import annotations

import asyncio
import os
import shutil
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
    DASK_MEMORY_GIB_ENV,
    DASK_ROLE_ENV,
    DASK_SCHEDULER_ADDRESS_ENV,
    DASK_SCHEDULER_FILE_ENV,
    managed_cluster_script,
    run_role_from_env,
)

if TYPE_CHECKING:
    from pathlib import Path

_ROLE_COMMAND = [
    sys.executable,
    "-c",
    "from misen.utils.dask_runtime import run_role_from_env; run_role_from_env()",
]

_FAKE_ROLE_SOURCE = """
import os
import sys
import time
from pathlib import Path

role = os.environ.get("MISEN_DASK_ROLE")
if role == "scheduler":
    Path(os.environ["MISEN_DASK_SCHEDULER_FILE"]).write_text("tcp://fake-scheduler")
    time.sleep(60)
elif role == "worker":
    exit_code = int(os.environ.get("MISEN_TEST_WORKER_EXIT", "0"))
    if exit_code:
        sys.exit(exit_code)
    time.sleep(60)
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
            metadata = list(client.scheduler_info()["workers"].values())
            expected = pow(2, 5)
            assert len(metadata) == len(workers)
            assert all(worker["memory_limit"] == 1024**3 for worker in metadata)
            assert client.submit(pow, 2, 5).result() == expected
    finally:
        for worker in workers:
            _stop(worker)
        _stop(scheduler)


def test_runtime_http_listeners_are_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, tuple[tuple[object, ...], dict[str, object]]] = {}

    class FakeServer:
        address = "tcp://127.0.0.1:1234"

        def __init__(self, role: str, *args: object, **kwargs: object) -> None:
            captured[role] = (args, kwargs)

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def finished(self) -> None:
            return None

    monkeypatch.setattr(distributed, "Scheduler", lambda **kwargs: FakeServer("scheduler", **kwargs))
    monkeypatch.setattr(distributed, "Worker", lambda *args, **kwargs: FakeServer("worker", *args, **kwargs))

    asyncio.run(dask_runtime._run_scheduler(tmp_path / "scheduler-address"))  # noqa: SLF001
    asyncio.run(dask_runtime._run_worker("tcp://scheduler", nthreads=1, memory_gib=1))  # noqa: SLF001

    for _, kwargs in captured.values():
        assert kwargs["dashboard_address"] is None
    assert captured["scheduler"][1]["local_directory"] == str(tmp_path)


def test_no_dask_role_leaves_normal_worker_execution_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DASK_ROLE_ENV, raising=False)

    assert not run_role_from_env()


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
