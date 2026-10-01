"""SSH scheduling and lifecycle contracts, including a real loopback SSH server."""

# ruff: noqa: ANN001, ANN002, ANN003, ANN202, D103, S101, SLF001
from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import sys
import threading
import time

import msgspec
import pytest
from processkit import process_is_alive

from misen import DASK_CLIENT, Task, meta
from misen.exceptions import ConfigError, JobFailedError
from misen.executor import Executor
from misen.executors.ssh import SSHExecutor, SSHWorker, _BufferedLog
from misen.task_metadata import _normalize_resources
from misen.utils.work_unit import WorkUnit
from misen.workspaces.disk import DiskWorkspace
from misen.workspaces.memory import InMemoryWorkspace


@meta(id="ssh-executor-probe", cache=False, resources={"cpus": 1, "memory": 1})
def _probe(value: str) -> str:
    print(value, flush=True)
    return value


@meta(id="ssh-executor-cached-add", cache=True, resources={"cpus": 1, "memory": 1})
def _add(a: int, b: int) -> int:
    print(f"adding {a} + {b}", flush=True)
    return a + b


@meta(id="ssh-executor-dask", cache=True, resources={"nodes": 2, "cpus": 1, "memory": 1})
def _distributed_count(client) -> int:
    return len(client.scheduler_info()["workers"])


def _unit(value="probe", **resources):
    return WorkUnit(root=Task(_probe, value).with_resources(**resources), dependencies=set())


@pytest.fixture
def ssh_server(tmp_path):
    if sys.platform != "linux":
        pytest.skip("SSH workers require Linux")
    sdk = pytest.importorskip("asyncssh")
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    key = sdk.generate_private_key("ssh-ed25519")
    client_key = sdk.generate_private_key("ssh-ed25519")
    key_path = tmp_path / "identity"
    client_key.write_private_key(key_path)
    key_path.chmod(0o600)
    connections = []
    sessions = set()

    class Server(sdk.SSHServer):
        def connection_made(self, connection):
            connections.append(connection)

        def public_key_auth_supported(self):
            return True

        def validate_public_key(self, username, public_key):
            return username == "misen-test" and public_key.export_public_key() == client_key.export_public_key()

    async def serve(process):
        connection = process.get_extra_info("connection")
        active = connection.get_extra_info("active_sessions", 0)
        if active >= 8:
            process.stderr.write(b"SSH server session limit reached\n")
            process.exit(99)
            return
        connection.set_extra_info(
            active_sessions=active + 1,
            peak_sessions=max(active + 1, connection.get_extra_info("peak_sessions", 0)),
        )
        sessions.add(asyncio.current_task())
        child = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "-c",
            "exec " + process.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def input_stream():
            try:
                while chunk := await process.stdin.read(65536):
                    child.stdin.write(chunk)
                    await child.stdin.drain()
            finally:
                child.stdin.close()

        async def output_stream(source, destination):
            while chunk := await source.read(65536):
                destination.write(chunk)
                await destination.drain()

        input_task = asyncio.create_task(input_stream())
        output_tasks = [
            asyncio.create_task(output_stream(child.stdout, process.stdout)),
            asyncio.create_task(output_stream(child.stderr, process.stderr)),
        ]
        try:
            status = await child.wait()
            await asyncio.gather(*output_tasks, return_exceptions=True)
            process.exit(status)
        finally:
            input_task.cancel()
            await asyncio.gather(input_task, return_exceptions=True)
            sessions.discard(asyncio.current_task())
            connection.set_extra_info(active_sessions=connection.get_extra_info("active_sessions") - 1)

    async def start():
        return await sdk.listen(
            "127.0.0.1",
            0,
            server_factory=Server,
            server_host_keys=[key],
            process_factory=serve,
            encoding=None,
        )

    server = asyncio.run_coroutine_threadsafe(start(), loop).result(timeout=5)
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text(f"[127.0.0.1]:{server.get_port()} {key.export_public_key().decode()}")
    config = tmp_path / "ssh_config"
    config.write_text(
        "Host node1 node2\n  Hostname 127.0.0.1\n"
        f"  Port {server.get_port()}\n  User misen-test\n  IdentityFile {key_path}\n"
    )
    yield dict(ssh_config=str(config), known_hosts=str(known_hosts), connect_timeout=3), connections, loop

    async def stop():
        for connection in connections:
            connection.close()
        await asyncio.gather(*(connection.wait_closed() for connection in connections), return_exceptions=True)
        if sessions:
            await asyncio.wait_for(asyncio.gather(*sessions, return_exceptions=True), timeout=10)
        server.close()
        await server.wait_closed()

    asyncio.run_coroutine_threadsafe(stop(), loop).result(timeout=10)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=10)
    loop.close()


class _Snapshot:
    snapshot_key = "test-snapshot"

    def __init__(self, command="true", **_kwargs):
        self.command = command
        self.count = 0

    def prepare_job(self, unit, workspace, **_kwargs):
        self.count += 1
        job_id = f"ssh-job-{self.count}"
        script = 'if [ -n "${MISEN_PREPARE_ONLY:-}" ]; then exit 0; fi\n' + self.command
        return job_id, ["bash", "-c", script], {}, workspace.get_job_log(job_id, unit)


class _PayloadSnapshot(_Snapshot):
    def prepare_job(self, unit, workspace, **_kwargs):
        self.count += 1
        job_id = str(self.count)
        payload = workspace.put_job_file("test", job_id + ".pkl", unit.as_payload(workspace, job_id))
        script = 'if [ -n "${MISEN_PREPARE_ONLY:-}" ]; then exit 0; fi\n' + shlex.join(
            [sys.executable, "-m", "misen.utils.execute", "--payload", payload]
        )
        return job_id, ["bash", "-c", script], {}, workspace.get_job_log(job_id, unit)


def _wait(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert predicate()


@pytest.mark.parametrize(
    "worker",
    [
        {"hosts": []},
        {"hosts": [""]},
        {"hosts": ["user@host"]},
        {"hosts": ["a", "a"]},
        {"hosts": ["a"], "cpus": 0},
        {"hosts": ["a"], "max_concurrent_jobs": 0},
        {"hosts": ["a"], "max_concurrent_jobs": -1},
        {"hosts": ["a"], "max_concurrent_jobs": True},
        {"hosts": ["a"], "memory": True},
        {"hosts": ["a"], "addresses": ["a", "b"]},
        {"hosts": ["a"], "accelerator_memory": 8},
        {"hosts": ["a"], "accelerators": 2, "accelerator_indices": [0, 0]},
    ],
)
def test_worker_rejects_invalid_configuration(worker):
    with pytest.raises(ValueError):
        SSHWorker(**worker)


def test_resource_matching_and_exclusive_multinode():
    worker = SSHWorker(hosts=["a"], cpus=4, memory=16, accelerators=2, accelerator_memory=24)
    request = _normalize_resources({"cpus": 2, "memory": 8, "accelerators": 1, "accelerator_memory": 16})
    assert worker.fits(request)
    assert worker.fits(request, [request])
    assert not worker.fits(request, [request, request])
    assert not worker.fits(request | {"accelerator_memory": 40})
    assert not worker.fits(request | {"accelerator_type": "rocm"})
    multi = SSHWorker(hosts=["a", "b"], cpus=4, memory=16)
    two_nodes = _normalize_resources({"nodes": 2})
    assert multi.fits(two_nodes)
    assert not multi.fits(two_nodes, [two_nodes])
    assert not multi.fits(_normalize_resources({}))


def test_alias_and_constructor_decode():
    assert Executor.resolve_type("ssh") is SSHExecutor
    executor = msgspec.convert({"workers": [{"hosts": ["box"], "cpus": 4, "max_concurrent_jobs": 2}]}, type=SSHExecutor)
    assert executor.workers[0].cpus == 4
    assert executor.workers[0].max_concurrent_jobs == 2
    for alias in ("aws", "skypilot"):
        with pytest.raises(ConfigError):
            Executor.resolve_type(alias)


def test_validation_rejects_memory_workspace_and_impossible_resources(monkeypatch):
    monkeypatch.setattr("misen.executors.ssh._sdk", lambda: object())
    executor = SSHExecutor(workers=[SSHWorker(hosts=["box"])])
    with pytest.raises(ConfigError, match="shared filesystem"):
        executor._validate_submission(work_graph=None, pending_work_units=[_unit()], workspace=InMemoryWorkspace())


def test_log_batches_preserve_all_bytes_and_flush_on_close():
    writes = []

    class Output:
        def write(self, value):
            writes.append(value)

    async def io(function, *args):
        return function(*args)

    async def run():
        log = _BufferedLog(Output(), io)
        for _ in range(100):
            await log.write(b"x" * 10000)
        await log.write(b"last bytes")
        await log.close()

    asyncio.run(run())
    assert b"".join(writes) == b"x" * 1000000 + b"last bytes"
    assert len(writes) < 10
    assert max(map(len, writes)) <= 256 * 1024 + 10000


def test_log_flushes_quiet_output_and_reports_write_failure(monkeypatch):
    monkeypatch.setattr("misen.executors.ssh._LOG_FLUSH_SECONDS", 0.01)

    async def run():
        wrote = asyncio.Event()

        class Output:
            def write(self, _value):
                wrote.set()
                raise OSError("disk full")

        async def io(function, *args):
            return function(*args)

        log = _BufferedLog(Output(), io)
        await log.write(b"quiet output")
        await asyncio.wait_for(wrote.wait(), timeout=2)
        await asyncio.wait_for(log.failed.wait(), timeout=2)
        with pytest.raises(OSError, match="disk full"):
            await log.close()
        assert log.writer.done()

    asyncio.run(run())


def test_cancelling_log_producer_waits_for_inflight_write(monkeypatch):
    monkeypatch.setattr("misen.executors.ssh._LOG_BATCH_BYTES", 1)

    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        writes = []

        class Output:
            def write(self, value):
                writes.append(value)

        async def io(function, *args):
            started.set()
            await release.wait()
            return function(*args)

        log = _BufferedLog(Output(), io)
        producer = asyncio.create_task(log.write(b"only once"))
        await started.wait()
        producer.cancel()
        await asyncio.sleep(0)
        assert not producer.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await producer
        await log.close()
        assert writes == [b"only once"]

    asyncio.run(run())


def test_ssh_logs_and_affinity(ssh_server, tmp_path):
    config, _, _ = ssh_server
    code = "import os,sys; print('cpus='+str(sorted(os.sched_getaffinity(0)))); print('err',file=sys.stderr); print('x'*300000)"
    snapshot = _Snapshot(shlex.join([sys.executable, "-c", code]))
    workspace = DiskWorkspace(directory=str(tmp_path / "workspace"))
    with SSHExecutor(workers=[SSHWorker(hosts=["node1"], cpus=1)], **config) as executor:
        job = executor._dispatch(_unit(), set(), workspace, snapshot)
        _wait(lambda: job.state() in {"done", "failed"})
        assert job.state() == "done", job.failure
        log = job.log_path.read_text()
        assert f"cpus={[min(os.sched_getaffinity(0))]}" in log
        assert "err" in log
        assert log.count("x") >= 300000


def test_queued_cancellation_and_dependency_failure(ssh_server, tmp_path):
    config, _, _ = ssh_server
    workspace = DiskWorkspace(directory=str(tmp_path / "workspace"))
    marker = tmp_path / "should-not-exist"
    snapshot = _Snapshot("sleep 20")
    with SSHExecutor(workers=[SSHWorker(hosts=["node1"])], **config) as executor:
        first = executor._dispatch(_unit("first"), set(), workspace, snapshot)
        second = executor._dispatch(_unit("second"), set(), workspace, snapshot)
        third = executor._dispatch(_unit("third"), {first}, workspace, snapshot)
        second.commands = ("true", f"touch {shlex.quote(str(marker))}")
        second.cancel()
        _wait(lambda: second.state() == "failed")
        first.cancel()
        _wait(lambda: third.state() == "failed")
        assert not marker.exists()
        assert "prerequisite" in third.failure.reason


def test_running_cancellation_kills_child_and_releases_capacity(ssh_server, tmp_path):
    config, _, _ = ssh_server
    workspace = DiskWorkspace(directory=str(tmp_path / "workspace"))
    pid_file = tmp_path / "pid"
    snapshot = _Snapshot(f"sleep 60 & echo $! > {shlex.quote(str(pid_file))}; wait")
    with SSHExecutor(workers=[SSHWorker(hosts=["node1"])], **config) as executor:
        job = executor._dispatch(_unit("cancel"), set(), workspace, snapshot)
        _wait(pid_file.exists)
        child = int(pid_file.read_text())
        job.cancel()
        _wait(lambda: job.state() == "failed")
        assert not process_is_alive(child)
        snapshot.command = "echo next-job"
        next_job = executor._dispatch(_unit("next"), set(), workspace, snapshot)
        _wait(lambda: next_job.state() in {"done", "failed"})
        assert next_job.state() == "done", next_job.failure


def test_connection_reuse_and_independent_cancellation(ssh_server, tmp_path, monkeypatch):
    if len(os.sched_getaffinity(0)) < 2:
        pytest.skip("Needs two CPU slots")
    config, connections, _ = ssh_server
    workspace = DiskWorkspace(directory=str(tmp_path / "workspace"))
    # Cancellation must wake the control task without waiting for a heartbeat.
    monkeypatch.setattr("misen.executors.ssh._HEARTBEAT_SECONDS", 30)
    first_pid, second_pid = tmp_path / "first.pid", tmp_path / "second.pid"
    release = tmp_path / "release"
    snapshot = _Snapshot(f"echo $$ > {shlex.quote(str(first_pid))}; sleep 60")
    with SSHExecutor(workers=[SSHWorker(hosts=["node1"], cpus=2)], **config) as executor:
        first = executor._dispatch(_unit("first"), set(), workspace, snapshot)
        snapshot.command = (
            f"echo $$ > {shlex.quote(str(second_pid))}; "
            f"while [ ! -f {shlex.quote(str(release))} ]; do sleep .05; done; echo independent-output"
        )
        second = executor._dispatch(_unit("second"), set(), workspace, snapshot)
        _wait(lambda: first_pid.exists() and second_pid.exists())
        assert len(connections) == 1
        first.cancel()
        _wait(lambda: first.state() == "failed", timeout=3)
        assert not process_is_alive(int(first_pid.read_text()))
        assert second.state() == "running"
        assert process_is_alive(int(second_pid.read_text()))
        release.touch()
        _wait(lambda: second.state() in {"done", "failed"})
        assert second.state() == "done", second.failure
        assert "independent-output" in second.log_path.read_text()
        snapshot.command = "echo reused-again"
        third = executor._dispatch(_unit("third"), set(), workspace, snapshot)
        _wait(lambda: third.state() in {"done", "failed"})
        assert third.state() == "done", third.failure
        assert "reused-again" in third.log_path.read_text()
        assert len(connections) == 1
        assert not connections[0].is_closed()
    _wait(connections[0].is_closed)


def test_more_than_eight_jobs_share_a_bounded_connection_pool(ssh_server, tmp_path):
    job_count = 12
    if len(os.sched_getaffinity(0)) < job_count:
        pytest.skip("Needs twelve CPU slots")
    config, connections, _ = ssh_server
    workspace = DiskWorkspace(directory=str(tmp_path / "workspace"))
    release = tmp_path / "release"
    pid_files = [tmp_path / f"pid-{index}" for index in range(job_count)]
    snapshot = _Snapshot()
    with SSHExecutor(workers=[SSHWorker(hosts=["node1"], cpus=job_count, memory=job_count)], **config) as executor:
        jobs = []
        for index, pid_file in enumerate(pid_files):
            snapshot.command = (
                f"echo $$ > {shlex.quote(str(pid_file))}; "
                f"while [ ! -f {shlex.quote(str(release))} ]; do sleep .1; done; echo pool-output-{index}"
            )
            jobs.append(executor._dispatch(_unit(f"parallel-{index}"), set(), workspace, snapshot))
        _wait(lambda: all(path.exists() and path.stat().st_size for path in pid_files))
        assert all(job.state() == "running" for job in jobs)
        assert len(connections) == 2
        assert sorted(connection.get_extra_info("peak_sessions") for connection in connections) == [4, 8]
        jobs[0].cancel()
        _wait(lambda: jobs[0].state() == "failed")
        assert not process_is_alive(int(pid_files[0].read_text()))
        assert all(job.state() == "running" for job in jobs[1:])
        release.touch()
        _wait(lambda: all(job.state() in {"done", "failed"} for job in jobs))
        for index, job in enumerate(jobs[1:], 1):
            assert job.state() == "done", job.failure
            assert f"pool-output-{index}" in job.log_path.read_text()
        snapshot.command = "echo reuse-after-pool"
        later = executor._dispatch(_unit("reuse"), set(), workspace, snapshot)
        _wait(lambda: later.state() in {"done", "failed"})
        assert later.state() == "done", later.failure
        assert len(connections) == 2
    _wait(lambda: all(connection.is_closed() for connection in connections))


def test_cancellation_while_waiting_for_job_limit(ssh_server, tmp_path):
    if len(os.sched_getaffinity(0)) < 2:
        pytest.skip("Needs two CPU slots")
    config, connections, _ = ssh_server
    workspace = DiskWorkspace(directory=str(tmp_path / "workspace"))
    started, unwanted = tmp_path / "started", tmp_path / "unwanted"
    snapshot = _Snapshot(f"touch {shlex.quote(str(started))}; sleep 60")
    with SSHExecutor(workers=[SSHWorker(hosts=["node1"], cpus=2, max_concurrent_jobs=1)], **config) as executor:
        first = executor._dispatch(_unit("active"), set(), workspace, snapshot)
        _wait(started.exists)
        snapshot.command = f"touch {shlex.quote(str(unwanted))}"
        waiting = executor._dispatch(_unit("waiting"), set(), workspace, snapshot)
        _wait(lambda: waiting in executor._session.jobs)
        assert waiting.state() == "pending"
        waiting.cancel()
        _wait(lambda: waiting.state() == "failed", timeout=3)
        assert first.state() == "running"
        assert not unwanted.exists()
        first.cancel()
        _wait(lambda: first.state() == "failed")
        snapshot.command = "echo released-slot"
        last = executor._dispatch(_unit("last"), set(), workspace, snapshot)
        _wait(lambda: last.state() in {"done", "failed"})
        assert last.state() == "done", last.failure
        assert len(connections) == 1


def test_unknown_host_key_is_rejected(ssh_server, tmp_path):
    config, _, _ = ssh_server
    wrong = tmp_path / "empty-known-hosts"
    wrong.write_text("")
    config["known_hosts"] = str(wrong)
    workspace = DiskWorkspace(directory=str(tmp_path / "workspace"))
    with SSHExecutor(workers=[SSHWorker(hosts=["node1"])], **config) as executor:
        job = executor._dispatch(_unit(), set(), workspace, _Snapshot())
        _wait(lambda: job.state() == "failed")
        assert "host key" in job.failure.reason.lower()


def test_submit_runs_dag_and_publishes_cached_results(ssh_server, tmp_path, monkeypatch):
    config, _, _ = ssh_server
    workspace = DiskWorkspace(directory=str(tmp_path / "workspace"))

    monkeypatch.setattr("misen.utils.snapshot.ProjectSnapshot", _PayloadSnapshot)
    a = Task(_add, 1, 2)
    b = Task(_add, 3, 4)
    total = Task(_add, a, b)
    with SSHExecutor(workers=[SSHWorker(hosts=["node1"], cpus=2, memory=8)], **config) as executor:
        executor.submit({total}, workspace, blocking=True)
        assert total.result(workspace=workspace) == 10
        graph = executor.submit({total}, workspace, blocking=True)
        assert all(job.state() == "done" for job in graph)


def test_remote_failure_propagates_to_blocking_submit(ssh_server, tmp_path, monkeypatch):
    config, _, _ = ssh_server
    workspace = DiskWorkspace(directory=str(tmp_path / "workspace"))
    monkeypatch.setattr("misen.utils.snapshot.ProjectSnapshot", lambda **kwargs: _Snapshot("exit 9"))
    with SSHExecutor(workers=[SSHWorker(hosts=["node1"])], **config) as executor:
        with pytest.raises(JobFailedError):
            executor.submit({_unit().root}, workspace, blocking=True)


def test_multinode_executes_task_once(ssh_server, tmp_path):
    config, _, _ = ssh_server
    workspace = DiskWorkspace(directory=str(tmp_path / "workspace"))
    marker = tmp_path / "ranks"
    snapshot = _Snapshot(f'echo "$MISEN_NODE_RANK" >> {shlex.quote(str(marker))}')
    with SSHExecutor(workers=[SSHWorker(hosts=["node1", "node2"])], **config) as executor:
        job = executor._dispatch(_unit(nodes=2), set(), workspace, snapshot)
        _wait(lambda: job.state() in {"done", "failed"})
        assert job.state() == "done", job.failure
        assert marker.read_text() == "0\n"


def test_multinode_dask_runs_and_stops_all_ranks(ssh_server, tmp_path, monkeypatch):
    import socket

    config, _, _ = ssh_server
    workspace = DiskWorkspace(directory=str(tmp_path / "workspace"))
    monkeypatch.setattr("misen.utils.snapshot.ProjectSnapshot", _PayloadSnapshot)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    task = Task(_distributed_count, DASK_CLIENT)
    with SSHExecutor(
        workers=[SSHWorker(hosts=["node1", "node2"], addresses=["127.0.0.1", "127.0.0.1"])],
        dask_scheduler_port=port,
        dask_startup_timeout=15,
        **config,
    ) as executor:
        jobs = executor.submit({task}, workspace)
        job = next(iter(jobs))
        _wait(lambda: job.state() in {"done", "failed"}, timeout=30)
        assert job.state() == "done", (job.failure, job.log_path.read_text())
        assert task.result(workspace=workspace) == 2
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))


@pytest.mark.parametrize("job_count", [1, 2])
def test_connection_loss_fails_job_and_cleans_remote_children(ssh_server, tmp_path, job_count):
    if len(os.sched_getaffinity(0)) < job_count:
        pytest.skip("Needs enough CPU slots for concurrent jobs")
    config, connections, loop = ssh_server
    workspace = DiskWorkspace(directory=str(tmp_path / "workspace"))
    pid_files = [tmp_path / f"pid-{index}" for index in range(job_count)]
    snapshot = _Snapshot()
    with SSHExecutor(workers=[SSHWorker(hosts=["node1"], cpus=job_count)], **config) as executor:
        jobs = []
        for index, pid_file in enumerate(pid_files):
            snapshot.command = f"sleep 60 & echo $! > {shlex.quote(str(pid_file))}; wait"
            jobs.append(executor._dispatch(_unit(f"disconnect-{index}"), set(), workspace, snapshot))
        _wait(lambda: all(path.exists() for path in pid_files))
        children = [int(path.read_text()) for path in pid_files]
        assert len(connections) == 1
        for connection in connections:
            loop.call_soon_threadsafe(connection.abort)
        _wait(lambda: all(job.state() == "failed" for job in jobs))
        _wait(lambda: not any(process_is_alive(child) for child in children))
        assert executor._session.unavailable == {0}
        marker = tmp_path / "no-replay"
        snapshot.command = f"touch {shlex.quote(str(marker))}"
        later = executor._dispatch(_unit("later"), set(), workspace, snapshot)
        _wait(lambda: later.state() == "failed")
        assert not marker.exists()
        assert len(connections) == 1


def test_close_cancels_running_and_queued_work(ssh_server, tmp_path):
    config, _, _ = ssh_server
    workspace = DiskWorkspace(directory=str(tmp_path / "workspace"))
    pid_file = tmp_path / "pid"
    snapshot = _Snapshot(f"sleep 60 & echo $! > {shlex.quote(str(pid_file))}; wait")
    executor = SSHExecutor(workers=[SSHWorker(hosts=["node1"])], **config)
    job = executor._dispatch(_unit("active"), set(), workspace, snapshot)
    queued = executor._dispatch(_unit("queued"), set(), workspace, snapshot)
    _wait(pid_file.exists)
    child = int(pid_file.read_text())
    executor.close()
    assert job.state() == queued.state() == "failed"
    assert not process_is_alive(child)
    executor.close()


def test_preparation_timeout_fails_without_running_user_code(ssh_server, tmp_path):
    config, _, _ = ssh_server
    workspace = DiskWorkspace(directory=str(tmp_path / "workspace"))
    with SSHExecutor(workers=[SSHWorker(hosts=["node1"])], startup_timeout=1, **config) as executor:
        monkeypatch = pytest.MonkeyPatch()
        with monkeypatch.context() as patch:
            patch.setattr(SSHExecutor, "_commands", lambda *_args: ("sleep 60", "echo user-work"))
            job = executor._dispatch(_unit(), set(), workspace, _Snapshot())
            _wait(lambda: job.state() == "failed")
        assert "status 124 during preparation" in job.failure.reason
        assert "user-work" not in job.log_path.read_text()


def test_process_exit_cleans_jobs_without_explicit_close(ssh_server, tmp_path):
    config, _, _ = ssh_server
    pid_file = tmp_path / "pid"
    command = f"sleep 60 & echo $! > {shlex.quote(str(pid_file))}; wait"
    code = (
        "from tests.test_ssh_executor import _Snapshot, _unit, _wait\n"
        "from misen.executors.ssh import SSHExecutor, SSHWorker\n"
        "from misen.workspaces.disk import DiskWorkspace\n"
        "from pathlib import Path\n"
        f"executor = SSHExecutor(workers=[SSHWorker(hosts=['node1'])], **{config!r})\n"
        f"workspace = DiskWorkspace(directory={str(tmp_path / 'workspace')!r})\n"
        f"executor._dispatch(_unit(), set(), workspace, _Snapshot({command!r}))\n"
        f"_wait(Path({str(pid_file)!r}).exists)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20, check=True)
    assert "Exception ignored" not in result.stderr, result.stderr
    child = int(pid_file.read_text())
    _wait(lambda: not process_is_alive(child))
