"""Execute snapshot-pinned work on existing Linux hosts through AsyncSSH."""

# ruff: noqa: EM101, EM102, TRY003, TRY301
from __future__ import annotations

import asyncio
import atexit
import contextlib
import importlib
import json
import logging
import shlex
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, Self

import msgspec
from tyro.constructors import PrimitiveConstructorSpec

from misen.exceptions import ConfigError, ExecutionError, SubmissionError
from misen.executor import Executor, Job, JobState
from misen.utils.dask_runtime import (
    DEFAULT_DASK_SCHEDULER_PORT,
    DEFAULT_DASK_STARTUP_TIMEOUT,
    managed_ranked_cluster_script,
)
from misen.utils.hashing import TaskHash
from misen.utils.resource_env import resource_environment
from misen.utils.ssh_runtime import agent_command

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import BinaryIO

    from misen.task_metadata import Resources
    from misen.utils.graph import DependencyGraph
    from misen.utils.snapshot import ProjectSnapshot
    from misen.utils.work_unit import WorkUnit
    from misen.workspace import Workspace

__all__ = ["SSHExecutor", "SSHJob", "SSHWorker"]
logger = logging.getLogger(__name__)
_TERMINAL = {"done", "failed"}
_LEASE_SECONDS = 15
_STOP_GRACE_SECONDS = 5
_HEARTBEAT_SECONDS = 2
_JOBS_PER_CONNECTION = 8
_LOG_BATCH_BYTES = 256 * 1024
_LOG_FLUSH_SECONDS = 0.2


class _BufferedLog:
    """Coalesce output with bounded buffering and propagate background write failures."""

    def __init__(self, output: BinaryIO, io: Callable[..., Any]) -> None:
        self.output, self.io = output, io
        self.buffer = bytearray()
        self.lock = asyncio.Lock()
        self.finished = asyncio.Event()
        self.failed = asyncio.Event()
        self.error: Exception | None = None
        self.writer = asyncio.create_task(self._periodic_flush())

    def check(self) -> None:
        if self.error is not None:
            raise self.error

    async def write(self, data: bytes) -> None:
        async with self.lock:
            self.check()
            self.buffer.extend(data)
            if len(self.buffer) >= _LOG_BATCH_BYTES:
                await self._flush_locked()

    async def _flush_locked(self) -> None:
        if self.buffer:
            try:
                write = asyncio.create_task(self.io(self.output.write, bytes(self.buffer)))
                try:
                    await asyncio.shield(write)
                finally:
                    # A cancelled stream must not leave a thread writing into a
                    # file which the job's finalizer has already closed.
                    await write
                    self.buffer.clear()
            except Exception as exc:
                self.error = exc
                self.failed.set()
                raise

    async def _periodic_flush(self) -> None:
        try:
            while True:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self.finished.wait(), _LOG_FLUSH_SECONDS)
                async with self.lock:
                    self.check()
                    await self._flush_locked()
                if self.finished.is_set():
                    return
        except Exception as exc:  # noqa: BLE001 -- surfaced by readers and close(), including quiet jobs
            self.error = exc
            self.failed.set()

    async def close(self) -> None:
        self.finished.set()
        # Join in-flight writes before closing the file or publishing the log.
        await self.writer
        self.check()


def _validate(config: msgspec.Struct) -> None:
    for item in msgspec.structs.fields(config):
        try:
            value = msgspec.convert(getattr(config, item.name), type=item.type)
        except msgspec.ValidationError as exc:
            raise ValueError(f"{item.name}: {exc}") from exc
        setattr(config, item.name, value)


def _sdk() -> Any:
    try:
        return importlib.import_module("asyncssh")
    except ModuleNotFoundError as exc:
        if exc.name != "asyncssh":
            raise
        raise ConfigError("SSHExecutor requires the optional dependency: pip install 'misen[ssh]'.") from exc


class SSHWorker(msgspec.Struct, kw_only=True, forbid_unknown_fields=True):
    """One existing host or a fixed multi-node group, with per-node capacity.

    Hosts are SSH config aliases or DNS names. Authentication, ports, and jump
    hosts come from SSH config. ``addresses`` optionally supplies node-to-node
    addresses for Dask when aliases only resolve on the submitting machine.
    CPU slots map to each host's inherited CPU affinity. Device indices are
    physical device ordinals. Resource budgets coordinate this executor only.
    """

    hosts: Annotated[list[str], msgspec.Meta(min_length=1)]
    cpus: Annotated[int, msgspec.Meta(ge=1)] = 1
    memory: Annotated[int, msgspec.Meta(ge=1)] = 8
    accelerators: Annotated[int, msgspec.Meta(ge=0)] = 0
    accelerator_type: Literal["cuda", "rocm", "xpu"] = "cuda"
    accelerator_memory: Annotated[int, msgspec.Meta(ge=1)] | None = None
    accelerator_indices: list[Annotated[int, msgspec.Meta(ge=0)]] | None = None
    addresses: list[str] | None = None
    max_concurrent_jobs: Annotated[int, msgspec.Meta(ge=1)] | None = None

    def __post_init__(self) -> None:
        """Validate host topology and whole-device scheduling capacity."""
        _validate(self)
        if any(not host or any(c.isspace() for c in host) or "@" in host for host in self.hosts):
            raise ValueError("hosts must be nonempty SSH aliases or hostnames; configure User and Port in SSH config.")
        if len(set(self.hosts)) != len(self.hosts):
            raise ValueError("A worker cannot contain duplicate hosts.")
        if self.addresses is not None and (
            len(self.addresses) != len(self.hosts)
            or any(not address or any(c.isspace() for c in address) for address in self.addresses)
        ):
            raise ValueError("addresses must contain one nonempty node-to-node address per host.")
        if self.accelerator_indices is not None and (
            len(self.accelerator_indices) != self.accelerators
            or len(set(self.accelerator_indices)) != self.accelerators
        ):
            raise ValueError("accelerator_indices must contain exactly accelerators distinct device indices.")
        if self.accelerator_memory is not None and not self.accelerators:
            raise ValueError("accelerator_memory requires accelerators > 0.")

    def fits(self, request: Resources, reserved: Sequence[Resources] = ()) -> bool:
        """Match node topology and resource budgets; reserve multi-node groups exclusively."""
        if request["nodes"] != len(self.hosts) or (len(self.hosts) > 1 and reserved):
            return False
        if self.max_concurrent_jobs is not None and len(reserved) >= self.max_concurrent_jobs:
            return False
        if any(sum(r[k] for r in reserved) + request[k] > getattr(self, k) for k in ("cpus", "memory", "accelerators")):
            return False
        return not request["accelerators"] or (
            request["accelerator_type"] == self.accelerator_type
            and (
                request["accelerator_memory"] is None
                or (self.accelerator_memory is not None and self.accelerator_memory >= request["accelerator_memory"])
            )
        )


def _workers_from_cli(values: list[str]) -> list[SSHWorker]:
    try:
        return msgspec.json.decode(values[0], type=list[SSHWorker])
    except (msgspec.DecodeError, ValueError) as exc:
        raise ValueError(f"Invalid SSH workers JSON: {exc}") from exc


def _workers_to_cli(workers: list[SSHWorker]) -> list[str]:
    return [msgspec.json.encode(workers).decode()]


class SSHJob(Job):
    """A locally observed remote job with cancellation and streamed logs."""

    def __init__(  # noqa: PLR0917 -- internal job construction
        self,
        unit: WorkUnit,
        job_id: str,
        log_path: Path,
        workspace: Workspace,
        dependencies: set[SSHJob],
        commands: tuple[str, str],
        session: _Session,
    ) -> None:
        """Create a pending handle; scheduling starts after acceptance by the session."""
        super().__init__(unit, job_id, log_path)
        self.workspace, self.dependencies = workspace, dependencies
        self.commands, self.session = commands, session
        self.cancelled = threading.Event()
        self.cancel_event = asyncio.Event()
        self._state: JobState = "pending"

    def state(self) -> JobState:
        """Read local state without issuing an SSH or object-store query."""
        return self._state

    def cancel(self) -> None:
        """Request remote group cleanup, or prevent a pending job from starting."""
        if self._state not in _TERMINAL:
            self.cancelled.set()
            self.session.cancel(self)

    def finish(self, reason: str | None = None) -> None:
        """Finalize logs before publishing a terminal state."""
        if reason:
            self._record_failure(reason)
        try:
            self._finalize_log(self.workspace, failed=reason is not None)
        except Exception as exc:  # noqa: BLE001 -- a background failure must become an observable job failure
            self._record_failure(f"Could not finalize job log: {exc}")
        self._state = "failed" if self._failure_reason else "done"


@dataclass
class _Allocation:
    worker: int
    cpus: list[int]
    devices: list[int]


@dataclass
class _Connection:
    task: asyncio.Task[Any]
    users: int = 0


@dataclass
class _Session:
    config: SSHExecutor
    jobs: list[SSHJob] = field(default_factory=list)
    allocations: dict[SSHJob, _Allocation] = field(default_factory=dict)
    running: dict[SSHJob, asyncio.Task[None]] = field(default_factory=dict)
    unavailable: set[int] = field(default_factory=set)
    closed: bool = False
    exiting: bool = False

    def __post_init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.changed = asyncio.Event()
        self.connections: dict[str, list[_Connection]] = {}
        self.connect_locks: dict[str, asyncio.Lock] = {}
        self.thread = threading.Thread(target=self._serve, name="misen-ssh", daemon=True)
        self.thread.start()
        self.scheduler = asyncio.run_coroutine_threadsafe(self._schedule(), self.loop)
        atexit.register(self._exit_close)

    async def _io(self, function: Callable[..., Any], *args: Any) -> Any:
        # Python shuts down ThreadPoolExecutor before atexit callbacks. During
        # that final cleanup only, perform the remaining workspace I/O inline.
        if self.exiting:
            return function(*args)
        return await asyncio.to_thread(function, *args)

    def _exit_close(self) -> None:
        self.exiting = True
        self.close()

    def _serve(self) -> None:
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        finally:
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.run_until_complete(self.loop.shutdown_default_executor())
            self.loop.close()

    def cancel(self, job: SSHJob) -> None:
        def notify() -> None:
            job.cancel_event.set()
            self.changed.set()

        if not self.loop.is_closed():
            self.loop.call_soon_threadsafe(notify)

    async def add(self, job: SSHJob) -> None:
        if self.closed:
            raise SubmissionError("SSH executor is closed.")
        self.jobs.append(job)
        self.changed.set()

    def _allocate(self, job: SSHJob) -> _Allocation | None:
        for index, worker in sorted(enumerate(self.config.workers), key=lambda item: item[1].accelerators):
            if index in self.unavailable:
                continue
            active = [other for other, allocation in self.allocations.items() if allocation.worker == index]
            if not worker.fits(job.resources, [other.resources for other in active]):
                continue
            used_cpus = {cpu for other in active for cpu in self.allocations[other].cpus}
            used_devices = {device for other in active for device in self.allocations[other].devices}
            devices = worker.accelerator_indices or list(range(worker.accelerators))
            return _Allocation(
                index,
                [cpu for cpu in range(worker.cpus) if cpu not in used_cpus][: job.resources["cpus"]],
                [device for device in devices if device not in used_devices][: job.resources["accelerators"]],
            )
        return None

    async def _schedule(self) -> None:
        try:
            while not self.closed:
                await self.changed.wait()
                self.changed.clear()
                for job in self.jobs:
                    if job.state() != "pending":
                        continue
                    states = [dependency.state() for dependency in job.dependencies]
                    if job.cancelled.is_set() or "failed" in states:
                        reason = "Job cancelled." if job.cancelled.is_set() else "A prerequisite job failed."
                        await self._io(job.finish, reason)
                        continue
                    if any(state != "done" for state in states):
                        continue
                    if not any(
                        w.fits(job.resources) for i, w in enumerate(self.config.workers) if i not in self.unavailable
                    ):
                        await self._io(
                            job.finish, "All compatible SSH workers are unavailable after a connection failure."
                        )
                        continue
                    allocation = self._allocate(job)
                    if allocation is not None:
                        self.allocations[job] = allocation
                        job._state = "starting"  # noqa: SLF001
                        self.running[job] = asyncio.create_task(self._run(job, allocation))
        except Exception as exc:
            logger.exception("SSH scheduler failed")
            self.closed = True
            for job in self.jobs:
                job.cancelled.set()
                job.cancel_event.set()
                if job.state() == "pending":
                    await self._io(job.finish, f"SSH scheduler failed: {exc}")

    async def _run(self, job: SSHJob, allocation: _Allocation) -> None:
        reason = None
        try:
            stack = contextlib.ExitStack()
            try:
                output = await self._io(self._open_log, job, stack)
                log = _BufferedLog(output, self._io)
                try:
                    await self._run_group(job, allocation, log)
                finally:
                    await log.close()
            finally:
                await self._io(stack.close)
            if job.cancelled.is_set():
                reason = "Job cancelled."
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            logger.exception("SSH job %s failed", job.job_id)
            try:
                await self._io(self._append_failure, job, reason)
            except OSError:
                logger.exception("Could not append failure to SSH job log")
        finally:
            await self._io(job.finish, reason)
            self.allocations.pop(job, None)
            self.running.pop(job, None)
            self.changed.set()

    @staticmethod
    def _append_failure(job: SSHJob, reason: str) -> None:
        if job.log_path is not None:
            with job.log_path.open("a") as output:
                output.write(f"\nSSH job failed: {reason}\n")

    @staticmethod
    def _open_log(job: SSHJob, stack: contextlib.ExitStack) -> BinaryIO:
        if job.log_path is None:
            raise ExecutionError("SSH job has no log path.")
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        output = stack.enter_context(job.log_path.open("ab", buffering=0))
        stack.enter_context(job.workspace.streaming_job_log(job.log_path))
        return output

    async def _connect(self, host: str) -> Any:
        sdk = _sdk()
        options: dict[str, Any] = {
            "connect_timeout": self.config.connect_timeout,
            "login_timeout": self.config.connect_timeout,
            "keepalive_interval": 5,
            "keepalive_count_max": 3,
            "agent_forwarding": False,
        }
        if self.config.ssh_config is not None:
            options["config"] = self.config.ssh_config
        if self.config.known_hosts is not None:
            options["known_hosts"] = self.config.known_hosts
        try:
            async with asyncio.timeout(self.config.connect_timeout):
                return await sdk.connect(host, **options)
        except (sdk.Error, OSError) as exc:
            raise ExecutionError(f"Could not connect to SSH host {host}: {exc}") from exc

    async def _connect_serially(self, host: str) -> Any:
        # Avoid a burst of unauthenticated connections to a single SSH daemon.
        async with self.connect_locks.setdefault(host, asyncio.Lock()):
            return await self._connect(host)

    def _reserve_connection(self, host: str) -> _Connection:
        # Each job opens at most one channel on a host at a time. Reserving
        # through both phases bounds sessions while preserving connection reuse.
        pool = self.connections.setdefault(host, [])
        connection = next((item for item in pool if item.users < _JOBS_PER_CONNECTION), None)
        if connection is None:
            connection = _Connection(asyncio.create_task(self._connect_serially(host)))
            pool.append(connection)
        connection.users += 1
        return connection

    async def _run_group(self, job: SSHJob, allocation: _Allocation, output: _BufferedLog) -> None:
        worker = self.config.workers[allocation.worker]
        stop = asyncio.Event()
        connections: list[Any] = []
        reservations = [self._reserve_connection(host) for host in worker.hosts]
        try:
            results = await asyncio.gather(
                *(asyncio.shield(item.task) for item in reservations), return_exceptions=True
            )
            connections = [value for value in results if not isinstance(value, BaseException)]
            for result in results:
                if isinstance(result, BaseException):
                    self.unavailable.add(allocation.worker)
                    raise result
            if job.cancelled.is_set():
                return
            addresses = worker.addresses or [connection.get_extra_info("host") for connection in connections]
            environments = [
                resource_environment(allocation.cpus, worker.accelerator_type, allocation.devices)
                | {
                    "MISEN_NODE_RANK": str(rank),
                    "MISEN_NODE_IPS": "\n".join(addresses),
                    "MISEN_JOB_LOG_CAPTURED": "1",
                }
                for rank in range(len(connections))
            ]

            async def phase(rank: int, command: str, duration: int, *, preparation: bool) -> None:
                try:
                    deadline = duration + _LEASE_SECONDS + _STOP_GRACE_SECONDS + self.config.connect_timeout + 10
                    async with asyncio.timeout(deadline):
                        code = await self._remote(
                            connections[rank],
                            command,
                            environments[rank],
                            allocation,
                            output,
                            rank,
                            duration=duration,
                            cancelled=job.cancel_event,
                            stop=stop,
                        )
                    if code != 0 and not (stop.is_set() or job.cancelled.is_set()):
                        label = "preparation" if preparation else "execution"
                        raise ExecutionError(f"SSH rank {rank} exited with status {code} during {label}.")
                    if rank == 0 and not preparation and job.work_unit.uses_dask_client:
                        stop.set()
                except BaseException:
                    stop.set()
                    raise

            async def run_phase(command: str, duration: int, *, preparation: bool) -> None:
                results = await asyncio.gather(
                    *(phase(rank, command, duration, preparation=preparation) for rank in range(len(connections))),
                    return_exceptions=True,
                )
                for result in results:
                    if isinstance(result, BaseException):
                        raise result

            await output.write(f"Preparing SSH hosts: {', '.join(worker.hosts)}\n".encode())
            await run_phase(job.commands[0], self.config.startup_timeout, preparation=True)
            if not job.cancelled.is_set():
                job._state = "running"  # noqa: SLF001
                await output.write(b"Environment ready; executing WorkUnit.\n")
                await run_phase(job.commands[1], job.resources["time"] * 60, preparation=False)
        except ExecutionError:
            raise
        except Exception as exc:
            # Do not reuse capacity when remote cleanup cannot be confirmed.
            self.unavailable.add(allocation.worker)
            raise ExecutionError(f"SSH connection or protocol failed; worker disabled for this session: {exc}") from exc
        finally:
            for reservation in reservations:
                reservation.users -= 1

    async def _remote(  # noqa: PLR0917 -- one protocol invocation
        self,
        connection: Any,
        command: str,
        environment: dict[str, str],
        allocation: _Allocation,
        output: _BufferedLog,
        rank: int,
        *,
        duration: int,
        cancelled: asyncio.Event,
        stop: asyncio.Event,
    ) -> int:
        worker = self.config.workers[allocation.worker]
        process = await asyncio.wait_for(
            connection.create_process(agent_command(), encoding=None), self.config.connect_timeout
        )
        message = {
            "command": command,
            "env": environment,
            "cpus": allocation.cpus,
            "capacity": worker.cpus,
            "timeout": duration,
            "lease": _LEASE_SECONDS,
            "grace": _STOP_GRACE_SECONDS,
        }

        async def drain(stream: Any) -> None:
            while data := await stream.read(65536):
                await output.write(f"[rank {rank}] ".encode() + data)

        readers = [asyncio.create_task(drain(stream)) for stream in (process.stdout, process.stderr)]
        exited = asyncio.create_task(process.wait_closed())
        signals = [asyncio.create_task(event.wait()) for event in (cancelled, stop)]

        async def control() -> None:
            process.stdin.write(json.dumps(message).encode() + b"\n")
            await process.stdin.drain()
            while not exited.done():
                if cancelled.is_set() or stop.is_set():
                    process.stdin.write(b"cancel\n")
                    await process.stdin.drain()
                    await asyncio.wait_for(asyncio.shield(exited), _LEASE_SECONDS + _STOP_GRACE_SECONDS + 5)
                    return
                process.stdin.write(b"ping\n")
                await process.stdin.drain()
                await asyncio.wait([exited, *signals], timeout=_HEARTBEAT_SECONDS, return_when=asyncio.FIRST_COMPLETED)

        controls = asyncio.create_task(control())
        log_failed = asyncio.create_task(output.failed.wait())
        tasks = [exited, controls, log_failed, *readers, *signals]
        try:
            async with asyncio.timeout(duration + _LEASE_SECONDS + _STOP_GRACE_SECONDS + 5):
                pending = {exited, controls, log_failed, *readers}
                while not exited.done():
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        task.result()
                    output.check()
                await exited
                await asyncio.gather(*readers, controls)
                output.check()
            if process.returncode is None:
                raise OSError("SSH stream closed without an exit status.")
            return int(process.returncode)
        finally:
            process.close()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _shutdown(self) -> None:
        self.closed = True
        for job in self.jobs:
            if job.state() not in _TERMINAL:
                job.cancelled.set()
                job.cancel_event.set()
                if job.state() == "pending":
                    await self._io(job.finish, "SSH executor closed before this job started.")
        self.changed.set()
        await asyncio.gather(*list(self.running.values()), return_exceptions=True)
        await asyncio.wrap_future(self.scheduler)
        connections = await asyncio.gather(
            *(item.task for pool in self.connections.values() for item in pool), return_exceptions=True
        )
        for connection in connections:
            if not isinstance(connection, BaseException):
                connection.close()
        await asyncio.gather(
            *(connection.wait_closed() for connection in connections if not isinstance(connection, BaseException)),
            return_exceptions=True,
        )

    def close(self) -> None:
        if self.loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self._shutdown(), self.loop).result()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()
        atexit.unregister(self._exit_close)


class SSHExecutor(Executor[SSHJob]):
    """Schedule work on explicitly configured SSH hosts sharing a workspace.

    Requires Linux, Bash, and Python 3 on each host, and either a DiskWorkspace
    mounted at the same absolute path everywhere or a CloudWorkspace with
    independent credentials on every host. The submitting process must remain
    alive. Use ``close()`` or a context manager to stop unfinished work; the
    remote lease terminates process groups after loss of the submitting process.
    Machines are never provisioned, shut down, or otherwise managed.
    """

    workers: Annotated[
        list[SSHWorker],
        PrimitiveConstructorSpec(
            nargs=1,
            metavar="JSON",
            instance_from_str=_workers_from_cli,
            is_instance=lambda value: isinstance(value, list),
            str_from_instance=_workers_to_cli,
        ),
    ] = msgspec.field(default_factory=list)
    ssh_config: str | None = None
    known_hosts: str | None = None
    connect_timeout: Annotated[int, msgspec.Meta(ge=1)] = 30
    startup_timeout: Annotated[int, msgspec.Meta(ge=1)] = 600
    dask_startup_timeout: Annotated[int, msgspec.Meta(ge=1)] = DEFAULT_DASK_STARTUP_TIMEOUT
    dask_scheduler_port: Annotated[int, msgspec.Meta(ge=1024, le=65535)] = DEFAULT_DASK_SCHEDULER_PORT
    _config_validation_errors: ClassVar[tuple[type[Exception], ...]] = (ValueError,)

    def __post_init__(self) -> None:
        """Validate configuration without importing AsyncSSH or opening connections."""
        _validate(self)
        if not self.workers:
            raise ValueError("SSHExecutor requires at least one worker group.")
        if not self.snapshot or self.prewarm_envs:
            raise ValueError("SSHExecutor requires snapshot=True and prewarm_envs=False.")
        hosts = [host for worker in self.workers for host in worker.hosts]
        if len(set(hosts)) != len(hosts):
            raise ValueError("SSH worker groups must use distinct hosts.")
        for name in ("ssh_config", "known_hosts"):
            if getattr(self, name) == "":
                raise ValueError(f"{name} cannot be empty.")
            if getattr(self, name) is not None:
                setattr(self, name, str(Path(getattr(self, name)).expanduser()))
        self._session: _Session | None = None
        self._lock = threading.RLock()
        self._jobs: dict[str, SSHJob] = {}

    def _validate_submission(
        self,
        *,
        work_graph: DependencyGraph[WorkUnit],
        pending_work_units: Sequence[WorkUnit],
        workspace: Workspace,
    ) -> None:
        del work_graph
        _sdk()
        from misen.workspaces.memory import InMemoryWorkspace

        if isinstance(workspace, InMemoryWorkspace):
            raise ConfigError("SSHExecutor requires a shared filesystem or cloud workspace, not InMemoryWorkspace.")
        if workspace.bootstrap_transport() is not None and workspace.get_temp_dir().is_absolute():
            raise ConfigError("SSH cloud workers require a relative cache_dir, such as '.cache/misen'.")
        for unit in pending_work_units:
            if not any(worker.fits(unit.resources) for worker in self.workers):
                raise ConfigError(f"No SSH worker group can satisfy {unit.resources} for {unit.root}.")

    def _dispatch(
        self,
        work_unit: WorkUnit,
        dependencies: set[SSHJob],
        workspace: Workspace,
        snapshot: ProjectSnapshot | None,
    ) -> SSHJob:
        if snapshot is None:
            raise SubmissionError("SSHExecutor requires a project snapshot.")
        with self._lock:
            key = TaskHash.from_object(
                (workspace, snapshot.snapshot_key, work_unit.root.task_hash(), work_unit.resources)
            ).b32()
            existing = self._jobs.get(key)
            if existing is not None and existing.state() not in _TERMINAL:
                return existing
            job_id, argv, env, log_path = snapshot.prepare_job(work_unit, workspace, reuse_env=True)
            if self._session is None:
                self._session = _Session(self)
            job = SSHJob(
                work_unit,
                job_id,
                log_path,
                workspace,
                dependencies,
                self._commands(argv, env, work_unit),
                self._session,
            )
            asyncio.run_coroutine_threadsafe(self._session.add(job), self._session.loop).result()
            self._jobs[key] = job
            return job

    def _commands(self, argv: list[str], env: dict[str, str], unit: WorkUnit) -> tuple[str, str]:
        resources = unit.resources
        env = env | {
            "MISEN_SSH_RESOURCES": msgspec.json.encode(resources).decode(),
            "MISEN_WORKER_PREFLIGHT": "misen.utils.ssh_runtime:worker_preflight",
        }
        payload = shlex.join(["env", *(f"{key}={value}" for key, value in env.items()), *argv])
        command = (
            managed_ranked_cluster_script(
                argv,
                environment=env,
                workers=resources["nodes"],
                cpus=resources["cpus"],
                memory_gib=resources["memory"],
                startup_timeout=self.dask_startup_timeout,
                node_rank_env="MISEN_NODE_RANK",
                node_ips_env="MISEN_NODE_IPS",
                scheduler_port=self.dask_scheduler_port,
            )
            if unit.uses_dask_client
            else payload
        )
        if resources["nodes"] > 1 and not unit.uses_dask_client:
            command = 'if [[ "${MISEN_NODE_RANK:-0}" != "0" ]]; then exit 0; fi\n' + command
        return f"exec env MISEN_PREPARE_ONLY=1 {payload}", command

    def close(self) -> None:
        """Cancel outstanding jobs, await remote cleanup, and stop SSH I/O."""
        with self._lock:
            if self._session is not None:
                self._session.close()
                self._session = None
                self._jobs.clear()

    def __enter__(self) -> Self:
        """Keep the executor alive for this context."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Stop owned remote processes on context exit."""
        self.close()
