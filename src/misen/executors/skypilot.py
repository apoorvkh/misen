"""Persistent subprocess workers on SkyPilot VMs, with a process-owned local API."""

# Keep exception messages next to their raises.
# ruff: noqa: EM101, EM102, TRY003

from __future__ import annotations

import atexit
import base64
import contextlib
import importlib
import inspect
import json
import logging
import math
import os
import queue
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Self, TypeVar, cast

import cloudpickle
import msgspec
from processkit import Command, SupervisionSession, Supervisor
from tyro.constructors import PrimitiveConstructorSpec

from misen.exceptions import ConfigError, ExecutionError, MisenError, StatusQueryError, StorageError, SubmissionError
from misen.executor import Executor, Job, JobState, _JobRecord
from misen.task_metadata import AcceleratorType, Resources
from misen.utils.dask_runtime import (
    DEFAULT_DASK_SCHEDULER_PORT,
    DEFAULT_DASK_STARTUP_TIMEOUT,
    managed_ranked_cluster_script,
)
from misen.utils.hashing import TaskHash
from misen.utils.resource_env import resource_environment
from misen.utils.runtime_events import runtime_event

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from misen.utils.graph import DependencyGraph
    from misen.utils.snapshot import ProjectSnapshot
    from misen.utils.work_unit import WorkUnit
    from misen.workspace import Workspace

logger = logging.getLogger(__name__)


_T = TypeVar("_T")


def _optional(read: Callable[[], _T]) -> _T | None:
    """Treat missing coordination files as absent, preserving other storage errors."""
    with contextlib.suppress(FileNotFoundError):
        return read()
    return None


def _atomic_write(path: Path, data: bytes) -> None:
    """Publish private local coordination files without partial reads."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.touch(mode=0o600)
    temp.write_bytes(data)
    temp.replace(path)


_PositiveInt = Annotated[int, msgspec.Meta(ge=1)]
_Name = Annotated[str, msgspec.Meta(pattern=r"^\S(?:[\s\S]*\S)?\Z")]
_NonBlank = Annotated[str, msgspec.Meta(pattern=r"\S")]


def _validate_fields(config: msgspec.Struct) -> None:
    """Apply declared constraints to direct constructors as well as decoded config."""
    for item in msgspec.structs.fields(config):
        try:
            value = msgspec.convert(getattr(config, item.name), type=item.type)
        except msgspec.ValidationError as exc:
            raise ValueError(f"{item.name}: {exc}") from exc
        setattr(config, item.name, value)


class SkyPilotWorker(msgspec.Struct, forbid_unknown_fields=True):
    """Repeatable worker allocation; CPU, RAM, and devices are per node.

    ``accelerators`` contains one concrete SkyPilot model and a whole-device
    count. Each entry is a worker type, not an ordered fallback or a named pool.
    ``cpus`` and ``memory`` are scheduling budgets as well as provisioning
    minimums; leave headroom for the OS and SkyPilot when pinning an instance.
    """

    cpus: _PositiveInt
    memory: _PositiveInt
    infra: _NonBlank = "aws"
    nodes: _PositiveInt = 1
    accelerators: Annotated[dict[_Name, _PositiveInt], msgspec.Meta(max_length=1)] = msgspec.field(default_factory=dict)
    accelerator_type: AcceleratorType = "cuda"
    max_workers: _PositiveInt = 1
    instance_type: _NonBlank | None = None
    use_spot: bool = False
    image_id: _NonBlank | None = None
    disk_size: _PositiveInt | None = None
    max_hourly_cost: float | None = None

    def __post_init__(self) -> None:
        """Validate declared budgets and reject nonfinite prices."""
        _validate_fields(self)
        self.infra = self.infra.strip()
        if self.max_hourly_cost is not None and _number(self.max_hourly_cost) is None:
            raise ValueError("Worker max_hourly_cost must be finite and positive.")


class WorkerOffering(msgspec.Struct):
    """A provisionable option with verified or explicitly declared capacity."""

    worker_index: int
    worker: SkyPilotWorker
    resource_args: dict[str, Any]
    device_memory: float | None = None
    hourly_cost: float | None = None

    def fits(self, request: Resources) -> bool:
        """Check per-node requirements, preserving whole allocation topology."""
        worker = self.worker
        if request["nodes"] != worker.nodes or request["cpus"] > worker.cpus or request["memory"] > worker.memory:
            return False
        if not request["accelerators"]:
            return True
        minimum = request["accelerator_memory"]
        return (
            request["accelerators"] <= sum(worker.accelerators.values())
            and request["accelerator_type"] == worker.accelerator_type
            and (
                minimum is None
                or (
                    request["accelerator_type"] in {"cuda", "rocm", "xpu"}
                    and self.device_memory is not None
                    and self.device_memory >= minimum
                )
            )
        )


def _number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
        return float(value)
    return None


def resolve_workers(
    sky: Any, workers: Sequence[SkyPilotWorker], memory_overrides: dict[str, int]
) -> list[WorkerOffering]:
    """Resolve GPU variants before launch and pin catalog-backed offerings.

    SkyPilot's SDK reports ``device_memory`` in GiB, already per device.
    Explicit metadata is a fallback, never permission to exceed a known
    catalog capacity. Unknown memory remains ineligible for constrained work.

    Raises:
        ConfigError: If catalog/resource validation fails or a worker has no
            matching public-cloud offering.
    """
    offerings: list[WorkerOffering] = []
    for index, worker in enumerate(workers):
        args: dict[str, Any] = {
            "infra": worker.infra,
            "cpus": f"{worker.cpus}+",
            "memory": f"{worker.memory}+",
            "use_spot": worker.use_spot,
        }
        for name in ("instance_type", "image_id", "disk_size", "max_hourly_cost"):
            if (value := getattr(worker, name)) is not None:
                args[name] = value
        if not worker.accelerators:
            offerings.append(WorkerOffering(index, worker, args))
            continue
        args["accelerators"] = worker.accelerators
        model, count = next(iter(worker.accelerators.items()))
        fallback = _number(memory_overrides.get(model))
        parts = worker.infra.split("/")
        # Attached infrastructure does not have a public instance catalog.
        if parts[0] in {"k8s", "kubernetes", "ssh", "slurm"}:
            offerings.append(WorkerOffering(index, worker, args, fallback))
            continue
        try:
            catalog = sky.get(
                # The top-level sky.list_accelerators is a synchronous local
                # catalog helper. Use the API SDK for asynchronous catalog
                # requests through this submission's local server.
                sky.client.sdk.list_accelerators(
                    # CPU rows lack GpuInfo; including them makes some
                    # SkyPilot catalogs drop memory metadata for every GPU.
                    gpus_only=worker.accelerator_type != "tpu",
                    name_filter=f"^{re.escape(model)}$",
                    quantity_filter=count,
                    clouds=parts[0],
                    all_regions=True,
                    require_price=False,
                )
            )
        except Exception as exc:
            raise ConfigError(
                f"Could not resolve worker {index} accelerator {model!r} through SkyPilot: {exc}"
            ) from exc
        records = catalog.get(model, [])
        matched = False
        for record in records:

            def field(name: str, record: Any = record) -> Any:
                return record.get(name) if isinstance(record, dict) else getattr(record, name, None)

            region, instance = field("region"), field("instance_type")
            if len(parts) > 1 and parts[1] and region != parts[1]:
                continue
            if worker.instance_type is not None and instance != worker.instance_type:
                continue
            if field("accelerator_count") != count:
                continue
            if (cpus := _number(field("cpu_count"))) is not None and cpus < worker.cpus:
                continue
            if (memory := _number(field("memory"))) is not None and memory < worker.memory:
                continue
            cost = _number(field("spot_price" if worker.use_spot else "price"))
            if worker.max_hourly_cost is not None and cost is not None and cost > worker.max_hourly_cost:
                continue
            capacity = _number(field("device_memory"))
            if capacity is None:
                capacity = fallback
            elif fallback is not None:
                capacity = min(capacity, fallback)
            # Memory metadata is only useful when the launch is pinned to the
            # exact offering it describes. Preserve an explicitly chosen zone.
            concrete = dict(args)
            if instance is not None:
                concrete["instance_type"] = instance
            if region and len(parts) < 3:  # noqa: PLR2004 -- cloud/region/zone
                concrete["infra"] = f"{parts[0]}/{region}"
            if instance is None:
                capacity = fallback
            offerings.append(WorkerOffering(index, worker, concrete, capacity, cost))
            matched = True
        if not matched:
            if fallback is None or records:
                raise ConfigError(f"No catalog offering matches worker {index} ({worker.infra}, {model}:{count}).")
            offerings.append(WorkerOffering(index, worker, args, fallback))
    try:
        if sky.server.common.is_api_server_local():
            for offering in offerings:
                sky.Resources(**offering.resource_args).validate()
    except Exception as exc:
        raise ConfigError(f"Invalid SkyPilot worker resources: {exc}") from exc
    return offerings


def _worker_preflight() -> None:
    """Check actual assigned GPUs inside the task environment, on every rank."""
    verify_accelerators(msgspec.json.decode(os.environ.pop("MISEN_SKYPILOT_RESOURCES"), type=Resources))


def verify_accelerators(resources: Resources) -> None:
    """Verify assigned device memory on the worker before entering user code.

    Whole GPUs are reserved by the scheduler. Capacity is total memory of
    each assigned device, not a claim about instantaneous free memory.
    """
    count = resources["accelerators"]
    minimum = resources["accelerator_memory"]
    if count and minimum is not None:
        backend = resources["accelerator_type"]
        if backend == "cuda":
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if not visible or len(visible.split(",")) < count:
                raise ConfigError("SkyPilot did not expose the requested CUDA devices.")
            try:
                result = subprocess.run(  # noqa: S603 -- fixed executable, device IDs are a separate argument
                    [
                        shutil.which("nvidia-smi") or "/usr/bin/nvidia-smi",
                        "--query-gpu=memory.total",
                        "--format=csv,noheader,nounits",
                        "-i",
                        visible,
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                )
                capacities = [float(line.strip()) / 1024 for line in result.stdout.splitlines() if line.strip()]
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                raise ConfigError(f"Could not verify allocated GPU memory: {exc}") from exc
        elif backend in {"rocm", "xpu"}:
            try:
                torch = importlib.import_module("torch")

                runtime = torch.cuda if backend == "rocm" else torch.xpu
                capacities = [runtime.get_device_properties(i).total_memory / 2**30 for i in range(count)]
            except Exception as exc:
                raise ConfigError(
                    f"Could not verify {backend} device memory through the task environment's PyTorch: {exc}"
                ) from exc
        else:
            raise ConfigError(
                f"Runtime verification of {backend} device memory is not supported by reusable SkyPilot workers."
            )
        if len(capacities) < count or any(not math.isfinite(value) or value < minimum for value in capacities):
            raise ConfigError(
                f"Assigned GPUs provide {capacities} GiB/device; "
                f"the WorkUnit requires {count} devices of {minimum} GiB."
            )


STATE_FILE = "controller-state.json"


_TERMINAL = {"done", "failed"}


class WorkSpec(msgspec.Struct):
    """Immutable command and dependency description, without user objects."""

    job_id: str
    command: str
    resources: Resources
    dependencies: list[str]
    submission_id: str = ""
    environment_key: str = ""
    duration_key: str = ""
    log_path: str = ""


def _event(directory: Path, message: str) -> None:
    """Relay lifecycle events to the submitting process's runtime UI."""
    try:
        with (directory / "events.jsonl").open("a") as output:
            output.write(json.dumps(message) + "\n")
        (directory / "logs").mkdir(exist_ok=True)
        with (directory / "logs" / "events.log").open("a") as output:
            output.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
    except OSError:
        logger.exception("Could not record SkyPilot progress: %s", message)


def _sky_wait(sky: Any, request: Any, path: Path) -> Any:
    """Retain SDK request output and exceptions, including failed provisioning."""
    with path.open("a", buffering=1) as output:
        try:
            return sky.stream_and_get(request, output_stream=output)
        except BaseException:
            traceback.print_exc(file=output)
            raise


class _JobLogs:
    """One controller owns job-log uploads, including bootstrap and all ranks."""

    def __init__(self, workspace: Workspace, directory: Path) -> None:
        self.workspace, self.directory = workspace, directory
        self.paths: dict[str, Path] = {}
        self.streams: dict[str, contextlib.ExitStack] = {}
        self.offsets: dict[Path, int] = {}
        self.copied: dict[str, int] = {}
        self.shared: Path | None = None
        self.last_sync = 0.0
        (directory / "logs").mkdir(exist_ok=True)

    def add(self, specs: list[WorkSpec]) -> None:
        for spec in specs:
            if not spec.log_path or spec.job_id in self.paths:
                continue
            path = Path(spec.log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if self.shared is None:
                self.shared = path.parent / f"{self.directory.name}_skypilot.log"
                self._open("", self.shared)
            self.paths[spec.job_id] = path
            path.touch(mode=0o600)
            path.write_text(f"SkyPilot job {spec.job_id}; session log: {self.shared}\n")

    def _open(self, key: str, path: Path) -> None:
        path.touch(mode=0o600)
        self.streams[key] = contextlib.ExitStack()
        self.streams[key].enter_context(self.workspace.streaming_job_log(path))

    def sync(self) -> None:
        """Collect native files incrementally; retain the originals for diagnosis."""
        if self.shared is None:
            return
        with self.shared.open("ab") as output:
            for path in sorted((self.directory / "logs").rglob("*.log")):
                with path.open("rb") as source:
                    offset = self.offsets.get(path, 0)
                    size = source.seek(0, 2)
                    source.seek(offset if size >= offset else 0)
                    if source.tell() < size:
                        output.write(f"\n--- {path.relative_to(self.directory)} ---\n".encode())
                        shutil.copyfileobj(source, output)
                    self.offsets[path] = source.tell()
        self.last_sync = time.monotonic()

    def write(self, job_id: str, data: bytes) -> None:
        if job_id not in self.paths:
            return
        # Keep upload threads bounded by running work, not the size of the DAG.
        if job_id not in self.streams:
            self._open(job_id, self.paths[job_id])
        with self.paths[job_id].open("ab") as output:
            if self.shared is not None:
                with self.shared.open("rb") as source:
                    source.seek(self.copied.get(job_id, 0))
                    shutil.copyfileobj(source, output)
                    self.copied[job_id] = source.tell()
            output.write(data)

    def finish(self, job_id: str, message: str) -> None:
        self.sync()
        self.write(job_id, f"\nSkyPilot: {message}\n".encode())
        if stream := self.streams.pop(job_id, None):
            stream.close()

    def close(self) -> None:
        with contextlib.ExitStack() as cleanup:
            for stream in self.streams.values():
                cleanup.callback(stream.close)
            self.sync()
        self.streams.clear()


class WorkState(msgspec.Struct):
    """One logical job's durable state and resource reservation."""

    state: str = "pending"
    reason: str | None = None
    cluster: str | None = None
    cpus: list[int] = msgspec.field(default_factory=list)
    gpus: list[int] = msgspec.field(default_factory=list)
    started: float | None = None


class WorkerState(msgspec.Struct, dict=True):
    """One owned cluster; only declared fields enter recovery checkpoints."""

    name: str
    offering: int
    state: str = "provisioning"
    idle_since: float | None = None

    def __post_init__(self) -> None:
        self.started = time.monotonic()
        self.future: Future[Any] | None = None
        self.connections: list[_Connection] = []
        self.ips: list[str] = []
        self.prepared: set[str] = set()
        self.preparing: tuple[str, str] | None = None
        self.completions: dict[str, dict[int, int]] = {}

    def send(self, message: dict[str, Any]) -> None:
        for connection in self.connections:
            connection.send(message)

    def close(self) -> bool:
        """Close every node, reporting whether all agents acknowledged exit."""
        acknowledgements = [c.close() for c in self.connections]
        return bool(acknowledgements) and all(acknowledgements)


class ControllerState(msgspec.Struct):
    """Recoverable graph status shared with lightweight client job handles."""

    jobs: dict[str, WorkState]
    workers: list[WorkerState] = msgspec.field(default_factory=list)
    sequence: int = 0
    failure: str | None = None


def _worker_main() -> None:
    """Run the stdlib-only agent sent over SSH; EOF/lease loss kills its children."""
    import base64
    import json
    import os
    import queue
    import signal
    import subprocess
    import sys
    import threading
    import time
    from pathlib import Path

    inbox: Any = queue.Queue()

    def read() -> None:
        for line in sys.stdin:
            inbox.put(json.loads(line))
        inbox.put(None)

    threading.Thread(target=read, daemon=True).start()
    jobs = {}
    seen = set()
    last_ping = time.monotonic()
    heartbeat = 0.0
    lease = Path(sys.argv[1]).expanduser()
    cpus = sorted(os.sched_getaffinity(0))

    def emit(**event: Any) -> None:
        print(json.dumps(event), flush=True)  # noqa: T201 -- SSH wire protocol

    def drain(job_id: str, reader: Any, *, complete: bool) -> None:
        while data := reader.read(65536):
            emit(kind="log", job_id=job_id, data=base64.b64encode(data).decode())
            if not complete:
                break

    def kill(process: subprocess.Popen) -> None:
        try:  # noqa: SIM105 -- self-contained remote agent
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    emit(kind="ready", cpus=cpus)
    try:
        while time.monotonic() - last_ping < 60:  # noqa: PLR2004 -- lease seconds
            message: dict[str, Any] | None
            try:
                message = inbox.get(timeout=0.1)
            except queue.Empty:
                message = {"kind": "idle"}
            if message is None:
                break
            if message["kind"] != "idle":
                last_ping = time.monotonic()
                lease.touch()
                kind = message["kind"]
                if kind == "run":
                    job_id = message["job_id"]
                    if job_id in seen:
                        raise RuntimeError("Duplicate dispatch: " + job_id)
                    seen.add(job_id)
                    assigned = [cpus[i] for i in message["cpus"]]
                    # taskset applies affinity before Bash or user code can run.
                    argv = ["taskset", "-c", ",".join(map(str, assigned)), "bash", "-c", message["command"]]
                    path = Path(message["log"]).expanduser()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    log = path.open("wb", buffering=0)
                    process = subprocess.Popen(  # noqa: S603 -- authenticated controller command
                        argv,
                        env=os.environ | message["env"],
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    jobs[job_id] = (process, log, path.open("rb"), time.monotonic())
                    emit(kind="started", job_id=job_id, pid=process.pid)
                elif kind == "cancel" and message["job_id"] in jobs:
                    kill(jobs[message["job_id"]][0])
            for job_id, (process, log, reader, started) in list(jobs.items()):
                code = process.poll()
                if code is not None:
                    kill(process)
                # Read before the completion event so the controller can finalize
                # the complete log before advertising a terminal job state.
                drain(job_id, reader, complete=code is not None)
                if code is not None:
                    log.close()
                    reader.close()
                    del jobs[job_id]
                    emit(kind="finished", job_id=job_id, code=code, seconds=time.monotonic() - started)
            if time.monotonic() - heartbeat > 5:  # noqa: PLR2004 -- heartbeat seconds
                emit(kind="heartbeat")
                heartbeat = time.monotonic()
    finally:
        for job_id, (process, log, reader, _) in jobs.items():
            kill(process)
            process.wait()
            drain(job_id, reader, complete=True)
            log.close()
            reader.close()


class _Connection:
    """One authenticated SSH stream per node; readers feed a shared event queue."""

    def __init__(self, argv: list[str], events: Any, name: str, rank: int, directory: Path) -> None:
        self.last_seen = time.monotonic()
        self.ready = False
        self.rank = rank
        self.log = (directory / f"{name}-{rank}.log").open("ab")
        self.process = subprocess.Popen(  # noqa: S603 -- SDK-generated SSH argv
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1
        )

        assert self.process.stdin is not None and self.process.stdout is not None  # noqa: S101, PT018
        self.stdin, self.stdout = self.process.stdin, self.process.stdout

        def read() -> None:
            try:
                for line in self.stdout:
                    event = json.loads(line)
                    self.last_seen = time.monotonic()
                    events.put((name, rank, event))
            except (OSError, ValueError) as exc:
                events.put((name, rank, {"kind": "lost", "reason": str(exc)}))
            finally:
                events.put((name, rank, {"kind": "lost", "reason": "SSH stream closed"}))

        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()

    def send(self, message: dict[str, Any]) -> None:
        self.stdin.write(json.dumps(message) + "\n")
        self.stdin.flush()

    def close(self) -> bool:
        with contextlib.suppress(OSError):
            self.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.reader.join(timeout=5)
        self.log.close()
        return self.process.returncode == 0


def _connect(handle: Any, name: str, events: Any, directory: Path) -> tuple[list[_Connection], list[str]]:
    """Use the SDK's authenticated SSH runners, never public worker ports."""
    from sky.utils.command_runner import SSHCommandRunner, SshMode

    connections = []
    try:
        for rank, runner in enumerate(handle.get_command_runners()):
            if not isinstance(runner, SSHCommandRunner):
                raise ConfigError("Persistent SkyPilot workers require SSH command runners.")  # noqa: TRY301 -- close already-connected nodes
            argv = runner.ssh_base_command(ssh_mode=SshMode.NON_INTERACTIVE, port_forward=None, connect_timeout=30)
            source = "from __future__ import annotations\n" + inspect.getsource(_worker_main) + "\n_worker_main()"
            argv += [shlex.join(["python3", "-u", "-c", source, f"~/.sky/{name}.lease"])]
            connections.append(_Connection(argv, events, name, rank, directory))
        return connections, handle.internal_ips()
    except BaseException:
        for connection in connections:
            connection.close()
        raise


@dataclass(eq=False, repr=False, kw_only=True)
class Controller:
    """Session scheduler: durable ownership, event-driven dispatch, bounded lookahead."""

    sky: Any
    workspace: Workspace
    submission_id: str
    specs: InitVar[list[WorkSpec]]
    offerings: list[WorkerOffering]
    config: SkyPilotExecutor
    directory: Path
    stop_requested: Callable[[], bool]
    log_capture: InitVar[_JobLogs | None] = None

    def __post_init__(self, specs: list[WorkSpec], log_capture: _JobLogs | None) -> None:
        self.logs = log_capture or _JobLogs(self.workspace, self.directory)
        self.work: dict[str, WorkSpec] = {}
        self.graphs: dict[str, list[str]] = {}
        self.state = ControllerState({})
        self.threads = ThreadPoolExecutor(max_workers=self.config.max_workers + 4)
        self.events: Any = queue.Queue()
        self.durations: dict[str, float] = {}
        self.last_ping = 0.0
        self.saved: dict[str, bytes] = {}
        self.add_graph(self.submission_id, specs)

    def add_graph(self, submission_id: str, specs: list[WorkSpec]) -> None:
        """Validate the whole graph before accepting any work onto the pool."""
        for spec in specs:
            if not any(o.fits(spec.resources) for o in self.offerings):
                raise ConfigError(
                    f"No worker type satisfies WorkUnit {spec.job_id}: {spec.resources}. GPU memory is per device."
                )
            if spec.job_id in self.work:
                raise ConfigError(f"Duplicate WorkUnit identity: {spec.job_id}")
        self.graphs[submission_id] = [s.job_id for s in specs]
        self.logs.add(specs)
        _event(self.directory, f"SkyPilot accepted {len(specs)} WorkUnits for graph {submission_id}.")
        for spec in specs:
            spec.submission_id = submission_id
            self.work[spec.job_id] = spec
            self.state.jobs[spec.job_id] = WorkState()
        self.save()

    def save(self) -> None:
        """Publish changed checkpoints only, locally and durably in the workspace."""
        for submission_id, jobs in self.graphs.items():
            state = ControllerState(
                {j: self.state.jobs[j] for j in jobs}, self.state.workers, self.state.sequence, self.state.failure
            )
            data = msgspec.json.encode(state)
            if self.saved.get(submission_id) != data:
                self.workspace.put_job_file(submission_id, STATE_FILE, data)
                _atomic_write(self.directory / f"{submission_id}.json", data)
                self.saved[submission_id] = data

    def _active(self, worker: WorkerState) -> list[str]:
        return [
            j for j, state in self.state.jobs.items() if state.cluster == worker.name and state.state not in _TERMINAL
        ]

    def _fits(self, worker: WorkerState, spec: WorkSpec, reserved: set[str], *, speculative: bool) -> bool:
        offering = self.offerings[worker.offering]
        active = reserved | (set(self._active(worker)) if worker.state == "ready" and not speculative else set())
        return (
            offering.fits(spec.resources)
            and not (offering.worker.nodes > 1 and active)
            and all(
                sum(self.work[j].resources[k] for j in active) + spec.resources[k] <= capacity
                for k, capacity in (
                    ("cpus", offering.worker.cpus),
                    ("memory", offering.worker.memory),
                    ("accelerators", sum(offering.worker.accelerators.values())),
                )
            )
        )

    def _choices(self, spec: WorkSpec, *, excluding: WorkerState | None = None) -> list[int]:
        """Provisionable offerings within both limits, counting draining workers."""
        counts = Counter(
            self.offerings[w.offering].worker_index
            for w in self.state.workers
            if w.state != "down" and w is not excluding
        )
        if sum(counts.values()) >= self.config.max_workers:
            return []
        return sorted(
            (
                i
                for i, o in enumerate(self.offerings)
                if o.fits(spec.resources) and counts[o.worker_index] < o.worker.max_workers
            ),
            key=lambda i: (
                bool(self.offerings[i].worker.accelerators),
                self.offerings[i].hourly_cost or float("inf"),
                self.offerings[i].worker.cpus,
            ),
        )

    def _finish(self, job_id: str, state: str, reason: str | None = None) -> None:
        job = self.state.jobs[job_id]
        submission = self.work[job_id].submission_id
        if (
            state != "done"
            and _optional(lambda: self.workspace.read_job_file(submission, f"{job_id}.state")) == b"done"
        ):
            state = "done"
        reason = reason if state != "done" else None
        try:
            self.logs.finish(job_id, f"{state}: {reason}" if reason else state)
        except (OSError, StorageError) as exc:
            if state == "done":
                raise
            reason = f"{reason or state}; additionally, publishing the job log failed: {exc}"
            logger.exception("Could not publish the failed job log %s", job_id)
        job.state, job.reason = state, reason
        self.workspace.put_job_file(submission, f"{job_id}.state", state.encode())
        if all(self.state.jobs[j].state in _TERMINAL for j in self.graphs[submission]):
            counts = Counter(self.state.jobs[j].state for j in self.graphs[submission])
            _event(self.directory, f"SkyPilot graph {submission}: {counts['done']} done, {counts['failed']} failed.")

    def _drain(self, worker: WorkerState) -> None:
        if worker.state == "down" or (worker.state == "draining" and worker.future is not None):
            return
        launch = worker.future if worker.state == "provisioning" else None
        worker.state = "draining"
        _event(self.directory, f"Terminating SkyPilot worker {worker.name}.")

        def down() -> None:
            if launch is not None:
                # A failed launch can still own VMs; always request teardown afterward.
                with contextlib.suppress(Exception):
                    connections, _ = launch.result()
                    for connection in connections:
                        connection.close()
            _sky_wait(self.sky, self.sky.down(worker.name), self.directory / "logs" / f"{worker.name}-down.log")

        self._request(worker, down)

    def _request(self, worker: WorkerState, action: Callable[[], Any]) -> None:
        worker.future = self.threads.submit(action)
        worker.future.add_done_callback(lambda future: self.events.put((worker.name, 0, future)))

    def _complete_request(self, worker: WorkerState) -> None:
        future, worker.future = worker.future, None
        assert future is not None  # noqa: S101 -- only called for an outstanding SDK request
        result = future.result()
        if worker.state == "provisioning":
            worker.connections, worker.ips = result
            if len(worker.connections) != self.offerings[worker.offering].worker.nodes:
                raise ExecutionError(f"Worker {worker.name} returned an unexpected node count.")
            worker.state = "connecting"
        else:
            worker.state = "down"
            _event(self.directory, f"SkyPilot worker {worker.name} terminated.")
            worker.close()
            worker.connections.clear()
            worker.prepared.clear()

    def _poll_jobs(self) -> None:
        # Local cancellation inbox: one directory scan, no per-WorkUnit object-store reads.
        for path in self.directory.glob("*.cancel"):
            job_id = path.stem
            job = self.state.jobs.get(job_id)
            if job is not None and job.state not in _TERMINAL:
                if job.cluster is None:
                    self._finish(job_id, "failed", "Cancelled by user.")
                else:
                    job.reason = "Cancelled by user."
                    worker = next(w for w in self.state.workers if w.name == job.cluster)
                    worker.send({"kind": "cancel", "job_id": job_id})
            path.unlink()
        now = time.monotonic()
        if now - self.last_ping >= 5:  # noqa: PLR2004 -- heartbeat seconds
            for worker in self.state.workers:
                if worker.state not in {"ready", "connecting"}:
                    continue
                worker.send({"kind": "ping"})
                for connection in worker.connections:
                    if now - connection.last_seen > 60:  # noqa: PLR2004 -- lease seconds
                        raise ExecutionError(f"Worker {worker.name} stopped responding; execution is uncertain.")
            self.last_ping = now
        while not self.events.empty():
            name, rank, event = self.events.get_nowait()
            worker = next(w for w in self.state.workers if w.name == name)
            if isinstance(event, Future):
                if worker.future is event:
                    self._complete_request(worker)
                continue
            if self._capture_output(worker, rank, event):
                continue
            if worker.state in {"down", "draining"}:
                continue
            if event["kind"] == "lost":
                raise ExecutionError(f"Worker {name} connection lost: {event['reason']}")
            if not worker.connections:
                if worker.future is not None and worker.future.done():
                    self._complete_request(worker)
                else:
                    self.events.put((name, rank, event))
                    break  # node readiness may precede the SDK completion event
            if event["kind"] == "ready":
                connection = worker.connections[rank]
                if len(event["cpus"]) < self.offerings[worker.offering].worker.cpus:
                    raise ExecutionError(f"Worker {name} exposes fewer CPUs than its declared budget.")
                connection.ready = True
                if all(c.ready for c in worker.connections):
                    worker.state, worker.idle_since = "ready", time.time()
                    _event(
                        self.directory,
                        f"SkyPilot worker {name} connected in {time.monotonic() - worker.started:.1f}s; "
                        "ready to prepare environments.",
                    )
            if event["kind"] != "finished":
                continue
            job_id = event["job_id"]
            codes = worker.completions.setdefault(job_id, {})
            codes[rank] = event["code"]
            if event["code"]:
                worker.send({"kind": "cancel", "job_id": job_id})
            if len(codes) != len(worker.connections):
                continue
            del worker.completions[job_id]
            if worker.preparing is not None and worker.preparing[0] == job_id:
                _, key = worker.preparing
                worker.preparing = None
                if any(codes.values()):
                    raise ExecutionError(
                        f"Environment preparation failed on {name}; see worker logs in {self.directory}."
                    )
                worker.prepared.add(key)
                _event(self.directory, f"Environment ready on {name} ({event['seconds']:.1f}s on rank {rank}).")
            else:
                spec, job = self.work[job_id], self.state.jobs[job_id]
                self.durations[spec.duration_key] = event["seconds"]
                self._finish(
                    job_id,
                    "failed" if any(codes.values()) or job.reason else "done",
                    job.reason or f"Worker subprocess exited with codes {codes}.",
                )

    def _capture_output(self, worker: WorkerState, rank: int, event: Any) -> bool:
        if not isinstance(event, dict) or event.get("kind") != "log":
            return False
        job_id = event["job_id"]
        data = base64.b64decode(event["data"])
        if job_id.startswith("prepare-"):
            with (self.directory / "logs" / f"{worker.name}-prepare-{rank}.log").open("ab") as output:
                output.write(data)
        else:
            if len(worker.connections) > 1:
                data = f"\n--- rank {rank} ---\n".encode() + data
            self.logs.write(job_id, data)
        return True

    def _dispatch(self, worker: WorkerState, spec: WorkSpec, *, prepare: bool = False) -> None:
        offering = self.offerings[worker.offering]
        active = self._active(worker)
        capacity = {"cpus": offering.worker.cpus, "gpus": sum(offering.worker.accelerators.values())}
        assigned = {}
        for key, total in capacity.items():
            used = {i for j in active for i in getattr(self.state.jobs[j], key)}
            count = spec.resources["accelerators"] if key == "gpus" else total if prepare else spec.resources["cpus"]
            assigned[key] = [i for i in range(total) if i not in used][:count]
        job_id = f"prepare-{spec.job_id}" if prepare else spec.job_id
        if prepare:
            worker.preparing = job_id, spec.environment_key
            _event(self.directory, f"Preparing environment on SkyPilot worker {worker.name}.")
        else:
            job = self.state.jobs[job_id]
            job.cluster, job.state, job.started = worker.name, "running", time.time()
            job.cpus, job.gpus = assigned["cpus"], assigned["gpus"]
            # Persist ownership before writing any command. Never replay after stream loss.
            self.save()
            self.logs.sync()
            self.logs.write(spec.job_id, f"\nRunning on {worker.name}; using prepared environment.\n".encode())
        worker.idle_since = None
        for rank, connection in enumerate(worker.connections):
            environment = resource_environment(
                cpu_indices=assigned["cpus"],
                accelerator_type=offering.worker.accelerator_type,
                accelerator_indices=assigned["gpus"],
            ) | {
                "SKYPILOT_NODE_RANK": str(rank),
                "SKYPILOT_NODE_IPS": "\n".join(worker.ips),
                "MISEN_PREPARE_ONLY": "1" if prepare else "",
                "MISEN_JOB_LOG_CAPTURED": "1",
            }
            connection.send(
                {
                    "kind": "run",
                    "job_id": job_id,
                    "command": spec.command,
                    "env": environment,
                    "cpus": assigned["cpus"],
                    "log": f"~/.sky/{worker.name}-{job_id}-{rank}.log",
                }
            )

    def _provision(self, index: int) -> WorkerState:
        self.state.sequence += 1
        worker = WorkerState(f"{self.config.name_prefix}-{self.submission_id.lower()}-w{self.state.sequence}", index)
        self.state.workers.append(worker)
        self.save()
        offering = self.offerings[index]
        _event(
            self.directory,
            f"Provisioning SkyPilot worker {worker.name} on {offering.worker.infra}: "
            f"{offering.worker.nodes} node(s), {offering.worker.cpus} CPUs, "
            f"{offering.worker.memory} GiB RAM/node, GPUs {offering.worker.accelerators or 'none'}.",
        )
        # One native job guards the entire worker lifetime, so SkyPilot does not
        # autodown a busy VM merely because WorkUnits run outside its job queue.
        guard = (
            "import os,time; p=os.path.expanduser(" + repr(f"~/.sky/{worker.name}.lease") + "); "
            "open(p,'a').close()\nwhile time.time()-os.stat(p).st_mtime < 120: time.sleep(5)"
        )
        task = self.sky.Task(
            name=worker.name,
            run=shlex.join(["python3", "-u", "-c", guard]),
            num_nodes=offering.worker.nodes,
            resources=self.sky.Resources(**offering.resource_args),
            api_server_access=False,
        )

        def launch() -> Any:
            request = self.sky.launch(task, cluster_name=worker.name, idle_minutes_to_autostop=10, down=True)
            _, handle = _sky_wait(self.sky, request, self.directory / "logs" / f"{worker.name}-launch.log")
            return _connect(handle, worker.name, self.events, self.directory / "logs")

        self._request(worker, launch)
        return worker

    def _frontiers(self) -> tuple[list[WorkSpec], list[WorkSpec]]:
        """Ready work and the independent work expected within the lookahead window."""
        ready, future, predicted = [], [], {}
        # Specs arrive in dependency order; use measured elapsed time, never task timeouts.
        for spec in self.work.values():
            job = self.state.jobs[spec.job_id]
            parents = [self.state.jobs[p].state for p in spec.dependencies]
            if job.state not in _TERMINAL and job.cluster is None and "failed" in parents:
                self._finish(spec.job_id, "failed", "A prerequisite WorkUnit failed.")
            if job.state in _TERMINAL:
                predicted[spec.job_id] = 0.0
                continue
            start = max((predicted[p] for p in spec.dependencies), default=0.0)
            duration = self.durations.get(spec.duration_key, float(self.config.lookahead_seconds or 90))
            predicted[spec.job_id] = max(0, duration - (time.time() - job.started)) if job.started else start + duration
            if job.cluster is None:
                if all(p == "done" for p in parents):
                    ready.append(spec)
                elif self.config.lookahead_seconds and start <= self.config.lookahead_seconds:
                    future.append(spec)
        future_ids = {s.job_id for s in future}
        return ready, [s for s in future if not future_ids.intersection(s.dependencies)]

    def _schedule(self, ready: list[WorkSpec], future: list[WorkSpec]) -> list[WorkSpec]:
        """Pack ready work first; reserve future capacity without executing it."""
        waiting = []
        for frontier, speculative in ((ready, False), (future, True)):
            reserved: dict[str, set[str]] = {}
            for spec in sorted(frontier, key=lambda s: (-s.resources["accelerators"], s.job_id)):
                candidates = [
                    w
                    for w in self.state.workers
                    if w.state in {"ready", "connecting", "provisioning"}
                    and self._fits(w, spec, reserved.get(w.name, set()), speculative=speculative)
                ]
                candidates.sort(
                    key=lambda w: (bool(self.offerings[w.offering].worker.accelerators), w.state != "ready", w.name)
                )
                if not candidates and (choices := self._choices(spec)):
                    candidates = [self._provision(choices[0])]
                if not candidates:
                    if not speculative:
                        waiting.append(spec)
                    continue
                worker = candidates[0]
                reserved.setdefault(worker.name, set()).add(spec.job_id)
                if worker.state != "ready" or worker.preparing is not None:
                    continue
                if spec.environment_key not in worker.prepared:
                    if not self._active(worker):
                        self._dispatch(worker, spec, prepare=True)
                elif not speculative:
                    reserved[worker.name].remove(spec.job_id)
                    self._dispatch(worker, spec)
        return waiting

    def tick(self) -> bool:
        """Accept graphs, consume events, schedule work, and retire idle workers."""
        if time.monotonic() - self.logs.last_sync >= 1:
            self.logs.sync()
        for path in self.directory.glob("graph-*.pkl"):
            submission_id = path.stem.removeprefix("graph-")
            specs = cloudpickle.loads(path.read_bytes())
            self.logs.add(specs)
            try:
                self.add_graph(submission_id, specs)
            except ConfigError as exc:
                for spec in specs:
                    self.logs.finish(spec.job_id, str(exc))
                state = ControllerState({s.job_id: WorkState(state="failed", reason=str(exc)) for s in specs})
                self.workspace.put_job_file(submission_id, STATE_FILE, msgspec.json.encode(state))
                _atomic_write(self.directory / f"{submission_id}.json.error", str(exc).encode())
            path.unlink()
        self._poll_jobs()
        waiting = self._schedule(*self._frontiers())
        pending = any(j.state not in _TERMINAL for j in self.state.jobs.values())
        stopping = self.stop_requested()
        if stopping and pending:
            raise ExecutionError("Local SkyPilot session is shutting down.")
        for worker in self.state.workers:
            if stopping:
                self._drain(worker)
                continue
            if worker.state != "ready" or self._active(worker) or worker.preparing is not None:
                continue
            if worker.idle_since is None:
                worker.idle_since = time.time()
            replace = (
                waiting
                and not any(w.state == "draining" for w in self.state.workers)
                and any(
                    not self.offerings[worker.offering].fits(s.resources) and self._choices(s, excluding=worker)
                    for s in waiting
                )
            )
            if (
                replace
                or (not pending and not self.config.reuse_workers)
                or time.time() - worker.idle_since >= self.config.idle_timeout_minutes * 60
            ):
                self._drain(worker)
        self.save()
        return bool(
            not pending
            and (stopping or not self.config.reuse_workers)
            and all(w.state == "down" for w in self.state.workers)
        )

    def run(self) -> None:
        """Never replay ambiguous execution; confirm teardown before publishing failure."""
        try:
            while not self.tick():
                time.sleep(0.1)
        except BaseException as exc:
            self.state.failure = str(exc)
            logger.exception("SkyPilot execution failed")
            _event(self.directory, f"SkyPilot execution failed: {exc}")
            # A normal agent exit acknowledges that its subprocesses stopped.
            # Cloud teardown acceptance alone does not establish this: a VM
            # can remain in "shutting-down" for minutes after sky.down returns.
            quiesced = {w.name for w in self.state.workers if w.close()}
            # Agent exit delivers its final buffered output before teardown. Do
            # not process scheduling/completion events while handling a failure.
            try:  # Best-effort log capture must not interrupt VM cleanup.
                while not self.events.empty():
                    name, rank, event = self.events.get_nowait()
                    self._capture_output(next(w for w in self.state.workers if w.name == name), rank, event)
            except Exception:
                logger.exception("Could not capture final worker output during SkyPilot cleanup")
            for worker in self.state.workers:
                try:
                    self._drain(worker)
                except Exception:
                    logger.exception("Could not request teardown of %s", worker.name)
            for worker in self.state.workers:
                if worker.state == "draining" and worker.future is not None:
                    try:
                        self._complete_request(worker)
                    except Exception:
                        logger.exception("Could not confirm teardown of %s", worker.name)
            try:
                uncertain = {w.name for w in self.state.workers if w.state != "down"}
                for job_id, job in self.state.jobs.items():
                    if job.state not in _TERMINAL and job.cluster not in uncertain:
                        state = "unknown" if job.cluster is not None and job.cluster not in quiesced else "failed"
                        self._finish(job_id, state, str(exc))
                self.save()
            except Exception:
                logger.exception("Could not persist controller failure")
            raise
        finally:
            for worker in self.state.workers:
                worker.close()
            self.threads.shutdown(wait=False, cancel_futures=True)


_SESSIONS: list[LocalSkySession] = []


@dataclass(eq=False, repr=False)
class LocalSkySession:
    """One controller child and its isolated local control-plane state."""

    directory: Path
    process: SupervisionSession | None = field(default=None, init=False)
    event_offset: int = field(default=0, init=False)

    def poll_events(self) -> None:
        """Display each child lifecycle event once, honoring runtime_events settings."""
        from rich.markup import escape

        with contextlib.suppress(OSError):  # Progress must never interrupt status checks or cleanup.
            with (self.directory / "events.jsonl").open("rb") as source:
                source.seek(self.event_offset)
                for line in source:
                    if not line.endswith(b"\n"):
                        break
                    self.event_offset += len(line)
                    with contextlib.suppress(json.JSONDecodeError, UnicodeDecodeError):
                        runtime_event(escape(json.loads(line)))

    @property
    def active(self) -> bool:
        """Whether the supervised controller is still alive."""
        return self.process is not None and self.process.status.is_active

    @property
    def failure(self) -> str:
        """Read a child error, or point to its diagnostic log."""
        try:
            return (self.directory / "ready.error").read_text()
        except OSError:
            return f"see {self.directory / 'logs' / 'controller.log'}"

    def start(self, *, timeout: int) -> None:
        """Start and wait for resource preflight before returning job handles."""
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        endpoint = f"http://127.0.0.1:{port}"
        log_dir = self.directory / "logs"
        log_dir.mkdir(exist_ok=True)
        runtime_event(f"Starting local SkyPilot API; logs: {log_dir}")
        environment = {
            "SKY_RUNTIME_DIR": str(self.directory / "runtime"),
            "SKYPILOT_API_SERVER_LOCAL_PORT": str(port),
            "SKYPILOT_API_SERVER_ENDPOINT": endpoint,
            "MISEN_SKYPILOT_OWNER_PID": str(os.getpid()),
            # Relative paths work on the submitter and on remote nodes. SkyPilot
            # uses the same log-directory string when generating remote commands.
            "MISEN_SKYPILOT_LOG_DIR": os.path.relpath(log_dir / "native"),
            "PATH": os.pathsep.join((str(Path(sys.executable).parent), os.environ.get("PATH", ""))),
        }
        (self.directory / "connection.json").write_text(json.dumps(environment))
        command = Command(sys.executable, args=["-m", "misen.executors.skypilot", str(self.directory)])
        command = (
            command.envs(environment)
            .stdout_file(log_dir / "controller.log", append=True)
            .stderr_file(log_dir / "controller.log", append=True)
        )
        # Linux's direct-child SIGKILL would orphan SkyPilot helper processes.
        # There the controller watches its parent and tears down the API tree.
        # Windows Job Objects already provide whole-tree parent-death cleanup.
        if Command.kill_on_parent_death_scope() == "whole_tree":
            command = command.kill_on_parent_death()
        self.process = Supervisor(command, restart="never").start()
        _SESSIONS.append(self)
        try:
            self._wait_ready(timeout)
        except BaseException:
            self.close_all([self])
            raise

    def _wait_ready(self, timeout: int, marker: str = "ready") -> None:
        deadline = time.monotonic() + timeout
        while not (self.directory / marker).exists():
            self.poll_events()
            if error := _optional((self.directory / f"{marker}.error").read_text):
                raise ConfigError(error)
            if not self.active or time.monotonic() >= deadline:
                raise SubmissionError(f"Local SkyPilot session stopped or timed out after {timeout}s: {self.failure}")
            time.sleep(0.1)
        self.poll_events()

    def submit(self, specs: list[WorkSpec], workspace: Workspace, config: SkyPilotExecutor) -> None:
        """Publish commands after graph ownership has been recorded in the workspace."""
        submission_id = specs[0].submission_id
        if self.active:
            runtime_event("Reusing the SkyPilot worker pool and cached environments.")
            _atomic_write(self.directory / f"graph-{submission_id}.pkl", cloudpickle.dumps(specs))
            self._wait_ready(config.startup_timeout, f"{submission_id}.json")
        else:
            payload = {
                "workspace": workspace,
                "submission_id": submission_id,
                "specs": specs,
                "config": msgspec.structs.replace(config),
            }
            _atomic_write(self.directory / "controller.pkl", cloudpickle.dumps(payload))
            self.start(timeout=config.startup_timeout)

    @staticmethod
    def close_all(sessions: Sequence[LocalSkySession], *, timeout: float = 30) -> None:
        """Request worker teardown before stopping the owned process trees."""
        active = [session for session in sessions if session.active]
        if active:
            runtime_event("Shutting down SkyPilot worker pools and local APIs.")
        for session in active:
            try:
                (session.directory / "stop").touch()
            except OSError:
                logger.exception("Could not request SkyPilot cleanup in %s", session.directory)
        deadline = time.monotonic() + timeout
        while any(session.active for session in active) and time.monotonic() < deadline:
            for session in active:
                session.poll_events()
            time.sleep(0.1)
        for session in active:
            if session.process is not None and session.active:
                logger.warning(
                    "SkyPilot cleanup did not finish in time; retaining runtime state in %s", session.directory
                )
                session.process.stop(grace_seconds=2)
        for session in sessions:
            session.poll_events()
            if session in _SESSIONS:
                _SESSIONS.remove(session)


def _check_cloud_access(sky: Any, workers: Sequence[SkyPilotWorker], directory: Path) -> None:
    """Populate the isolated runtime's enabled-cloud state before launching."""
    if not workers:
        return
    infras = tuple(sorted({worker.infra.split("/")[0] for worker in workers}))
    _event(directory, f"Checking SkyPilot credentials for {', '.join(infras)}.")
    results = _sky_wait(
        sky, sky.client.sdk.check(infra_list=infras, verbose=False), directory / "logs" / "credentials.log"
    )
    enabled = {
        name.lower()
        for workspace in results.values()
        for name, capabilities in workspace.items()
        if "compute" in capabilities
    }
    missing = {
        str(sky.Resources(infra=infra).cloud)
        for infra in infras
        if str(sky.Resources(infra=infra).cloud).lower() not in enabled
    }
    if missing:
        raise ConfigError(
            f"SkyPilot compute credentials are not enabled for: {', '.join(sorted(missing))}. See the local API log."
        )


def __getattr__(name: str) -> Any:
    """Load the API logging plugin lazily, keeping SkyPilot an optional import."""
    if name != "_LogPlugin":
        raise AttributeError(name)
    from sky.server.plugins import BasePlugin
    from sky.skylet import constants

    class LogPlugin(BasePlugin):
        def install(self, extension_context: Any) -> None:
            del extension_context
            # The plugin runs in both API and spawned request processes. Only
            # our private API config enables it; shared SkyPilot is untouched.
            cast("Any", constants).SKY_LOGS_DIRECTORY = os.environ["MISEN_SKYPILOT_LOG_DIR"]

    return LogPlugin


def main(directory: Path) -> None:
    """Own the foreground API and bound cleanup after submitter loss."""
    api: SupervisionSession | None = None
    finished = threading.Event()
    owner_pid = int(os.environ.get("MISEN_SKYPILOT_OWNER_PID", os.getppid()))
    payload = cloudpickle.loads((directory / "controller.pkl").read_bytes())
    logs = _JobLogs(payload["workspace"], directory)
    logs.add(payload["specs"])

    def watch_owner() -> None:
        while not finished.wait(0.2):
            if os.getppid() != owner_pid:
                try:
                    (directory / "stop").touch()
                except OSError:
                    logger.exception("Could not request cleanup after SkyPilot owner loss")
                if not finished.wait(30):
                    if api is not None:
                        api.stop(grace_seconds=2)
                    os._exit(1)
                return

    try:
        with contextlib.ExitStack() as cleanup:
            cleanup.callback(finished.set)
            threading.Thread(target=watch_owner, name="misen-skypilot-owner", daemon=True).start()
            sky = _load_skypilot()
            if "MISEN_SKYPILOT_LOG_DIR" in os.environ:
                from sky.server import plugins

                plugin_config = dict(plugins._load_plugin_config() or {})  # noqa: SLF001 -- preserve user plugins
                plugin_config["plugins"] = [
                    *plugin_config.get("plugins", []),
                    {"class": "misen.executors.skypilot._LogPlugin"},
                ]
                config_path = directory / "plugins.json"
                _atomic_write(config_path, json.dumps(plugin_config).encode())
                os.environ["SKYPILOT_SERVER_PLUGINS_CONFIG"] = str(config_path)
            port = int(os.environ["SKYPILOT_API_SERVER_LOCAL_PORT"])
            # Foreground mode never creates a detached shared API daemon.
            command = Command(sys.executable, args=["-c", f"import sky; sky.api_start(foreground=True, port={port})"])
            command = command.stdout_file(directory / "logs" / "api.log", append=True).stderr_file(
                directory / "logs" / "api.log", append=True
            )
            api = Supervisor(command.kill_on_parent_death(), restart="never").start()
            cleanup.callback(api.stop, grace_seconds=2)
            endpoint = os.environ["SKYPILOT_API_SERVER_ENDPOINT"]
            while True:
                if time.monotonic() - logs.last_sync >= 1:
                    logs.sync()
                if (directory / "stop").exists() or not api.status.is_active:
                    raise ExecutionError(  # noqa: TRY301 -- persist startup failure
                        f"Owned SkyPilot API stopped during startup; see {directory / 'logs' / 'api.log'}."
                    )
                try:
                    with urllib.request.urlopen(f"{endpoint}/api/health", timeout=1) as response:  # noqa: S310
                        if response.status == 200:  # noqa: PLR2004
                            break
                except (OSError, urllib.error.URLError):
                    time.sleep(0.1)
            os.environ["SKYPILOT_DISABLE_LOCAL_API_SERVER"] = "1"
            _event(directory, "Local SkyPilot API is ready.")
            config = payload["config"]
            _check_cloud_access(sky, config.workers, directory)
            _event(directory, "Resolving SkyPilot worker offerings and GPU memory requirements.")
            controller = Controller(
                sky=sky,
                directory=directory,
                offerings=resolve_workers(sky, config.workers, config.accelerator_memory),
                stop_requested=lambda: (directory / "stop").exists() or not api.status.is_active,
                log_capture=logs,
                **payload,
            )
            (directory / "ready").touch()
            controller.run()
    except BaseException as exc:
        _event(directory, f"SkyPilot session failed: {exc}")
        with (directory / "logs" / "controller.log").open("a") as output:
            traceback.print_exc(file=output)
        # No launch precedes readiness, so ordinary startup failures are retryable.
        if not (directory / "ready").exists():
            try:
                payload = cloudpickle.loads((directory / "controller.pkl").read_bytes())
                state = ControllerState(
                    {spec.job_id: WorkState(state="failed", reason=str(exc)) for spec in payload["specs"]},
                    failure=str(exc),
                )
                payload["workspace"].put_job_file(payload["submission_id"], STATE_FILE, msgspec.json.encode(state))
                for spec in payload["specs"]:
                    logs.finish(spec.job_id, str(exc))
            except Exception:
                logger.exception("Could not record SkyPilot startup failure")
        _atomic_write(directory / "ready.error", str(exc).encode())
        raise
    finally:
        failed = sys.exc_info()[0] is not None
        try:
            logs.close()
        except (OSError, StorageError):
            if not failed:
                raise
            logger.exception("Could not finalize SkyPilot session logs")


atexit.register(lambda: LocalSkySession.close_all(list(_SESSIONS)))


__all__ = ("SkyPilotExecutor", "SkyPilotJob", "SkyPilotWorker")


class _GraphRecord(msgspec.Struct):
    """Persist ownership before starting a controller, preventing blind replay."""

    directory: str
    records: dict[str, _JobRecord]


def _load_skypilot() -> Any:
    """Load the optional SDK without starting or configuring a server."""
    try:
        sky: Any = importlib.import_module("sky")
    except ModuleNotFoundError as exc:
        if exc.name != "sky":
            raise
        raise ConfigError(
            'SkyPilotExecutor requires skypilot-nightly; install with `uv pip install "misen[skypilot]"`.'
        ) from exc
    if not callable(getattr(sky.server.common, "get_local_api_server_port", None)):
        raise ConfigError(
            "SkyPilotExecutor requires skypilot-nightly>=1.0.0.dev20260905 for isolated local API servers."
        )
    return sky


@dataclass(eq=False, repr=False, slots=True, kw_only=True)
class SkyPilotJob(Job):
    """A logical WorkUnit owned by a local graph controller."""

    work_unit: WorkUnit
    job_id: str
    submission_id: str
    log_path: Path
    workspace: Workspace
    session: LocalSkySession
    _terminal_state: JobState | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Initialize the shared job lifecycle."""
        Job.__init__(self, self.work_unit, self.job_id, self.log_path)

    def state(self) -> JobState:
        """Read logical execution status, including controller failure."""
        return type(self).bulk_state([self])[self]

    def cancel(self) -> None:
        """Cancel this WorkUnit without cancelling unrelated work."""
        try:
            (self.session.directory / f"{self.job_id}.cancel").touch()
        except (MisenError, OSError) as exc:
            raise ExecutionError(f"Could not request cancellation of {self.label}: {exc}") from exc

    @classmethod
    def bulk_state(cls, jobs: Sequence[Job]) -> dict[Job, JobState]:
        """Read one checkpoint per submission; committed results take precedence."""
        result: dict[Job, JobState] = {}
        checkpoints: dict[str, ControllerState | None] = {}
        for session in {job.session for job in cast("Sequence[SkyPilotJob]", jobs)}:
            session.poll_events()
        for job in cast("Sequence[SkyPilotJob]", jobs):
            if job._terminal_state is not None:  # noqa: SLF001
                result[job] = job._terminal_state  # noqa: SLF001
                continue
            try:
                if job.submission_id not in checkpoints:
                    checkpoints[job.submission_id] = _optional(
                        lambda job=job: msgspec.json.decode(
                            (job.session.directory / f"{job.submission_id}.json").read_bytes(), type=ControllerState
                        )
                    )
                checkpoint = checkpoints[job.submission_id]
                logical = checkpoint.jobs.get(job.job_id) if checkpoint is not None else None
                state = cast("JobState", logical.state) if logical is not None else "pending"
                reason = logical.reason if logical is not None else None
                if state not in _TERMINAL and not job.session.active:
                    uncertain = (
                        checkpoint is None or state == "unknown" or any(w.state != "down" for w in checkpoint.workers)
                    )
                    state = "unknown" if uncertain else "failed"
                    reason = f"Local SkyPilot controller exited: {job.session.failure}"
                if state in {"failed", "unknown"}:
                    committed = (
                        _optional(lambda job=job: job.workspace.read_job_file(job.submission_id, f"{job.job_id}.state"))
                        == b"done"
                    )
                    if committed or job.work_unit.done(workspace=job.workspace):
                        state = "done"
                    elif state == "failed":
                        job._record_failure(reason or "WorkUnit failed in the graph controller.")  # noqa: SLF001
                if state in _TERMINAL:
                    job._finalize_log(job.workspace, failed=state == "failed")  # noqa: SLF001
                    job._terminal_state = state  # noqa: SLF001
                result[job] = state
            except (MisenError, OSError, msgspec.DecodeError) as exc:
                raise StatusQueryError(f"Could not query SkyPilot graph status: {exc}") from exc
        return result


def _workers_from_cli(args: list[str]) -> list[SkyPilotWorker]:
    return msgspec.json.decode(args[0], type=list[SkyPilotWorker])


def _workers_to_cli(workers: list[SkyPilotWorker]) -> list[str]:
    return [msgspec.json.encode(workers).decode()]


class SkyPilotExecutor(Executor[SkyPilotJob]):
    """Run ready work on an unordered list of reusable worker types.

    Single-node jobs share CPU, RAM, and whole GPUs within declared budgets;
    multi-node jobs reserve a whole worker group. ``max_workers`` bounds groups
    across the session, including provisioning and draining groups. GPU memory is
    resolved per catalog offering and checked on assigned devices at runtime.

    A supervised local controller owns an isolated foreground SkyPilot API
    server. Scheduling requires the submitting process to stay alive. Use
    ``close()`` or the context manager to cancel unfinished work and clean up;
    normal process exit also requests cleanup. Autodown backs up abrupt exits.
    Provider extras and credentials must be available on the submitting host.
    """

    workers: Annotated[
        list[SkyPilotWorker],
        PrimitiveConstructorSpec(
            nargs=1,
            metavar="JSON",
            instance_from_str=_workers_from_cli,
            is_instance=lambda value: isinstance(value, list),
            str_from_instance=_workers_to_cli,
        ),
    ] = msgspec.field(default_factory=list)
    max_workers: _PositiveInt = 4
    idle_timeout_minutes: _PositiveInt = 10
    startup_timeout: _PositiveInt = 300
    accelerator_memory: dict[_Name, _PositiveInt] = msgspec.field(default_factory=dict)
    dask_startup_timeout: _PositiveInt = DEFAULT_DASK_STARTUP_TIMEOUT
    dask_scheduler_port: Annotated[int, msgspec.Meta(ge=1024, le=65535)] = DEFAULT_DASK_SCHEDULER_PORT
    name_prefix: Annotated[str, msgspec.Meta(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*\Z", max_length=30)] = "misen"
    lookahead_seconds: Annotated[int, msgspec.Meta(ge=0)] = 90
    reuse_workers: bool = True
    _config_validation_errors: ClassVar[tuple[type[Exception], ...]] = (ValueError,)

    def __post_init__(self) -> None:
        """Validate configuration without importing the optional SDK."""
        _validate_fields(self)
        if not self.workers:
            raise ValueError("workers must contain at least one worker type.")
        if not self.snapshot or self.prewarm_envs:
            raise ValueError("SkyPilotExecutor requires snapshot=True and prewarm_envs=False.")
        self._sessions: dict[str, tuple[LocalSkySession, _GraphRecord]] = {}
        self._pool: LocalSkySession | None = None
        self._pool_key: str | None = None

    def close(self) -> None:
        """Stop owned submissions and request cloud-worker cleanup."""
        LocalSkySession.close_all([self._pool] if self._pool is not None else [])

    def __enter__(self) -> Self:
        """Keep scheduling alive for this context's lifetime."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Clean up owned sessions on context exit."""
        self.close()

    def _validate_submission(
        self,
        *,
        work_graph: DependencyGraph[WorkUnit],
        pending_work_units: Sequence[WorkUnit],
        workspace: Workspace,
    ) -> None:
        del work_graph, pending_work_units
        if workspace.bootstrap_transport() is None:
            raise ConfigError("SkyPilotExecutor requires a remotely fetchable workspace transport; use CloudWorkspace.")
        if workspace.get_temp_dir().is_absolute():
            raise ConfigError("SkyPilotExecutor requires a relative workspace cache_dir, such as '.cache/misen'.")
        if not workspace.supports_job_file_reads():
            raise ConfigError("SkyPilotExecutor requires a workspace that supports submission-file coordination reads.")
        if any(worker.infra.split("/")[0] in {"k8s", "kubernetes", "slurm"} for worker in self.workers):
            raise ConfigError("Persistent SkyPilot workers require SSH-accessible VMs or SSH node pools.")

    def _dispatch_work_graph(
        self,
        *,
        pending_work_units: Sequence[WorkUnit],
        jobs: dict,
        workspace: Workspace,
        snapshot: ProjectSnapshot | None,
        progress: Callable[[int], None],
    ) -> None:
        if snapshot is None:
            raise SubmissionError("SkyPilot workers require a project snapshot.")
        identity = sorted(
            (w.root.task_hash().b32(), w.resources, sorted(d.root.task_hash().b32() for d in w.dependencies))
            for w in set(jobs) | set(pending_work_units)
        )
        graph_key = TaskHash.from_object((self, workspace, snapshot.snapshot_key, identity)).b32()
        with workspace.lock("job", graph_key).context():
            owned = self._sessions.get(graph_key)
            if owned is not None and owned[0].active:
                session, record = owned
            else:
                previous = _optional(
                    lambda: msgspec.json.decode(
                        workspace.read_job_file("jobs", f"graph-{graph_key}.json"), type=_GraphRecord
                    )
                )
                if previous is not None:
                    submission_id = next(iter(previous.records.values())).submission_id
                    checkpoint = _optional(
                        lambda: msgspec.json.decode(
                            workspace.read_job_file(submission_id, STATE_FILE), type=ControllerState
                        )
                    )
                    if (
                        checkpoint is None
                        or any(w.state != "down" for w in checkpoint.workers)
                        or any(j.state not in _TERMINAL for j in checkpoint.jobs.values())
                    ):
                        raise SubmissionError(
                            "Previous SkyPilot graph has unconfirmed execution or cleanup; "
                            f"inspect {previous.directory} before retrying."
                        )
                session, record = self._start_graph(pending_work_units, workspace, snapshot, graph_key)
                self._sessions[graph_key] = session, record
            for unit in pending_work_units:
                key = unit.root.task_hash().b32()
                if key not in record.records:
                    raise SubmissionError(
                        "The active graph manifest does not include previously cached work; "
                        "finish it before resubmitting."
                    )
                job = record.records[key]
                jobs[unit] = SkyPilotJob(
                    work_unit=unit,
                    job_id=job.job_id,
                    submission_id=job.submission_id,
                    log_path=workspace.get_job_log(job.job_id, unit),
                    workspace=workspace,
                    session=session,
                )
            progress(len(pending_work_units))

    def _start_graph(
        self, units: Sequence[WorkUnit], workspace: Workspace, snapshot: ProjectSnapshot, graph_key: str
    ) -> tuple[LocalSkySession, _GraphRecord]:
        prepared = {unit: snapshot.prepare_job(unit, workspace, dependency_jobs={}, reuse_env=True) for unit in units}
        specs = []
        for unit, (job_id, argv, env, log_path) in prepared.items():
            resources = unit.resources
            command = self._run_command(argv, env, log_path, resources, uses_dask_client=unit.uses_dask_client)
            specs.append(
                WorkSpec(
                    job_id,
                    command,
                    resources,
                    [prepared[d][0] for d in unit.dependencies if d in prepared],
                    snapshot.submission_id,
                    snapshot.snapshot_key,
                    f"{unit.root.func.__module__}.{unit.root.func.__qualname__}",
                    str(log_path),
                )
            )
        pool_key = TaskHash.from_object(workspace).b32()
        if self._pool is not None and self._pool.active and self._pool_key != pool_key:
            raise ConfigError(
                "Close the SkyPilot executor before switching workspaces; its worker budget spans one workspace."
            )
        session = self._pool
        reuse = session is not None and session.active
        directory = (
            session.directory if reuse else (workspace.get_temp_dir() / "skypilot" / snapshot.submission_id).absolute()
        )
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        record = _GraphRecord(
            str(directory),
            {
                unit.root.task_hash().b32(): _JobRecord(
                    values[0], "local", snapshot.submission_id, unit.resources["time"]
                )
                for unit, values in prepared.items()
            },
        )
        workspace.put_job_file("jobs", f"graph-{graph_key}.json", msgspec.json.encode(record))
        if not reuse:
            session = LocalSkySession(directory)
            self._pool, self._pool_key = session, pool_key
        session.submit(specs, workspace, self)
        return session, record

    def _dispatch(
        self,
        work_unit: WorkUnit,
        dependencies: set[SkyPilotJob],
        workspace: Workspace,
        snapshot: ProjectSnapshot | None,
    ) -> SkyPilotJob:
        raise NotImplementedError("SkyPilot dispatch requires the complete work graph.")

    def _run_command(
        self,
        argv: list[str],
        env: dict[str, str],
        log_path: Path,
        resources: Resources,
        *,
        uses_dask_client: bool,
    ) -> str:
        """Render one bounded rank-aware SkyPilot worker command."""
        env = env | {
            "MISEN_SKYPILOT_RESOURCES": msgspec.json.encode(resources).decode(),
            "MISEN_WORKER_PREFLIGHT": "misen.executors.skypilot:_worker_preflight",
        }
        payload = shlex.join(["env", *(f"{key}={value}" for key, value in env.items()), *argv])
        timeout = f"timeout --signal=TERM --kill-after=30s {resources['time']}m"
        command = (
            managed_ranked_cluster_script(
                argv,
                environment=env,
                workers=resources["nodes"],
                cpus=resources["cpus"],
                memory_gib=resources["memory"],
                startup_timeout=self.dask_startup_timeout,
                node_rank_env="SKYPILOT_NODE_RANK",
                node_ips_env="SKYPILOT_NODE_IPS",
                scheduler_port=self.dask_scheduler_port,
            )
            if uses_dask_client
            else payload
        )
        lines = ["set -o pipefail", f'if [[ -n "${{MISEN_PREPARE_ONLY:-}}" ]]; then exec {timeout} {payload}; fi']
        if resources["nodes"] > 1 and not uses_dask_client:
            lines.append('if [[ "${SKYPILOT_NODE_RANK:-0}" != "0" ]]; then exit 0; fi')
        lines.extend(
            (
                f"mkdir -p {shlex.quote(str(log_path.parent))}",
                f"{timeout} bash -c {shlex.quote(command)} 2>&1 | tee -a {shlex.quote(str(log_path))}",
            )
        )
        return "\n".join(lines)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
