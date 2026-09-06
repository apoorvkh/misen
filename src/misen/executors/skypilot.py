"""SkyPilot execution, capacity, coordination, and process lifecycle.

SkyPilot remains optional and is loaded only for explicit SDK operations.
The same module provides child-only broker, server, and worker-guard roles;
importing it never starts a process or changes the caller's environment.
"""

from __future__ import annotations

import argparse
import contextlib
import contextvars
import functools
import hashlib
import importlib
import json
import logging
import math
import os
import queue as queue_module
import re
import runpy
import secrets
import select
import shlex
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing.connection import Client, Listener
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast
from urllib.error import URLError
from urllib.request import urlopen

import cloudpickle
import msgspec

from misen.exceptions import (
    ConfigError,
    ExecutionError,
    LockUnavailableError,
    MisenError,
    StatusQueryError,
    StorageError,
    SubmissionError,
)
from misen.executor import CompletedJob, Executor, Job, JobState, _JobRecord
from misen.task_metadata import AcceleratorType, Resources, aggregate_resources, meta
from misen.utils.job_dependencies import dependency_state_name, publish_dependency_state
from misen.utils.resource_env import narrow_accelerator_environment, resource_environment

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence
    from multiprocessing.connection import Connection
    from typing import BinaryIO, TextIO

    from misen.tasks import Task
    from misen.utils.graph import DependencyGraph
    from misen.utils.snapshot import ProjectSnapshot
    from misen.utils.work_unit import WorkUnit
    from misen.workspace import Workspace

__all__ = ("SkyPilotCapacity", "SkyPilotExecutor", "SkyPilotJob", "SkyPilotTaskJob")

logger = logging.getLogger(__name__)

# Capacity profiles
# ----------------------------------------------------------------------------

_POOL_NAME = re.compile(r"[a-zA-Z](?:[-_.a-zA-Z0-9]*[a-zA-Z0-9])?")


_CLUSTER_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


_MODEL_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*")


_MAX_SOURCE_NAME_LENGTH = 63


_MAX_OPTION_LENGTH = 512


def _positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = f"{name} must be a positive integer."
        raise ValueError(msg)


def _option_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        msg = f"{name} must be a non-empty string."
        raise ValueError(msg)  # noqa: TRY004 - configuration validation uses ValueError
    normalized = value.strip()
    if not normalized or len(normalized) > _MAX_OPTION_LENGTH or not normalized.isprintable():
        msg = (
            f"{name} must be a non-empty string of at most {_MAX_OPTION_LENGTH} characters without control characters."
        )
        raise ValueError(msg)
    return normalized


class SkyPilotCapacity(msgspec.Struct, kw_only=True, forbid_unknown_fields=True):
    """A fixed per-worker reservation from one SkyPilot capacity source.

    Exactly one of ``pool``, ``cluster``, or ``infra`` identifies the source.
    Existing pools and clusters are borrowed: Misen must not reconfigure or
    terminate them. ``infra`` creates run-owned workers. ``cpus``, ``memory``,
    and accelerator quantities declare the reservation per node, not inferred
    hardware capacity; the allocation backend must validate borrowed capacity
    before assigning work. A cluster supplies one worker reservation, whereas
    ``max_workers`` bounds independently acquired pool or run-owned workers.

    A multi-node reservation requires ``dedicated=True``. Dedicated capacity
    runs one work unit at a time and reserves its entire declared shape even
    for a smaller request. Device names are concrete SkyPilot models; the
    programming backend and per-device memory are declared separately.
    """

    pool: str | None = None
    cluster: str | None = None
    infra: str | list[str] | None = None
    cpus: int = 1
    memory: int = 8
    accelerators: dict[str, int] = msgspec.field(default_factory=dict)
    accelerator_type: AcceleratorType = "cuda"
    accelerator_memory: int | None = None
    max_workers: int = 1
    dedicated: bool = False
    nodes: int = 1
    use_spot: bool = False
    instance_type: str | None = None
    image_id: str | None = None
    disk_size: int | None = None

    def __post_init__(self) -> None:
        """Validate resource limits without importing SkyPilot or contacting it."""
        if sum(source is not None for source in (self.pool, self.cluster, self.infra)) != 1:
            msg = "Exactly one capacity source must be set: pool, cluster, or infra."
            raise ValueError(msg)
        for name, pattern in (("pool", _POOL_NAME), ("cluster", _CLUSTER_NAME)):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or len(value) > _MAX_SOURCE_NAME_LENGTH or pattern.fullmatch(value) is None
            ):
                msg = f"{name} must be a valid SkyPilot name of at most {_MAX_SOURCE_NAME_LENGTH} characters."
                raise ValueError(msg)
        if self.infra is not None:
            if isinstance(self.infra, str):
                self.infra = _option_string(self.infra, "infra")
            elif isinstance(self.infra, list) and self.infra:
                alternatives = [_option_string(value, "infra") for value in self.infra]
                if len(set(alternatives)) != len(alternatives):
                    msg = "infra must not contain duplicate alternatives."
                    raise ValueError(msg)
                self.infra = alternatives
            else:
                msg = "infra must be a non-empty SkyPilot infrastructure string or list of strings."
                raise ValueError(msg)
        for name in ("cpus", "memory", "max_workers", "nodes"):
            _positive_integer(getattr(self, name), name)
        for name in ("dedicated", "use_spot"):
            if not isinstance(getattr(self, name), bool):
                msg = f"{name} must be a boolean."
                raise ValueError(msg)  # noqa: TRY004 - configuration validation uses ValueError
        if self.nodes > 1 and not self.dedicated:
            msg = "Multi-node capacity requires dedicated=True."
            raise ValueError(msg)
        if self.cluster is not None and self.max_workers != 1:
            msg = "An existing cluster requires max_workers=1; it is one declared reservation."
            raise ValueError(msg)
        if self.accelerator_type not in ("cuda", "rocm", "xpu", "mps", "tpu"):
            msg = f"Unsupported accelerator type: {self.accelerator_type!r}."
            raise ValueError(msg)
        if not isinstance(self.accelerators, dict) or len(self.accelerators) > 1:
            msg = "accelerators must be a dictionary containing at most one concrete SkyPilot model."
            raise ValueError(msg)
        normalized_accelerators: dict[str, int] = {}
        for raw_model, count in self.accelerators.items():
            model = _option_string(raw_model, "accelerator model")
            if _MODEL_NAME.fullmatch(model) is None:
                msg = "accelerator model must contain only letters, digits, periods, underscores, or hyphens."
                raise ValueError(msg)
            _positive_integer(count, "accelerator count")
            normalized_accelerators[model] = count
        self.accelerators = normalized_accelerators
        if self.accelerator_memory is not None:
            _positive_integer(self.accelerator_memory, "accelerator_memory")
            if not self.accelerators:
                msg = "accelerator_memory requires an accelerator model."
                raise ValueError(msg)
        for name in ("instance_type", "image_id"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, _option_string(value, name))
        if self.disk_size is not None:
            _positive_integer(self.disk_size, "disk_size")
        if self.borrowed and (
            self.use_spot or self.instance_type is not None or self.image_id is not None or self.disk_size is not None
        ):
            msg = (
                "Borrowed pool/cluster capacity cannot set creation options: "
                "use_spot, instance_type, image_id, disk_size."
            )
            raise ValueError(msg)

    @property
    def borrowed(self) -> bool:
        """Whether this source belongs to the user rather than the current run."""
        return self.infra is None

    @property
    def accelerator_count(self) -> int:
        """Return the number of accelerator devices reserved per node."""
        return sum(self.accelerators.values())

    def fits(self, resources: Resources) -> bool:
        """Whether a valid work-unit request fits an otherwise idle reservation.

        This is shape compatibility, not current availability or SkyPilot
        hardware verification. CPU-only work may fit accelerator capacity;
        placement policy should prefer CPU-only workers when suitable.
        Unknown device memory never satisfies an explicit minimum.
        """
        request = aggregate_resources((resources,), sum_time=False)
        if request["cpus"] > self.cpus or request["memory"] > self.memory or request["nodes"] > self.nodes:
            return False
        if request["accelerators"] == 0:
            return True
        if request["accelerators"] > self.accelerator_count or request["accelerator_type"] != self.accelerator_type:
            return False
        required_memory = request["accelerator_memory"]
        return required_memory is None or (
            self.accelerator_memory is not None and required_memory <= self.accelerator_memory
        )

    def as_sky_options(self) -> dict[str, Any]:
        """Build SkyPilot ``Resources`` arguments for the reserved shape.

        Node count belongs on ``sky.Task(num_nodes=...)`` and is not included
        here. Borrowed sources omit infrastructure and creation-only options.
        The caller selects the pool or cluster separately.
        """
        options: dict[str, Any] = {"cpus": f"{self.cpus}+", "memory": f"{self.memory}+"}
        if self.accelerators:
            options["accelerators"] = dict(self.accelerators)
        if not self.borrowed:
            options["infra"] = list(self.infra) if isinstance(self.infra, list) else self.infra
            options["use_spot"] = self.use_spot
            for name in ("instance_type", "image_id", "disk_size"):
                value = getattr(self, name)
                if value is not None:
                    options[name] = value
        return options


# Durable graph records and readiness
# ----------------------------------------------------------------------------


class GraphWork(msgspec.Struct, forbid_unknown_fields=True):
    """One immutable logical work unit, with an already-staged payload."""

    job_id: str
    dependencies: list[str]
    profile: str
    argv: list[str]
    env: dict[str, str]
    log_path: str
    resources: Resources
    uses_dask_client: bool = False


class AgentWork(msgspec.Struct, forbid_unknown_fields=True):
    """A staged agent bootstrap; its capacity may execute many logical jobs."""

    worker_id: str
    profile: str
    job_id: str
    argv: list[str]
    env: dict[str, str]
    log_path: str


class RunManifest(msgspec.Struct, forbid_unknown_fields=True):
    """Durable submission data; no live coordinator or SDK objects."""

    run_id: str
    snapshot_key: str
    nodes: list[GraphWork]
    agents: list[AgentWork]
    version: Literal[1] = 1


class LogicalState(msgspec.Struct, forbid_unknown_fields=True):
    """Persisted state of a logical job, not a SkyPilot allocation."""

    state: Literal["pending", "running", "done", "failed", "unknown"] = "pending"
    reason: str | None = "Waiting for dependencies."
    attempt_id: str | None = None
    worker_id: str | None = None


class RunState(msgspec.Struct, forbid_unknown_fields=True):
    """Coalesced index for batched observation without per-job cloud queries."""

    run_id: str
    jobs: dict[str, LogicalState]
    status: Literal["running", "done", "failed", "interrupted"] = "running"
    cleanup_errors: list[str] = msgspec.field(default_factory=list)
    version: Literal[1] = 1
    heartbeat_at: float = 0.0


class ReadyGraph:
    """Linear-time dependency accounting and ready-only admission.

    Each node has one attempt in this implementation. Uncertain execution is
    never replayed automatically. Independent branches survive another branch's
    failure, and blocked descendants never consume a capacity reservation.
    """

    def __init__(self, nodes: Iterable[GraphWork]) -> None:
        """Validate an immutable DAG and initialize its ready queue."""
        ordered = list(nodes)
        self.nodes = {node.job_id: node for node in ordered}
        if len(self.nodes) != len(ordered):
            msg = "Duplicate logical job identity in run manifest."
            raise ValueError(msg)
        self.states = {key: LogicalState() for key in self.nodes}
        self.remaining: dict[str, int] = {}
        self.dependents: dict[str, list[str]] = {key: [] for key in self.nodes}
        for node in ordered:
            if len(set(node.dependencies)) != len(node.dependencies):
                msg = "Duplicate dependency in run manifest."
                raise ValueError(msg)
            self.remaining[node.job_id] = len(node.dependencies)
            for dependency in node.dependencies:
                if dependency not in self.nodes:
                    msg = "Unknown logical dependency in run manifest."
                    raise ValueError(msg)
                self.dependents[dependency].append(node.job_id)
        counts = dict(self.remaining)
        queue = deque(key for key, count in counts.items() if count == 0)
        visited = 0
        while queue:
            key = queue.popleft()
            visited += 1
            for child in self.dependents[key]:
                counts[child] -= 1
                if counts[child] == 0:
                    queue.append(child)
        if visited != len(self.nodes):
            msg = "Run manifest contains a dependency cycle."
            raise ValueError(msg)
        self.ready = deque(key for key, count in self.remaining.items() if count == 0)
        for key in self.ready:
            self.states[key].reason = "Waiting for compatible capacity."
        self.active: set[str] = set()
        self.finished: set[str] = set()
        self.revision = 0

    @property
    def complete(self) -> bool:
        """Whether every logical node has a terminal outcome."""
        return len(self.finished) == len(self.nodes)

    def assign(self, job_id: str, attempt_id: str, worker_id: str) -> None:
        """Claim only a ready, unassigned logical job."""
        if self.remaining[job_id] or job_id in self.active or job_id in self.finished:
            msg = "Only a ready, unassigned work unit can be assigned."
            raise ValueError(msg)
        state = self.states[job_id]
        state.attempt_id = attempt_id
        state.worker_id = worker_id
        state.reason = "Preparing execution environment."
        self.active.add(job_id)
        self.revision += 1

    def running(self, job_id: str, attempt_id: str) -> bool:
        """Apply a matching started event; ignore stale or duplicate events."""
        state = self.states[job_id]
        if job_id not in self.active or state.attempt_id != attempt_id:
            return False
        if state.state == "running":
            return False
        state.state = "running"
        state.reason = None
        self.revision += 1
        return True

    def finish(self, job_id: str, *, success: bool, reason: str | None = None) -> None:
        """Commit a terminal logical outcome and advance affected descendants."""
        if job_id in self.finished:
            return
        queue = deque([(job_id, success, reason)])
        while queue:
            key, succeeded, failure = queue.popleft()
            if key in self.finished:
                continue
            self.finished.add(key)
            self.revision += 1
            self.active.discard(key)
            self.states[key].state = "done" if succeeded else "failed"
            self.states[key].reason = failure
            for child in self.dependents[key]:
                if child in self.finished:
                    continue
                if not succeeded:
                    queue.append((child, False, f"Dependency {key} did not succeed."))
                else:
                    self.remaining[child] -= 1
                    if self.remaining[child] == 0:
                        self.states[child].reason = "Waiting for compatible capacity."
                        self.ready.append(child)

    def apply_result(self, job_id: str, attempt_id: str, *, success: bool, reason: str | None = None) -> bool:
        """Accept an outcome only for the currently assigned attempt."""
        if job_id not in self.active or self.states[job_id].attempt_id != attempt_id:
            return False
        self.finish(job_id, success=success, reason=reason)
        return True


def read_run_state(workspace: Workspace, run_id: str) -> RunState:
    """Read the batched durable state index, rejecting wrong-run records.

    Raises:
        StorageError: If the durable record is malformed or belongs to another run.
    """
    try:
        state = msgspec.json.decode(workspace.read_job_file(run_id, "run-state.json"), type=RunState)
    except msgspec.DecodeError as exc:
        msg = "Invalid graph run state."
        raise StorageError(msg) from exc
    if state.run_id != run_id:
        msg = "Graph state belongs to a different run."
        raise StorageError(msg)
    return state


# Isolated API sessions and JSON client
# ----------------------------------------------------------------------------

_START_TIMEOUT_S = 120


_STOP_TIMEOUT_S = 25


_CALL_TIMEOUT_S = 120


_active_session: ContextVar[ManagedSkyPilotSession | None] = ContextVar("misen_skypilot_session", default=None)


_NAMESPACE_PATTERN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}")


def namespace_directory(namespace: str) -> Path:
    """Resolve persistent state outside project snapshots and disposable caches."""
    if not isinstance(namespace, str) or not _NAMESPACE_PATTERN.fullmatch(namespace) or namespace in {".", ".."}:
        msg = "api_server_namespace must be 1-64 letters, digits, periods, underscores, or hyphens."
        raise ValueError(msg)
    state_home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return (state_home / "misen" / "skypilot" / namespace).expanduser().resolve()


def active_session() -> ManagedSkyPilotSession | None:
    """Return this context's session; jobs retain it for polling in other threads."""
    return _active_session.get()


@contextlib.contextmanager
def managed_session(namespace: str = "default") -> Iterator[ManagedSkyPilotSession]:
    """Nest matching sessions and isolate different namespaces and threads."""
    directory = namespace_directory(namespace)
    existing = active_session()
    if existing is not None and existing.directory == directory:
        existing.check_open()
        yield existing
        return
    session = ManagedSkyPilotSession(directory)
    token = _active_session.set(session)
    error: BaseException | None = None
    try:
        yield session
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            session.close(error)
        finally:
            _active_session.reset(token)


class ManagedSkyPilotSession:
    """A lazy, scoped lease on a namespace's API server and isolated SDK."""

    def __init__(self, directory: Path, *, _call_timeout_s: float = _CALL_TIMEOUT_S) -> None:
        """Initialize a session without importing SkyPilot or starting processes."""
        if (
            isinstance(_call_timeout_s, bool)
            or not isinstance(_call_timeout_s, (int, float))
            or not math.isfinite(_call_timeout_s)
            or _call_timeout_s <= 0
        ):
            msg = "SkyPilot broker call timeout must be finite and positive."
            raise ValueError(msg)
        self.directory = directory
        self.jobs: list[SkyPilotJob] = []
        self.closed = False
        self.endpoint: str | None = None
        self.log_path: Path | None = None
        self._connection: Connection | None = None
        self._call_timeout_s = _call_timeout_s
        self._connection_error: str | None = None
        self._closing = False
        self._closing_thread: int | None = None
        self._close_deadline: float | None = None
        self._lock = threading.RLock()
        self.client = _SkyClient(self)

    def check_open(self) -> None:
        """Prevent closed handles from silently creating another server."""
        if self.closed:
            msg = "This SkyPilot job's API session is closed; resubmit inside executor.session() to reattach."
            raise ExecutionError(msg)
        if self._connection_error is not None:
            raise ExecutionError(self._connection_error)
        if self._closing and self._closing_thread != threading.get_ident():
            msg = "SkyPilot session is closing and cannot accept new calls."
            raise ExecutionError(msg)

    def ensure_started(self) -> None:
        """Connect to a live namespace broker or launch one under its own lock."""
        try:
            from filelock import FileLock
        except ModuleNotFoundError as exc:
            msg = "Isolated API sessions require `misen[skypilot-managed]`."
            raise ConfigError(msg) from exc

        with self._lock:
            self.check_open()
            if self._connection is not None:
                return
            if self._closing:
                msg = "SkyPilot session is closing and cannot start another broker."
                raise ExecutionError(msg)
            if os.name != "posix":
                msg = "manage_api_server requires Linux or macOS."
                raise ConfigError(msg)
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            with FileLock(self.directory / "session.lock", timeout=_START_TIMEOUT_S):
                descriptor_path = self.directory / "server.json"
                if descriptor_path.exists():
                    try:
                        self._connect(json.loads(descriptor_path.read_text()))
                    except (OSError, EOFError, ExecutionError, ValueError, KeyError):
                        # A previous last client may still be shutting down.
                        # The new broker takes a namespace lifetime lock before
                        # touching any SkyPilot state.
                        pass
                    else:
                        return
                self._start()

    def _connect(self, descriptor: dict[str, Any]) -> None:
        connection = Client(descriptor["address"], family="AF_UNIX", authkey=bytes.fromhex(descriptor["authkey"]))
        try:
            connection.send_bytes(b'{"op":"acquire"}')
            if not connection.poll(_START_TIMEOUT_S):
                msg = "Timed out acquiring a SkyPilot namespace session."
                raise ExecutionError(msg)  # noqa: TRY301
            reply = json.loads(connection.recv_bytes())
            if reply.get("result") != "acquired":
                msg = "SkyPilot namespace server is shutting down; retry the submission."
                raise ExecutionError(msg)  # noqa: TRY301
            self.check_open()
        except BaseException:
            connection.close()
            raise
        self._connection = connection
        self.endpoint = descriptor["endpoint"]
        self.log_path = Path(descriptor["log_path"])

    def _start(self) -> None:
        identity_path = self.directory / "identity"
        if not identity_path.exists():
            identity_path.touch(mode=0o600, exist_ok=False)
            identity_path.write_text(uuid.uuid4().hex[:8])
        identity = identity_path.read_text().strip()
        if not re.fullmatch(r"[a-f0-9]{8}", identity):
            msg = f"Invalid SkyPilot namespace identity in {identity_path}."
            raise ConfigError(msg)
        config_path = self.directory / "config.yaml"
        if not config_path.exists():
            config_path.touch(mode=0o600, exist_ok=False)
            config_path.write_text('{"jobs": {"controller": {"consolidation_mode": false}}}\n')
        self.log_path = self.directory / f"server-{time.time_ns()}.log"
        self.log_path.touch(mode=0o600, exist_ok=False)
        env = _isolated_environment(self.directory, identity, config_path)
        process: subprocess.Popen[bytes] | None = None
        try:
            with self.log_path.open("wb") as log:
                process = subprocess.Popen(  # noqa: S603
                    [
                        sys.executable,
                        "-m",
                        "misen.executors.skypilot",
                        "--broker",
                        str(self.directory),
                        str(self.log_path),
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=log,
                    start_new_session=True,
                    env=env,
                    cwd=self.directory,
                )
            if process.stdout is None:
                msg = "SkyPilot broker did not provide a startup pipe."
                raise ExecutionError(msg)
            readable, _, _ = select.select([process.stdout], [], [], _START_TIMEOUT_S)
            if not readable:
                msg = f"Timed out starting Misen's SkyPilot server; see {self.log_path}."
                raise ExecutionError(msg)
            line = process.stdout.readline()
            if not line:
                msg = f"SkyPilot broker exited during startup; see {self.log_path}."
                raise ExecutionError(msg)
            descriptor = json.loads(line)
            if "error" in descriptor:
                raise ConfigError(descriptor["error"])
            self._connect(descriptor)
            logger.info("Connected to Misen's SkyPilot namespace at %s (state=%s).", self.endpoint, self.directory)
        finally:
            if process is not None:
                # EOF on the startup pipe before a socket lease is acquired
                # also handles the creator being killed during startup.
                if process.stdin is not None:
                    process.stdin.close()
                if process.stdout is not None:
                    process.stdout.close()
                threading.Thread(target=process.wait, daemon=True, name="misen-skypilot-reaper").start()

    def call(self, operation: str, **arguments: Any) -> Any:
        """Call the isolated SDK over authenticated JSON, never Python pickle."""
        with self._lock:
            self.ensure_started()
            return self._exchange({"op": operation, "args": arguments}, timeout=self._call_timeout_s)

    def _invalidate_connection(self, message: str) -> None:
        """Discard an uncertain RPC stream; never consume its eventual stale reply."""
        connection, self._connection = self._connection, None
        self._connection_error = message
        if connection is not None:
            # Closing an fd in another thread need not wake a blocked read.
            # Shutdown the duplicated Unix socket first to interrupt both RPC
            # directions; the original connection still owns its descriptor.
            with contextlib.suppress(OSError, TypeError, ValueError):
                with socket.socket(fileno=os.dup(connection.fileno())) as stream:
                    stream.shutdown(socket.SHUT_RDWR)
            connection.close()

    def _exchange(self, message: dict[str, Any], *, timeout: float | None = None) -> Any:
        connection = self._connection
        if connection is None:
            msg = "SkyPilot session has no broker connection."
            raise ExecutionError(msg)
        timeout = self._call_timeout_s if timeout is None else timeout
        if self._close_deadline is not None:
            timeout = min(timeout, max(0.0, self._close_deadline - time.monotonic()))
        completed = threading.Event()
        received: list[bytes | BaseException] = []

        def exchange() -> None:
            try:
                connection.send_bytes(json.dumps(message).encode())
                received.append(connection.recv_bytes())
            except BaseException as exc:  # noqa: BLE001 -- forward transport failures to the calling thread
                received.append(exc)
            finally:
                completed.set()

        threading.Thread(target=exchange, daemon=True, name="misen-skypilot-rpc").start()
        try:
            if not completed.wait(timeout):
                operation = "server shutdown" if message.get("op") == "release" else f"broker {message.get('op')} call"
                msg = f"SkyPilot {operation} timed out; connection discarded; see {self.log_path}."
                self._invalidate_connection(msg)
                raise ExecutionError(msg)
            response = received[0]
            if isinstance(response, BaseException):
                raise response
            reply = json.loads(response)
        except (OSError, EOFError) as exc:
            msg = f"Lost Misen's SkyPilot server connection; see {self.log_path}."
            self._invalidate_connection(msg)
            raise ExecutionError(msg) from exc
        except (ValueError, UnicodeError) as exc:
            msg = f"Invalid reply from Misen's SkyPilot server; see {self.log_path}."
            self._invalidate_connection(msg)
            raise ExecutionError(msg) from exc
        except (KeyboardInterrupt, SystemExit):
            self._invalidate_connection("SkyPilot broker call was interrupted; its connection was discarded.")
            raise
        if not isinstance(reply, dict) or ("error" not in reply and "result" not in reply):
            msg = f"Invalid reply from Misen's SkyPilot server; see {self.log_path}."
            self._invalidate_connection(msg)
            raise ExecutionError(msg)
        if "error" in reply:
            raise ExecutionError(reply["error"])
        return _decode_result(reply["result"])

    def check(self, infra_list: Sequence[str], *, verbose: bool = False) -> Any:
        """Check credentials and enable the selected clouds in this namespace.

        Run once before using a fresh namespace, and again after changing cloud
        credentials. This does not provision workers or modify other namespaces.
        """
        if (
            isinstance(infra_list, str)
            or not infra_list
            or any(not isinstance(item, str) or not item for item in infra_list)
        ):
            msg = "infra_list must be a nonempty sequence of infrastructure names, e.g. ['aws']."
            raise ValueError(msg)
        return self.call("check", infra_list=list(infra_list), verbose=verbose)

    def pool_apply(self, pool_name: str, config: str | Path) -> None:
        """Create/update a pool in this namespace, waiting for SkyPilot's result."""
        self.call("pool_apply", pool_name=pool_name, config=str(Path(config).expanduser().resolve()))

    def pool_down(self, pool_name: str) -> None:
        """Terminate only the named pool in this namespace."""
        self.call("pool_down", pool_name=pool_name)

    def pool_status(self) -> Any:
        """Return this namespace's pool statuses."""
        return self.call("pool_status")

    def close(self, original_error: BaseException | None = None) -> None:
        """Drain/release within one deadline, discarding blocked connections safely."""
        failures: list[Exception] = []
        if self.closed:
            return
        if not self._closing:
            self._closing = True
            self._closing_thread = threading.get_ident()
            self._close_deadline = time.monotonic() + _STOP_TIMEOUT_S
        acquired = self._lock.acquire(timeout=_STOP_TIMEOUT_S)
        if not acquired:
            msg = "SkyPilot session cleanup timed out waiting for an active call; broker connection discarded."
            self.closed = True
            self._invalidate_connection(msg)
            if original_error is not None:
                original_error.add_note(msg)
                return
            raise ExecutionError(msg)
        try:
            if self.closed:
                return
            try:
                if self._connection is not None:
                    for job in self.jobs:
                        if self._connection is None:
                            break
                        if job.managed_job_id is None and job._terminal_state is None:  # noqa: SLF001
                            try:
                                job._resolve_managed_job_id(self.client)  # noqa: SLF001
                            except Exception as exc:  # noqa: BLE001
                                failures.append(exc)
                    if self._connection is not None:
                        try:
                            self._exchange({"op": "release"}, timeout=_STOP_TIMEOUT_S)
                        except Exception as exc:  # noqa: BLE001
                            failures.append(exc)
            finally:
                self.closed = True
                if self._connection is not None:
                    self._connection.close()
            if failures:
                msg = "SkyPilot session cleanup failed: " + "; ".join(str(exc) for exc in failures)
                if original_error is not None:
                    original_error.add_note(msg)
                else:
                    raise ExecutionError(msg) from failures[0]
        finally:
            self._lock.release()


def _isolated_environment(directory: Path, identity: str, config_path: Path) -> dict[str, str]:
    """Keep credentials available while replacing only the child's SkyPilot control settings."""
    # Provider credentials (AWS_*, GOOGLE_*, AZURE_*, etc.) remain available;
    # ambient SkyPilot control/auth flags belong to the ordinary namespace.
    env = {key: value for key, value in os.environ.items() if not key.startswith(("SKYPILOT_", "SKY_", "IS_SKYPILOT_"))}
    env.update(
        SKY_RUNTIME_DIR=str(directory),
        SKYPILOT_USER_ID=identity,
        SKYPILOT_GLOBAL_CONFIG=str(config_path),
        SKYPILOT_PROJECT_CONFIG=str(config_path),
        SKYPILOT_API_COOKIE_FILE=str(directory / "cookies.txt"),
    )
    return env


def _decode_result(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"__tuple__"}:
            return tuple(_decode_result(item) for item in value["__tuple__"])
        return {key: _decode_result(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_result(item) for item in value]
    return value


class _Resources:
    def __init__(self, session: ManagedSkyPilotSession, **kwargs: Any) -> None:
        self.session = session
        self.options = kwargs

    def validate(self) -> None:
        self.session.call("validate_resources", options=self.options)


class _Task:
    def __init__(self, **kwargs: Any) -> None:
        self.options = kwargs


def _task_options(task: _Task) -> dict[str, Any]:
    """Serialize resource candidates identically for managed and cluster jobs."""
    options = dict(task.options)
    resources = options.get("resources")
    if resources is not None:
        if isinstance(resources, _Resources):
            resources = [resources]
        options["resources"] = [dict(resource.options) for resource in resources]
    return options


def _cluster_arguments(cluster_name: str, job_ids: Sequence[int] | None = None) -> dict[str, Any]:
    """Require one explicit cluster and, when provided, explicit positive job IDs."""
    if (
        not isinstance(cluster_name, str)
        or not cluster_name
        or cluster_name.strip() != cluster_name
        or any(char in cluster_name for char in "*?[]\x00\n\r")
    ):
        msg = "A single explicit SkyPilot cluster name is required."
        raise ValueError(msg)
    arguments: dict[str, Any] = {"cluster_name": cluster_name}
    if job_ids is not None:
        if (
            not isinstance(job_ids, (list, tuple))
            or not job_ids
            or any(type(job_id) is not int or job_id < 1 for job_id in job_ids)
        ):
            msg = "Cluster operations require a nonempty sequence of positive job IDs."
            raise ValueError(msg)
        arguments["job_ids"] = list(job_ids)
    return arguments


class _SkyJobs:
    def __init__(self, session: ManagedSkyPilotSession) -> None:
        self.session = session

    def launch(self, task: _Task, **kwargs: Any) -> str:
        return self.session.call("launch", task=_task_options(task), **kwargs)

    def queue_v2(self, **kwargs: Any) -> str:
        return self.session.call("queue_v2", **kwargs)

    def cancel(self, **kwargs: Any) -> str:
        return self.session.call("cancel", **kwargs)


class _SkyClient:
    """The small SDK surface used by SkyPilotExecutor, without importing sky."""

    Task = _Task

    def __init__(self, session: ManagedSkyPilotSession) -> None:
        self.session = session
        self.jobs = _SkyJobs(session)
        self.server = SimpleNamespace(common=SimpleNamespace(is_api_server_local=lambda: True))

    def Resources(self, **kwargs: Any) -> _Resources:  # noqa: N802
        return _Resources(self.session, **kwargs)

    def get(self, request_id: str) -> Any:
        return self.session.call("get", request_id=request_id)

    def exec(self, task: _Task, cluster_name: str) -> str:
        return self.session.call("cluster_exec", task=_task_options(task), **_cluster_arguments(cluster_name))

    def job_status(self, cluster_name: str, job_ids: Sequence[int]) -> str:
        return self.session.call("cluster_job_status", **_cluster_arguments(cluster_name, job_ids))

    def cancel(self, cluster_name: str, job_ids: Sequence[int]) -> str:
        return self.session.call("cluster_cancel", **_cluster_arguments(cluster_name, job_ids))

    def queue(self, cluster_name: str, *, skip_finished: bool = False, all_users: bool = False) -> str:
        if type(skip_finished) is not bool or all_users is not False:
            msg = "Cluster queue requires a boolean skip_finished and all_users=False."
            raise ValueError(msg)
        return self.session.call(
            "cluster_queue", **_cluster_arguments(cluster_name), skip_finished=skip_finished, all_users=False
        )

    def api_status(self, **kwargs: Any) -> Any:
        return self.session.call("api_status", **kwargs)

    def api_info(self) -> Any:
        return self.session.call("api_info")


# Child-only SDK broker and server supervisor
# ----------------------------------------------------------------------------

_GRACE_S = 5


def _stop_tree(process: subprocess.Popen[bytes]) -> None:
    """Reap only our own child and its descendants, including detached workers."""
    import psutil

    descendants = []
    with contextlib.suppress(psutil.NoSuchProcess):
        descendants = psutil.Process(process.pid).children(recursive=True)
    with contextlib.suppress(ProcessLookupError):
        process.terminate()
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=_GRACE_S)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    for child in descendants:
        with contextlib.suppress(psutil.NoSuchProcess):
            child.kill()
    process.wait(timeout=_GRACE_S)
    psutil.wait_procs(descendants, timeout=_GRACE_S)


def _choose_port() -> int:
    """Probe both the HTTP port and SkyPilot's derived internal queue port."""
    for _ in range(100):
        port = 20000 + secrets.randbelow(20000)
        queue_port = 50000 + (port - 46569) % 10000
        try:
            with socket.socket() as http, socket.socket() as internal:
                http.bind(("127.0.0.1", port))
                internal.bind(("127.0.0.1", queue_port))
        except OSError:
            continue
        return port
    msg = "Could not find free SkyPilot HTTP and request-queue ports."
    raise RuntimeError(msg)


def _load_isolated_sdk() -> Any:
    """Require native runtime isolation and relocate the remaining identity file."""
    try:
        import sky
        from sky.skylet import runtime_utils
        from sky.utils import cluster_utils, common_utils
    except ModuleNotFoundError as exc:
        msg = "Isolated API sessions require `misen[skypilot-managed]` (SkyPilot nightly)."
        raise RuntimeError(msg) from exc
    if not hasattr(runtime_utils, "runtime_tilde_path"):
        msg = (
            "This SkyPilot build lacks runtime isolation; "
            "install `misen[skypilot-managed]` instead of `misen[skypilot]`."
        )
        raise RuntimeError(msg)
    # Upstream's runtime isolation does not yet cover this one legacy path.
    # SKYPILOT_USER_ID is set before imports; only the server startup writes it.
    common_utils.USER_HASH_FILE = runtime_utils.expanduser("~/.sky/user_hash")
    # Generated SSH shortcuts are runtime output, not user credentials. Keep
    # pool/controller provisioning from rewriting the ordinary SSH config.
    ssh = cluster_utils.SSHConfigHelper
    ssh.ssh_conf_path = runtime_utils.expanduser("~/.ssh/config")
    ssh.ssh_conf_lock_path = runtime_utils.expanduser("~/.sky/locks/.ssh_config.lock")
    ssh.ssh_conf_per_cluster_lock_path = runtime_utils.expanduser("~/.sky/locks/.ssh_config_{}.lock")
    ssh.ssh_cluster_path = runtime_utils.expanduser("~/.sky/generated/ssh/{}")
    ssh.ssh_cluster_key_path = runtime_utils.expanduser("~/.sky/generated/ssh-keys/{}.key")
    if sky.skypilot_config.get_nested(("jobs", "controller", "consolidation_mode"), default_value=False):
        msg = "Misen namespaces require jobs.controller.consolidation_mode=false so cloud jobs outlive local sessions."
        raise RuntimeError(msg)
    if sky.skypilot_config.get_nested(("db",), default_value=None):
        msg = "Misen namespaces cannot use an external/shared SkyPilot database."
        raise RuntimeError(msg)
    return sky


def _encode_result(value: Any) -> Any:
    """Serialize the executor's SDK responses without pickling backend objects."""
    if isinstance(value, Enum):
        return _encode_result(value.value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return {"__tuple__": [_encode_result(item) for item in value]}
    if isinstance(value, list):
        return [_encode_result(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _encode_result(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return _encode_result(value.model_dump(mode="json"))
    msg = f"Unsupported SkyPilot response type: {type(value).__name__}"
    raise TypeError(msg)


def _api_status(**arguments: Any) -> Any:
    """Query our endpoint without SkyPilot's native command-line process probe.

    Our server runs through this module's isolation bootstrap, so the SDK's
    search for ``-m sky.server.server`` incorrectly reports it as stopped.
    Use the pinned SDK's authenticated transport and payloads, without changing
    its process detection or exposing any global start/stop operations.
    """
    from sky.client import common as client_common
    from sky.server import common as server_common
    from sky.server.requests import payloads

    body = payloads.RequestStatusBody(**arguments)
    response = server_common.make_authenticated_request(
        "GET",
        "/api/status",
        params=server_common.request_body_to_params(body),
        timeout=(client_common.API_SERVER_REQUEST_CONNECTION_TIMEOUT_SECONDS, None),
    )
    server_common.handle_request_error(response)
    return [payloads.RequestPayload(**request) for request in response.json()]


def _dispatch(sky: Any, operation: str, arguments: dict[str, Any]) -> Any:
    """Expose only the SDK operations required by Misen and namespace pools."""
    if operation == "validate_resources":
        sky.Resources(**arguments["options"]).validate()
        return None
    if operation in {"launch", "cluster_exec"}:
        arguments = dict(arguments)
        task = _task_from_options(sky, arguments.pop("task"))
        if operation == "launch":
            return str(sky.jobs.launch(task, **arguments))

        if set(arguments) != {"cluster_name"}:
            msg = "Cluster execution accepts only an explicit cluster name."
            raise ValueError(msg)
        return str(sky.exec(task, **_cluster_arguments(arguments["cluster_name"])))
    if operation in {"cluster_job_status", "cluster_cancel"}:
        if set(arguments) != {"cluster_name", "job_ids"} or arguments["job_ids"] is None:
            msg = "Cluster status and cancellation require explicit job IDs."
            raise ValueError(msg)
        options = _cluster_arguments(arguments["cluster_name"], arguments["job_ids"])
        method = sky.job_status if operation == "cluster_job_status" else sky.cancel
        return str(method(**options))
    if operation == "cluster_queue":
        if (
            set(arguments) != {"cluster_name", "skip_finished", "all_users"}
            or type(arguments["skip_finished"]) is not bool
            or arguments["all_users"] is not False
        ):
            msg = "Cluster queue is scoped to the current user's explicitly named cluster."
            raise ValueError(msg)
        return str(
            sky.queue(
                **_cluster_arguments(arguments["cluster_name"]),
                skip_finished=arguments["skip_finished"],
                all_users=False,
            )
        )
    if operation in {"queue_v2", "cancel"}:
        return str(getattr(sky.jobs, operation)(**arguments))
    if operation == "get":
        result = sky.get(arguments["request_id"])
        # Managed launches return a list of IDs; cluster exec returns one ID.
        # Backend handles are deliberately never transported to the parent.
        if (
            isinstance(result, tuple)
            and len(result) == 2  # noqa: PLR2004
            and (result[0] is None or isinstance(result[0], list) or type(result[0]) is int)
        ):
            return result[0], None
        return result
    if operation == "api_status":
        return _api_status(**arguments)
    if operation == "api_info":
        return sky.api_info()
    if operation == "check":
        from sky.client import sdk

        return sky.get(sdk.check(infra_list=tuple(arguments["infra_list"]), verbose=arguments["verbose"]))
    if operation == "pool_apply":
        from sky.serve.serve_utils import UpdateMode

        return sky.get(
            sky.jobs.pool_apply(
                sky.Task.from_yaml(arguments["config"]), pool_name=arguments["pool_name"], mode=UpdateMode.ROLLING
            )
        )
    if operation == "pool_down":
        return sky.get(sky.jobs.pool_down(pool_names=[arguments["pool_name"]]))
    if operation == "pool_status":
        records = sky.get(sky.jobs.pool_status(pool_names=None))
        # The nightly already includes readable cloud/region/resource strings.
        # Drop legacy opaque handles, which are unnecessary for pool status.
        return [
            dict(
                record,
                replica_info=[
                    {key: value for key, value in worker.items() if key != "handle"}
                    for worker in record.get("replica_info", [])
                ],
            )
            for record in records
        ]
    msg = f"Unsupported SkyPilot broker operation: {operation}"
    raise ValueError(msg)


def _task_from_options(sky: Any, options: dict[str, Any]) -> Any:
    """Reconstruct an SDK task without transporting opaque resource objects."""
    options = dict(options)
    if "resources" in options:
        options["resources"] = [sky.Resources(**item) for item in options["resources"]]
    return sky.Task(**options)


class _Leases:
    """Track live clients independently of potentially blocking SDK requests."""

    def __init__(self, dispatch: Callable[[str, dict[str, Any]], Any]) -> None:
        self.dispatch = dispatch
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.clients = 0
        self.acquired = False
        self.last_connection: Connection | None = None

    def bootstrap(self, fd: int) -> None:
        """Detect creator death before it acquires its first socket lease."""
        os.read(fd, 1)
        with self.lock:
            if not self.acquired:
                self.stop.set()

    def accept(self, listener: Listener) -> None:
        """Accept only local authenticated connections."""
        while not self.stop.is_set():
            try:
                connection = listener.accept()
            except (OSError, EOFError):
                return
            threading.Thread(target=self._read, args=(connection,), daemon=True).start()

    def _read(self, connection: Connection) -> None:
        leased = False
        defer_close = False
        requests: queue_module.Queue[dict[str, Any] | None] = queue_module.Queue()
        worker = threading.Thread(target=self._work, args=(connection, requests), daemon=True)
        try:
            while True:
                message = json.loads(connection.recv_bytes())
                if message["op"] == "acquire":
                    with self.lock:
                        if leased or self.stop.is_set():
                            connection.send_bytes(b'{"error":"Server is shutting down"}')
                            return
                        self.clients += 1
                        leased = self.acquired = True
                    connection.send_bytes(b'{"result":"acquired"}')
                    worker.start()
                elif not leased:
                    return
                elif message["op"] == "release":
                    with self.lock:
                        self.clients -= 1
                        leased = False
                        if not self.clients:
                            defer_close = True
                            self.last_connection = connection
                            self.stop.set()
                    if not defer_close:
                        connection.send_bytes(b'{"result":null}')
                    return
                else:
                    requests.put(message)
        except (OSError, EOFError, ValueError, KeyError):
            pass
        finally:
            requests.put(None)
            with self.lock:
                if leased:
                    self.clients -= 1
                    if not self.clients:
                        self.stop.set()
            if not defer_close:
                connection.close()

    def _work(self, connection: Connection, requests: queue_module.Queue[dict[str, Any] | None]) -> None:
        while (message := requests.get()) is not None and not self.stop.is_set():
            try:
                result = self.dispatch(message["op"], message.get("args", {}))
                reply = {"result": _encode_result(result)}
            except Exception as exc:  # noqa: BLE001
                reply = {"error": f"{type(exc).__name__}: {exc}"}
            try:
                connection.send_bytes(json.dumps(reply).encode())
            except (OSError, EOFError):
                return

    def finish(self) -> None:
        """Acknowledge the last graceful release only after the tree is stopped."""
        if self.last_connection is not None:
            with contextlib.suppress(OSError, EOFError):
                self.last_connection.send_bytes(b'{"result":null}')
            self.last_connection.close()


def _wait_ready(child: subprocess.Popen[bytes], endpoint: str, stop: threading.Event) -> None:
    deadline = time.monotonic() + 100
    while not stop.is_set() and child.poll() is None and time.monotonic() < deadline:
        try:
            with urlopen(f"{endpoint}/api/health", timeout=1) as response:  # noqa: S310
                if json.load(response).get("status") == "healthy":
                    return
        except (URLError, OSError, ValueError):
            pass
        stop.wait(0.1)
    msg = "SkyPilot server failed to become healthy before timeout or client disconnect."
    raise RuntimeError(msg)


def _serve(directory: Path, log_path: Path, handshake: TextIO) -> None:
    from filelock import FileLock

    leases = _Leases(lambda operation, arguments: _dispatch(sky, operation, arguments))
    threading.Thread(target=leases.bootstrap, args=(sys.stdin.fileno(),), daemon=True).start()
    signal.signal(signal.SIGTERM, lambda *_: leases.stop.set())
    signal.signal(signal.SIGINT, lambda *_: leases.stop.set())
    with FileLock(directory / "lifetime.lock", timeout=25):
        if leases.stop.is_set():
            return
        port = _choose_port()
        endpoint = f"http://127.0.0.1:{port}"
        os.environ.update(SKYPILOT_API_SERVER_LOCAL_PORT=str(port), SKYPILOT_API_SERVER_ENDPOINT=endpoint)
        sky = _load_isolated_sdk()
        # Never let this SDK spawn an unowned replacement if our child dies.
        sky.server.common.check_server_healthy_or_start_fn = lambda *_args, **_kwargs: (
            sky.server.common.check_server_healthy()
        )
        env = dict(os.environ, IS_SKYPILOT_SERVER="true")
        child = subprocess.Popen(  # noqa: S603
            [sys.executable, "-m", "misen.executors.skypilot", "--server", "--host=127.0.0.1", f"--port={port}"],
            stdin=subprocess.DEVNULL,
            stdout=sys.stderr,
            stderr=sys.stderr,
            start_new_session=True,
            env=env,
        )
        descriptor_path = directory / "server.json"
        try:
            _wait_ready(child, endpoint, leases.stop)
            with tempfile.TemporaryDirectory(prefix="misen-sky-") as socket_dir:
                address = str(Path(socket_dir) / "rpc.sock")
                authkey = secrets.token_bytes(32)
                with Listener(address, family="AF_UNIX", authkey=authkey) as listener:
                    Path(address).chmod(0o600)
                    descriptor = {
                        "address": address,
                        "authkey": authkey.hex(),
                        "endpoint": endpoint,
                        "log_path": str(log_path),
                        "pid": os.getpid(),
                        "server_pid": child.pid,
                    }
                    descriptor_path.touch(mode=0o600, exist_ok=True)
                    descriptor_path.write_text(json.dumps(descriptor))
                    threading.Thread(target=leases.accept, args=(listener,), daemon=True).start()
                    handshake.write(json.dumps(descriptor) + "\n")
                    handshake.flush()
                    while not leases.stop.wait(0.2) and child.poll() is None:
                        pass
        finally:
            leases.stop.set()
            _stop_tree(child)
            descriptor_path.unlink(missing_ok=True)
            leases.finish()


def _broker_main() -> None:
    if sys.argv[1] == "--server":
        _load_isolated_sdk()
        sys.argv = ["sky.server.server", *sys.argv[2:]]
        runpy.run_module("sky.server.server", run_name="__main__")
        return
    # SkyPilot/loggers may write to stdout on import. Keep the one-message
    # handshake on its own fd; all other output belongs in the private log.
    with os.fdopen(os.dup(sys.stdout.fileno()), "w") as handshake:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        try:
            _serve(Path(sys.argv[1]), Path(sys.argv[2]), handshake)
        except Exception as exc:
            handshake.write(json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n")
            handshake.flush()
            raise
    # SDK background threads must not keep an otherwise cleaned broker alive.
    os._exit(0)


# Per-attempt process guard
# ----------------------------------------------------------------------------

_MAX_CONFIG_BYTES = 2 * 1024 * 1024


_POLL_S = 0.01


_FRAME_HEADER_BYTES = 4


_FAILURE_EXIT = 125


_TIMEOUT_EXIT = 124


def _read_exact(fd: int, size: int, deadline: float, stopped: list[bool]) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        remaining = deadline - time.monotonic()
        if stopped[0] or remaining <= 0:
            msg = "Worker guard stopped before receiving its configuration."
            raise ValueError(msg)
        readable, _, _ = select.select([fd], [], [], min(_POLL_S, remaining))
        if not readable:
            continue
        data = os.read(fd, size - len(chunks))
        if not data:
            msg = "Worker agent exited before publishing the full guard configuration."
            raise ValueError(msg)
        chunks.extend(data)
    return bytes(chunks)


def _configuration(fd: int, deadline: float, stopped: list[bool]) -> tuple[list[str], dict[str, str]]:
    size = struct.unpack("!I", _read_exact(fd, _FRAME_HEADER_BYTES, deadline, stopped))[0]
    if not 0 < size <= _MAX_CONFIG_BYTES:
        msg = "Invalid worker guard configuration size."
        raise ValueError(msg)
    record: Any = json.loads(_read_exact(fd, size, deadline, stopped))
    if not isinstance(record, dict):
        msg = "Worker guard configuration must be a JSON object."
        raise ValueError(msg)  # noqa: TRY004 -- all invalid wire data uses ValueError
    argv, env = record.get("argv"), record.get("env")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(arg, str) or "\x00" in arg for arg in argv)
        or not argv[0]
        or not isinstance(env, dict)
        or any(
            not key or "=" in key or "\x00" in key or not isinstance(value, str) or "\x00" in value
            for key, value in env.items()
        )
    ):
        msg = "Invalid worker guard argv or environment."
        raise ValueError(msg)
    return argv, env


def _parent_closed(fd: int) -> bool:
    readable, _, _ = select.select([fd], [], [], 0)
    if not readable:
        return False
    # No further messages are part of this protocol. EOF or unexpected data
    # both revoke the permission to keep executing this assignment.
    os.read(fd, 1)
    return True


def _stop_task_group(process: subprocess.Popen[bytes], grace_s: float) -> None:
    """Terminate and reap the leader, and kill descendants in its private group."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(min(_POLL_S, max(0.0, deadline - time.monotonic())))
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=grace_s)


def guard(parent_fd: int, *, deadline: float, grace_s: float) -> int:
    """Run one payload until completion, pipe EOF, a signal, or its hard deadline."""
    if os.name != "posix" or parent_fd < 0 or not math.isfinite(deadline) or not math.isfinite(grace_s) or grace_s <= 0:
        msg = "Worker guards require POSIX, a valid lifetime pipe, and bounded deadlines."
        raise ValueError(msg)
    stopped = [False]

    def stop(_signum: int, _frame: Any) -> None:
        stopped[0] = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    os.set_inheritable(parent_fd, False)  # noqa: FBT003 -- os exposes this flag as a positional-only argument
    process: subprocess.Popen[bytes] | None = None
    try:
        argv, env = _configuration(parent_fd, deadline, stopped)
        if stopped[0] or _parent_closed(parent_fd):
            return -signal.SIGTERM
        if time.monotonic() >= deadline:
            return _TIMEOUT_EXIT
        process = subprocess.Popen(  # noqa: S603 -- authenticated agent provides validated argv; shell=False
            argv, env=env, stdin=subprocess.DEVNULL, close_fds=True, start_new_session=True
        )
        while True:
            if stopped[0] or _parent_closed(parent_fd):
                return -signal.SIGTERM
            if time.monotonic() >= deadline:
                return _TIMEOUT_EXIT
            if (exit_code := process.poll()) is not None:
                return exit_code
            select.select([parent_fd], [], [], min(_POLL_S, max(0.0, deadline - time.monotonic())))
    finally:
        try:
            if process is not None:
                _stop_task_group(process, grace_s)
        finally:
            os.close(parent_fd)


def _guard_main() -> None:
    """Parse private launcher arguments and preserve payload exit/signal status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-fd", type=int, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--grace-s", type=float, required=True)
    args = parser.parse_args()
    try:
        status = guard(args.parent_fd, deadline=args.deadline, grace_s=args.grace_s)
    except Exception as exc:  # noqa: BLE001 -- never dump the configuration/environment into diagnostics
        sys.stderr.write(f"Worker process guard failed ({type(exc).__name__}).\n")
        sys.stderr.flush()
        status = _FAILURE_EXIT
    if status < 0:
        signum = -status
        if signum not in (signal.SIGKILL, signal.SIGSTOP):
            signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    raise SystemExit(status)


# Reusable worker agent
# ----------------------------------------------------------------------------

_WORKER_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


_MAX_JSON_BYTES = 1024 * 1024


_MAX_ARGUMENTS = 4096


_MAX_ENVIRONMENT = 4096


_VERSION = 1


_IO_ERRORS = (StorageError, OSError)


def _token(value: object) -> str:
    if not isinstance(value, str) or _WORKER_TOKEN.fullmatch(value) is None:
        msg = "Worker protocol identifiers must be bounded alphanumeric tokens."
        raise ValueError(msg)
    return value


def worker_file_name(worker_id: str, kind: Literal["command", "lease", "state"]) -> str:
    """Return a flat, validated worker coordination filename."""
    if kind not in {"command", "lease", "state"}:
        msg = "Invalid worker record kind."
        raise ValueError(msg)
    return f"worker-{_token(worker_id)}.{kind}.json"


def attempt_file_name(attempt_id: str, kind: Literal["accepted", "started", "result"] | None = None) -> str:
    """Return a flat, validated attempt filename (terminal outcome by default)."""
    if kind not in {None, "accepted", "started", "result"}:
        msg = "Invalid attempt record kind."
        raise ValueError(msg)
    suffix = f".{kind}" if kind else ""
    return f"attempt-{_token(attempt_id)}{suffix}.json"


def _read_json(workspace: Workspace, run_id: str, name: str) -> dict[str, Any]:
    data = workspace.read_job_file(run_id, name)
    if len(data) > _MAX_JSON_BYTES:
        msg = "Worker protocol record exceeds the size limit."
        raise ValueError(msg)
    try:
        record = json.loads(data)
    except (ValueError, UnicodeError, RecursionError) as exc:
        msg = "Worker protocol record is not valid JSON."
        raise ValueError(msg) from exc
    if not isinstance(record, dict):
        msg = "Worker protocol record must be a JSON object."
        raise ValueError(msg)  # noqa: TRY004 -- malformed protocol data uses one validation error type
    if type(record.get("version")) is not int or record["version"] != _VERSION or record.get("run_id") != run_id:
        msg = "Worker protocol record has an invalid version or run identity."
        raise ValueError(msg)
    return record


def _write_json(workspace: Workspace, run_id: str, name: str, record: dict[str, Any]) -> None:
    data = json.dumps({"version": _VERSION, "run_id": run_id, **record}, allow_nan=False).encode()
    if len(data) > _MAX_JSON_BYTES:
        msg = "Worker protocol record exceeds the size limit."
        raise ValueError(msg)
    workspace.put_job_file(run_id, name, data)


def _positive_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = "Worker time limits must be finite positive numbers."
        raise ValueError(msg)  # noqa: TRY004 -- protocol validation consistently raises ValueError
    try:
        number = float(value)
    except OverflowError as exc:
        msg = "Worker time limits must be finite positive numbers."
        raise ValueError(msg) from exc
    if not math.isfinite(number) or number <= 0:
        msg = "Worker time limits must be finite positive numbers."
        raise ValueError(msg)
    return number


def _log_path(value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        msg = "Worker log paths must be safe relative paths."
        raise ValueError(msg)
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        msg = "Worker log paths must be safe relative paths."
        raise ValueError(msg)
    root = Path.cwd().resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        msg = "Worker log path escapes the working directory."
        raise ValueError(msg)
    return resolved


@dataclass(frozen=True)
class _Command:
    attempt_id: str
    job_id: str
    argv: list[str]
    env: dict[str, str]
    log_path: Path
    execution_timeout_s: float
    setup_timeout_s: float

    @classmethod
    def parse(cls, record: dict[str, Any]) -> _Command:
        argv = record.get("argv")
        env = record.get("env")
        if (
            not isinstance(argv, list)
            or not argv
            or len(argv) > _MAX_ARGUMENTS
            or any(not isinstance(arg, str) or "\x00" in arg for arg in argv)
            or not argv[0]
        ):
            msg = "Worker argv must be a bounded nonempty list of strings."
            raise ValueError(msg)
        if (
            not isinstance(env, dict)
            or len(env) > _MAX_ENVIRONMENT
            or any(
                not isinstance(key, str)
                or not key
                or "=" in key
                or "\x00" in key
                or not isinstance(value, str)
                or "\x00" in value
                for key, value in env.items()
            )
        ):
            msg = "Worker env must contain bounded string environment entries."
            raise ValueError(msg)
        return cls(
            attempt_id=_token(record.get("attempt_id")),
            job_id=_token(record.get("job_id")),
            argv=argv,
            env=env,
            log_path=_log_path(record.get("log_path")),
            execution_timeout_s=_positive_number(record.get("execution_timeout_s")),
            setup_timeout_s=_positive_number(record.get("setup_timeout_s")),
        )


def _terminate_group(process: subprocess.Popen[bytes], grace_s: float) -> None:
    """Reap the child and kill its remaining process group, including descendants."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        pass
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


@dataclass
class _ActiveAttempt:
    command: _Command
    process: subprocess.Popen[bytes]
    log: BinaryIO
    launched_at: float
    lifetime_fd: int
    execution_started_at: float | None = None


@dataclass
class _Agent:
    workspace: Workspace
    run_id: str
    worker_id: str
    lease_timeout_s: float
    shutdown_grace_s: float
    poll_interval_s: float
    max_runtime_s: float
    generation: str = field(default_factory=lambda: uuid.uuid4().hex)
    base_env: dict[str, str] = field(default_factory=lambda: dict(os.environ))
    seen_attempts: set[str] = field(default_factory=set)
    cancelled_attempts: set[str] = field(default_factory=set)
    active: _ActiveAttempt | None = None
    lease_sequence: int = -1
    lease_at: float = field(default_factory=time.monotonic)

    def _state(self, state: str, attempt_id: str | None = None) -> None:
        _write_json(
            self.workspace,
            self.run_id,
            worker_file_name(self.worker_id, "state"),
            {"worker_id": self.worker_id, "generation": self.generation, "state": state, "attempt_id": attempt_id},
        )

    def _lease(self) -> str | None:
        try:
            record = _read_json(self.workspace, self.run_id, worker_file_name(self.worker_id, "lease"))
        except FileNotFoundError:
            return "coordinator lease disappeared" if self.lease_sequence >= 0 else None
        except (*_IO_ERRORS, ValueError):
            return "coordinator lease could not be read safely"
        sequence = record.get("sequence")
        stop = record.get("stop")
        if (
            record.get("worker_id") != self.worker_id
            or type(sequence) is not int
            or sequence < 0
            or type(stop) is not bool
        ):
            return "invalid coordinator lease"
        if sequence > self.lease_sequence:
            self.lease_sequence = sequence
            self.lease_at = time.monotonic()
            if stop:
                return "coordinator requested stop"
            cancelled = record.get("cancel_attempt_id")
            if cancelled is not None:
                _token(cancelled)
                self.cancelled_attempts.add(cancelled)
                if self.active is not None and self.active.command.attempt_id == cancelled:
                    self._finish(forced=("failed", "attempt cancelled by coordinator"))
                    self._state("idle")
        return None

    def _claim(self, command: _Command) -> bool:
        if command.attempt_id in self.seen_attempts:
            return False
        lock = self.workspace.lock("job", f"agent-{self.run_id}-{command.attempt_id}")
        try:
            with lock.context(blocking=False):
                for kind in (None, "accepted", "started", "result"):
                    try:
                        _read_json(self.workspace, self.run_id, attempt_file_name(command.attempt_id, kind))
                    except FileNotFoundError:
                        continue
                    self.seen_attempts.add(command.attempt_id)
                    return False
                if not lock.is_locked():
                    return False
                _write_json(
                    self.workspace,
                    self.run_id,
                    attempt_file_name(command.attempt_id, "accepted"),
                    {
                        "worker_id": self.worker_id,
                        "generation": self.generation,
                        "attempt_id": command.attempt_id,
                        "job_id": command.job_id,
                    },
                )
        except LockUnavailableError:
            return False
        self.seen_attempts.add(command.attempt_id)
        return True

    def _outcome(self, command: _Command, state: str, reason: str | None) -> None:
        _write_json(
            self.workspace,
            self.run_id,
            attempt_file_name(command.attempt_id),
            {
                "worker_id": self.worker_id,
                "generation": self.generation,
                "attempt_id": command.attempt_id,
                "job_id": command.job_id,
                "state": state,
                "reason": reason,
            },
        )

    def _admit(self, deadline: float) -> None:
        try:
            record = _read_json(self.workspace, self.run_id, worker_file_name(self.worker_id, "command"))
        except FileNotFoundError:
            return
        if record.get("worker_id") != self.worker_id or record.get("generation") != self.generation:
            return
        command = _Command.parse(record)
        if not self._claim(command):
            return
        if command.attempt_id in self.cancelled_attempts:
            self._outcome(command, "failed", "attempt cancelled by coordinator before process launch")
            return
        if time.monotonic() - self.lease_at >= self.lease_timeout_s or time.monotonic() >= deadline:
            self._outcome(command, "failed", "worker lease or lifetime expired before process launch")
            return
        log = None
        lifetime_read: int | None = None
        lifetime_write: int | None = None
        try:
            command.log_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(command.log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
            os.fchmod(descriptor, 0o600)
            log = os.fdopen(descriptor, "ab", buffering=0)
            env = {
                **narrow_accelerator_environment(self.base_env, command.env),
                "MISEN_RUN_ID": self.run_id,
                "MISEN_ATTEMPT_ID": command.attempt_id,
            }
            configuration = json.dumps({"argv": command.argv, "env": env}).encode()
            if len(configuration) > 2 * _MAX_JSON_BYTES:
                msg = "Worker guard configuration exceeds the size limit."
                raise ValueError(msg)  # noqa: TRY301 -- retain the normal accepted-attempt failure path
            lifetime_read, lifetime_write = os.pipe()
            # Guard cleanup (TERM grace plus bounded reap) fits inside half
            # the agent's grace, leaving room before the fallback SIGKILL.
            guard_grace_s = min(5.0, self.shutdown_grace_s / 4)
            guard_deadline = min(deadline, time.monotonic() + command.setup_timeout_s + command.execution_timeout_s)
            process = subprocess.Popen(  # noqa: S603 -- authenticated coordinator supplies argv, never a shell string
                [
                    sys.executable,
                    "-m",
                    "misen.executors.skypilot",
                    "--worker-guard",
                    "--parent-fd",
                    str(lifetime_read),
                    "--deadline",
                    str(guard_deadline),
                    "--grace-s",
                    str(guard_grace_s),
                ],
                env=self.base_env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(lifetime_read,),
            )
            self.active = _ActiveAttempt(command, process, log, time.monotonic(), lifetime_write)
            lifetime_write = None  # Ownership moves to the active attempt only.
            os.close(lifetime_read)
            lifetime_read = None
            framed = struct.pack("!I", len(configuration)) + configuration
            offset = 0
            while offset < len(framed):
                offset += os.write(self.active.lifetime_fd, framed[offset:])
            self._state("running", command.attempt_id)
        except BaseException:
            if self.active is None:
                if log is not None:
                    log.close()
                self._outcome(command, "failed", "worker could not start the task process")
            raise
        finally:
            if lifetime_read is not None:
                os.close(lifetime_read)
            if lifetime_write is not None:
                os.close(lifetime_write)

    def _completion(self, active: _ActiveAttempt) -> tuple[str, str | None]:
        try:
            result = _read_json(self.workspace, self.run_id, attempt_file_name(active.command.attempt_id, "result"))
        except (FileNotFoundError, *_IO_ERRORS, ValueError):
            result = {}
        if active.process.returncode != 0:
            reason = f"task process exited with status {active.process.returncode}; committed outputs are preserved"
            return "failed", reason
        if (
            result.get("attempt_id") != active.command.attempt_id
            or not isinstance(result.get("state"), str)
            or result["state"] not in {"done", "failed"}
        ):
            return "unknown", "task process exited without a matching durable result"
        if result["state"] == "failed":
            return "failed", "task callable reported failure"
        return "done", None

    def _finish(self, *, forced: tuple[str, str] | None = None) -> None:
        active = self.active
        if active is None:
            return
        try:
            os.close(active.lifetime_fd)
            _terminate_group(active.process, self.shutdown_grace_s)
            state, reason = forced if forced is not None else self._completion(active)
            self._outcome(active.command, state, reason)
        finally:
            active.log.close()
            self.active = None

    def _poll_active(self) -> None:
        active = self.active
        if active is None:
            return
        if active.process.poll() is not None:
            self._finish()
            self._state("idle")
            return
        if active.execution_started_at is None:
            try:
                started = _read_json(
                    self.workspace, self.run_id, attempt_file_name(active.command.attempt_id, "started")
                )
            except FileNotFoundError:
                started = {}
            if started.get("attempt_id") == active.command.attempt_id and started.get("state") == "running":
                active.execution_started_at = time.monotonic()
        now = time.monotonic()
        if active.execution_started_at is None:
            expired = now - active.launched_at >= active.command.setup_timeout_s
            reason = "task setup timed out"
        else:
            expired = now - active.execution_started_at >= active.command.execution_timeout_s
            reason = "task execution timed out"
        if expired:
            self._finish(forced=("failed", reason))
            self._state("idle")

    def run(self) -> None:
        deadline = time.monotonic() + self.max_runtime_s
        stop_reason = "worker stopped before task completion"
        try:
            self._state("idle")
            while True:
                lease_stop = self._lease()
                if lease_stop is not None:
                    stop_reason = lease_stop
                    break
                if time.monotonic() - self.lease_at >= self.lease_timeout_s:
                    stop_reason = "coordinator lease expired"
                    break
                if time.monotonic() >= deadline:
                    stop_reason = "worker maximum lifetime expired"
                    break
                self._poll_active()
                if self.active is None and self.lease_sequence >= 0:
                    self._admit(deadline)
                time.sleep(min(self.poll_interval_s, max(0, deadline - time.monotonic())))
        finally:
            try:
                self._finish(forced=("unknown", stop_reason))
            finally:
                self._state("stopped")


def run_worker_agent(
    workspace: Workspace,
    run_id: str,
    worker_id: str,
    *,
    lease_timeout_s: float = 60,
    shutdown_grace_s: float = 30,
    poll_interval_s: float = 0.2,
    max_runtime_s: float = 86400,
) -> None:
    """Run a leased single-slot agent, terminating only its own task processes.

    The parent coordinator must refresh a strictly increasing lease sequence.
    Repeated leases cannot extend lifetime. Setup and callable execution have
    separate deadlines; durable callable success alone does not prove process
    cleanup succeeded. Accepted attempts never replay after a process restart.
    """
    if os.name != "posix":
        msg = "Reusable worker agents require POSIX process-group support."
        raise ValueError(msg)
    if not workspace.supports_job_file_reads():
        msg = "Reusable worker agents require readable workspace coordination files."
        raise ValueError(msg)
    _Agent(
        workspace=workspace,
        run_id=_token(run_id),
        worker_id=_token(worker_id),
        lease_timeout_s=_positive_number(lease_timeout_s),
        shutdown_grace_s=_positive_number(shutdown_grace_s),
        poll_interval_s=_positive_number(poll_interval_s),
        max_runtime_s=_positive_number(max_runtime_s),
    ).run()


# Worker execution claims and durable outcomes
# ----------------------------------------------------------------------------

_RUN_ID_ENV = "MISEN_RUN_ID"


_ATTEMPT_ID_ENV = "MISEN_ATTEMPT_ID"


_ATTEMPT_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


_ERROR_TYPE_LIMIT = 80


_ATTEMPT_RECORD_LIMIT = 4096


_CLAIM_LOCK_TIMEOUT = 30


def _consume_attempt_identity() -> tuple[str, str] | None:
    """Remove and validate optional worker coordination identifiers."""
    run_id = os.environ.pop(_RUN_ID_ENV, None)
    attempt_id = os.environ.pop(_ATTEMPT_ID_ENV, None)
    if run_id is None and attempt_id is None:
        return None
    if (
        run_id is None
        or attempt_id is None
        or _ATTEMPT_TOKEN.fullmatch(run_id) is None
        or _ATTEMPT_TOKEN.fullmatch(attempt_id) is None
    ):
        msg = "MISEN_RUN_ID and MISEN_ATTEMPT_ID must both contain valid bounded coordination identifiers."
        raise ValueError(msg)
    return run_id, attempt_id


def _publish_attempt(
    workspace: Workspace,
    identity: tuple[str, str],
    state: str,
    error: BaseException | None = None,
) -> None:
    """Publish a bounded outcome without copying exception text or credentials."""
    run_id, attempt_id = identity
    record: dict[str, int | str] = {
        "version": 1,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "state": state,
    }
    if error is not None:
        error_type = re.sub(r"[^A-Za-z0-9_]", "_", type(error).__name__)[:_ERROR_TYPE_LIMIT] or "BaseException"
        record["error_type"] = error_type
        # Exception messages can contain credentials, URLs, or whole payloads.
        # Keep them in the existing traceback only, never coordination records.
        record["error_message"] = f"Worker payload raised {error_type}; inspect its task log for details."
    suffix = {"claimed": "execution", "running": "started"}.get(state, "result")
    workspace.put_job_file(run_id, f"attempt-{attempt_id}.{suffix}.json", json.dumps(record).encode("utf-8"))


def _claim_attempt(workspace: Workspace, identity: tuple[str, str]) -> bool:
    """Claim execution once, refusing to replay an uncertain earlier invocation."""
    run_id, attempt_id = identity
    lock_key = "execution-" + hashlib.sha256(json.dumps(identity).encode("utf-8")).hexdigest()
    lock = workspace.lock("job", lock_key)
    with lock.context(timeout=_CLAIM_LOCK_TIMEOUT):
        try:
            result_data = workspace.read_job_file(run_id, f"attempt-{attempt_id}.result.json")
        except FileNotFoundError:
            result_data = None
        if result_data is not None:
            result = None
            if len(result_data) <= _ATTEMPT_RECORD_LIMIT:
                with contextlib.suppress(ValueError, UnicodeError):
                    result = json.loads(result_data)
            if (
                isinstance(result, dict)
                and type(result.get("version")) is int
                and result["version"] == 1
                and result.get("run_id") == run_id
                and result.get("attempt_id") == attempt_id
                and result.get("state") == "done"
            ):
                return False
            msg = "The worker attempt already has an unsuccessful or invalid outcome; refusing to replay execution."
            raise ExecutionError(msg)
        try:
            workspace.read_job_file(run_id, f"attempt-{attempt_id}.execution.json")
        except FileNotFoundError:
            pass
        else:
            msg = "An earlier worker invocation claimed this attempt without committed success; refusing to replay it."
            raise ExecutionError(msg)
        if not lock.is_locked():
            msg = "Lost the execution claim lock before publishing the attempt claim."
            raise ExecutionError(msg)
        _publish_attempt(workspace, identity, "claimed")
    return True


def _execute_attempt(workspace: Workspace, payload_fn: Callable[[], None], identity: tuple[str, str]) -> None:
    """Report execution boundaries without replacing a user-code exception."""
    if not _claim_attempt(workspace, identity):
        return
    _publish_attempt(workspace, identity, "running")
    try:
        payload_fn()
    except BaseException as exc:
        try:
            _publish_attempt(workspace, identity, "failed", exc)
        except BaseException:  # noqa: BLE001 -- retain the original payload traceback
            exc.add_note("Additionally, publishing the failed attempt result did not succeed.")
        raise
    _publish_attempt(workspace, identity, "done")


# Native SkyPilot allocation handles
# ----------------------------------------------------------------------------

_SKYPILOT_INSTALL = 'uv pip install "misen[skypilot]"'


_QUEUE_FIELDS = ("job_id", "task_id", "status", "failure_reason", "end_at")


_RECOVERY_QUEUE_FIELDS = ("job_id", "task_id", "job_name")


_ACTIVE_REQUEST_STATES = frozenset({"PENDING", "WAITING", "RUNNING"})


_CANCELLED_REQUEST_RECONCILE_S = 60


_CONTROLLER_FAILURE_KILL_GRACE_S = 30


_SKYPILOT_STATE_MAP: dict[str, JobState] = {
    **dict.fromkeys(("PENDING", "SUBMITTED", "STARTING"), "pending"),
    **dict.fromkeys(("RUNNING", "WINDING_DOWN", "RECOVERING", "CANCELLING"), "running"),
    "SUCCEEDED": "done",
    **dict.fromkeys(
        (
            "CANCELLED",
            "FAILED",
            "FAILED_SETUP",
            "FAILED_PRECHECKS",
            "FAILED_NO_RESOURCE",
            "FAILED_CONTROLLER",
        ),
        "failed",
    ),
}


def _load_skypilot() -> Any:
    """Load the optional SkyPilot SDK on first use."""
    if (session := active_session()) is not None:
        session.check_open()
        return session.client
    return _load_external_skypilot()


def _load_external_skypilot() -> Any:
    """Load the ambient SDK without changing its endpoint or configuration."""
    try:
        sky = importlib.import_module("sky")
    except ModuleNotFoundError as exc:
        if exc.name != "sky":
            raise
        msg = f"SkyPilotExecutor requires SkyPilot >=0.13; install it with `{_SKYPILOT_INSTALL}`."
        raise ConfigError(msg) from exc
    # Packaging is a dependency of the optional SkyPilot SDK, not imported by
    # ordinary Misen execution. Parse prerelease/nightly versions accurately.
    from packaging.version import InvalidVersion, Version

    version_text = getattr(sky, "__version__", None)
    if not isinstance(version_text, str) or not version_text.strip():
        msg = (
            f"Cannot determine the installed SkyPilot SDK version; install SkyPilot >=0.13 with `{_SKYPILOT_INSTALL}`."
        )
        raise ConfigError(msg)
    try:
        version = Version(version_text)
    except InvalidVersion as exc:
        msg = f"Cannot parse the installed SkyPilot SDK version; install SkyPilot >=0.13 with `{_SKYPILOT_INSTALL}`."
        raise ConfigError(msg) from exc
    if version < Version("0.13"):
        msg = f"SkyPilotExecutor requires SkyPilot >=0.13; upgrade it with `{_SKYPILOT_INSTALL}`."
        raise ConfigError(msg)
    return sky


def _field(record: object, name: str, default: Any = None) -> Any:
    """Read one field from a dict or a SkyPilot response model."""
    if isinstance(record, dict):
        return cast("dict[str, Any]", record).get(name, default)
    return getattr(record, name, default)


def _status_name(value: object) -> str:
    """Normalize SkyPilot enum/string status values to an uppercase name."""
    if isinstance(value, Enum):
        value = value.value
    text = str(value or "")
    return text.rsplit(".", 1)[-1].upper()


def _normalize_skypilot_state(value: object) -> JobState:
    """Map a SkyPilot managed-job status to Misen's lifecycle."""
    status = _status_name(value)
    if status.startswith("FAILED"):
        return "failed"
    return _SKYPILOT_STATE_MAP.get(status, "unknown")


def _queue_records(result: object) -> list[object]:
    """Extract queue records from SkyPilot's queue_v2 response."""
    records = result[0] if isinstance(result, tuple) and result else None
    if not isinstance(records, (list, tuple)):
        msg = f"SkyPilot queue_v2 returned an unexpected response: {result!r}"
        raise StatusQueryError(msg, retryable=False)
    return list(records)


class SkyPilotJob(Job):
    """One Misen work unit backed by one SkyPilot managed job."""

    __slots__ = (
        "_api_session",
        "_managed_job_id_persisted",
        "_terminal_state",
        "deadline_minutes",
        "managed_job_id",
        "managed_job_name",
        "request_id",
        "submission_id",
        "workspace",
    )

    def __init__(
        self,
        *,
        work_unit: WorkUnit,
        job_id: str,
        managed_job_id: int | None,
        submission_id: str,
        deadline_minutes: int,
        log_path: Path,
        workspace: Workspace,
        request_id: str | None = None,
        managed_job_name: str | None = None,
    ) -> None:
        """Initialize a handle for a launch request or resolved managed job."""
        if managed_job_id is None and request_id is None:
            msg = "A SkyPilot job requires a launch request ID or managed-job ID."
            raise ValueError(msg)
        super().__init__(work_unit=work_unit, job_id=job_id, log_path=log_path)
        self.managed_job_id = managed_job_id
        self._managed_job_id_persisted = managed_job_id is not None
        self.managed_job_name = managed_job_name
        self.request_id = request_id
        self.submission_id = submission_id
        self.deadline_minutes = deadline_minutes
        self.workspace = workspace
        self._terminal_state: JobState | None = None
        self._api_session = active_session()
        if self._api_session is not None:
            self._api_session.jobs.append(self)

    def state(self) -> JobState:
        """Return this managed job's normalized SkyPilot state."""
        return type(self).bulk_state([self]).get(self, "unknown")

    def cancel(self) -> None:
        """Cancel an unresolved launch request or its assigned managed job."""
        if self._api_session is not None:
            self._api_session.check_open()
        sky = (
            self._api_session.client
            if self._api_session is not None
            else (_load_skypilot() if active_session() is None else _load_external_skypilot())
        )
        try:
            self._cancel(sky)
        except ExecutionError:
            raise
        except Exception as exc:
            identity = self.managed_job_id if self.managed_job_id is not None else self.request_id
            msg = f"Could not cancel SkyPilot job {identity}: {exc}"
            raise ExecutionError(msg) from exc

    def _cancel(self, sky: Any) -> None:
        """Resolve an accepted launch, then cancel its managed job safely."""
        managed_job_id = self.managed_job_id
        resolved_here = managed_job_id is None
        if managed_job_id is None:
            # SkyPilot marks a launch request CANCELLED immediately after
            # sending SIGTERM, before its handler is guaranteed to quiesce.
            # Waiting for the launch result here is slower, but prevents a
            # managed job accepted during that race from being orphaned.
            managed_job_id = self._resolve_managed_job_id(sky, persist=False, missing_ok=True)
            if managed_job_id is None:
                # A terminal request plus a successful exact-name recovery
                # query with no match means no managed job was accepted.
                return
        try:
            sky.get(sky.jobs.cancel(job_ids=[managed_job_id]))
        except Exception:
            if resolved_here:
                # Keep the durable record provisional so a retry re-resolves
                # the launch instead of assuming cancellation succeeded.
                self.managed_job_id = None
                self._managed_job_id_persisted = False
            raise
        if not self._managed_job_id_persisted:
            # Cancel first: losing durable-storage access must never prevent
            # cancellation of a managed job whose ID is already known.
            self._remember_managed_job_id(managed_job_id)

    @classmethod
    def _from_record(cls, work_unit: WorkUnit, workspace: Workspace, record: _JobRecord) -> SkyPilotJob:
        native_id = record.native_id
        managed_job_id = native_id if isinstance(native_id, int) and not isinstance(native_id, bool) else None
        request_id = record.request_id
        if isinstance(native_id, str) and request_id is None:
            # Legacy records accepted numeric strings as managed-job IDs.
            # Non-numeric strings are provisional launch request IDs.
            try:
                managed_job_id = int(native_id)
            except ValueError:
                request_id = native_id
        if managed_job_id is None and request_id is None:
            msg = f"SkyPilot durable job record {record.job_id!r} has no usable native identity."
            raise StorageError(msg)
        return cls(
            work_unit=work_unit,
            job_id=record.job_id,
            managed_job_id=managed_job_id,
            submission_id=record.submission_id,
            deadline_minutes=record.deadline_minutes,
            log_path=workspace.get_job_log(record.job_id, work_unit),
            workspace=workspace,
            request_id=request_id,
            managed_job_name=record.native_name,
        )

    def _record(self) -> _JobRecord:
        native_id: str | int = self.managed_job_id if self.managed_job_id is not None else cast("str", self.request_id)
        return _JobRecord(
            cast("str", self.job_id),
            native_id,
            self.submission_id,
            self.deadline_minutes,
            request_id=self.request_id,
            native_name=self.managed_job_name,
        )

    def _resolve_managed_job_id(
        self,
        sky: Any,
        *,
        persist: bool = True,
        missing_ok: bool = False,
    ) -> int | None:
        """Resolve this launch request's managed-job ID, optionally persisting it."""
        if self.managed_job_id is not None:
            if persist and not self._managed_job_id_persisted:
                return self._remember_managed_job_id(self.managed_job_id)
            return self.managed_job_id
        if self.request_id is None:  # guarded by construction and record decoding
            msg = f"SkyPilot job {self.label} has no launch request ID."
            raise StatusQueryError(msg, retryable=False)
        try:
            launch_result = sky.get(self.request_id)
        except Exception as exc:
            try:
                recovered = type(self)._recover_managed_job_ids(sky, [self], persist=persist)  # noqa: SLF001
            except StatusQueryError as recovery_exc:
                recovery_exc.add_note(f"The original launch-request lookup failed with: {exc}")
                raise recovery_exc from exc
            if recovered:
                return cast("int", self.managed_job_id)
            if missing_ok:
                try:
                    records = sky.api_status(request_ids=[self.request_id])
                except Exception as status_exc:
                    msg = f"Could not determine whether SkyPilot launch request {self.request_id!r} accepted a job."
                    error = StatusQueryError(msg)
                    error.add_note(f"The launch-result lookup failed with: {exc}")
                    raise error from status_exc
                record = next(
                    (record for record in records if _field(record, "request_id") == self.request_id),
                    None,
                )
                if _status_name(_field(record, "status")) == "FAILED":
                    return None
            msg = f"Could not resolve SkyPilot launch request {self.request_id!r} for {self.label}: {exc}"
            raise StatusQueryError(msg) from exc

        managed_ids = launch_result[0] if isinstance(launch_result, tuple) and launch_result else None
        if (
            not isinstance(managed_ids, (list, tuple))
            or len(managed_ids) != 1
            or not isinstance(managed_ids[0], int)
            or isinstance(managed_ids[0], bool)
            or managed_ids[0] < 1
        ):
            msg = (
                f"SkyPilot launch request {self.request_id!r} for {self.label} returned an unexpected result: "
                f"{launch_result!r}."
            )
            raise StatusQueryError(msg, retryable=False)

        if persist:
            return self._remember_managed_job_id(managed_ids[0])
        self.managed_job_id = managed_ids[0]
        self._managed_job_id_persisted = False
        return managed_ids[0]

    def _remember_managed_job_id(self, managed_job_id: int) -> int:
        """Store a resolved managed-job ID in memory and durable storage."""
        self.managed_job_id = managed_job_id
        self._managed_job_id_persisted = False
        try:
            self._refresh_record()
        except (MisenError, OSError) as exc:
            msg = f"Could not persist managed-job ID {managed_job_id} for {self.label}: {exc}"
            raise StatusQueryError(msg) from exc
        self._managed_job_id_persisted = True
        logger.info(
            "Resolved SkyPilot launch request %s to managed job %d.",
            self.request_id,
            managed_job_id,
        )
        return managed_job_id

    @classmethod
    def _recover_managed_job_ids(
        cls,
        sky: Any,
        jobs: Sequence[SkyPilotJob],
        *,
        refresh: bool = True,
        persist: bool = True,
    ) -> list[SkyPilotJob]:
        """Recover managed IDs by exact launch name after request metadata expires."""
        named_jobs = [job for job in jobs if job.managed_job_name is not None]
        if not named_jobs:
            return []
        try:
            queue_request_id = sky.jobs.queue_v2(
                refresh=refresh,
                fields=_RECOVERY_QUEUE_FIELDS,
            )
            records = _queue_records(sky.get(queue_request_id))
        except StatusQueryError:
            raise
        except Exception as exc:
            names = [cast("str", job.managed_job_name) for job in named_jobs]
            msg = f"Could not recover SkyPilot managed jobs by name {names}: {exc}"
            raise StatusQueryError(msg) from exc

        ids_by_name: dict[str, list[int]] = {}
        for record in records:
            raw_job_id = _field(record, "job_id")
            raw_task_id = _field(record, "task_id")
            raw_name = _field(record, "job_name")
            if isinstance(raw_job_id, int) and raw_task_id in (0, None) and isinstance(raw_name, str):
                ids_by_name.setdefault(raw_name, []).append(raw_job_id)

        target_names = {cast("str", job.managed_job_name) for job in named_jobs}
        ambiguous = {
            name: job_ids for name, job_ids in ids_by_name.items() if name in target_names and len(job_ids) > 1
        }
        if ambiguous:
            msg = f"Multiple SkyPilot managed jobs matched durable launch names: {ambiguous}."
            raise StatusQueryError(msg, retryable=False)

        recovered: list[SkyPilotJob] = []
        for job in named_jobs:
            matching_ids = ids_by_name.get(cast("str", job.managed_job_name), [])
            if len(matching_ids) == 1:
                if persist:
                    job._remember_managed_job_id(matching_ids[0])  # noqa: SLF001
                else:
                    job.managed_job_id = matching_ids[0]
                    job._managed_job_id_persisted = False  # noqa: SLF001
                recovered.append(job)
        return recovered

    @staticmethod
    def _remember_terminal_state(job: SkyPilotJob, state: JobState) -> None:
        """Finalize and cache one terminal state already present in storage."""
        job._finalize_log(job.workspace, failed=state == "failed")
        job._terminal_state = state

    @classmethod
    def _cache_terminal_state(cls, job: SkyPilotJob, state: JobState) -> JobState:
        """Publish, finalize, and cache one terminal dependency state."""
        if state == "failed" and cls._workspace_terminal_state(job) == "done":
            state = "done"
        try:
            published = publish_dependency_state(
                job.workspace,
                job.submission_id,
                cast("str", job.job_id),
                state.encode(),
            )
        except (MisenError, OSError) as exc:
            msg = f"Could not publish terminal dependency state for {job.label}: {exc}"
            raise StatusQueryError(msg) from exc
        authoritative_state = cast("JobState", published.decode())
        cls._remember_terminal_state(job, authoritative_state)
        return authoritative_state

    @staticmethod
    def _workspace_terminal_state(job: SkyPilotJob) -> JobState | None:
        """Read a terminal worker/controller marker for request-GC recovery."""
        marker: bytes | None = None
        try:
            marker = job.workspace.read_job_file(
                job.submission_id,
                dependency_state_name(cast("str", job.job_id)),
            )
        except FileNotFoundError:
            pass
        except (MisenError, OSError) as exc:
            msg = f"Could not verify the completion marker for {job.label}: {exc}"
            raise StatusQueryError(msg) from exc
        if marker == b"done":
            return "done"
        if marker == b"failed":
            return "failed"
        try:
            if job.work_unit.done(workspace=job.workspace):
                return "done"
        except (MisenError, OSError) as exc:
            msg = f"Could not verify workspace completion for {job.label}: {exc}"
            raise StatusQueryError(msg) from exc
        return None

    @classmethod
    def _resolve_launch_requests(
        cls,
        sky: Any,
        jobs: Sequence[SkyPilotJob],
        result: dict[Job, JobState],
    ) -> list[SkyPilotJob]:
        """Resolve completed launch requests without blocking on active ones."""
        request_ids = [cast("str", job.request_id) for job in jobs]
        try:
            records = sky.api_status(request_ids=request_ids)
        except Exception as exc:
            msg = f"Could not query SkyPilot launch requests {request_ids}: {exc}"
            raise StatusQueryError(msg) from exc
        if not isinstance(records, (list, tuple)):
            msg = f"SkyPilot api_status returned an unexpected response: {records!r}"
            raise StatusQueryError(msg, retryable=False)

        by_request_id = {
            raw_request_id: record
            for record in records
            if isinstance((raw_request_id := _field(record, "request_id")), str)
        }
        resolved: list[SkyPilotJob] = []
        missing: list[SkyPilotJob] = []
        terminal_requests: dict[SkyPilotJob, tuple[str, object]] = {}
        for job in jobs:
            record = by_request_id.get(job.request_id)
            status = _status_name(_field(record, "status"))
            if status in _ACTIVE_REQUEST_STATES:
                result[job] = "pending"
                continue
            if status == "SUCCEEDED":
                job._resolve_managed_job_id(sky)  # noqa: SLF001
                resolved.append(job)
                continue
            if status in {"FAILED", "CANCELLED"}:
                terminal_state = cls._workspace_terminal_state(job)
                if terminal_state is not None:
                    if status == "CANCELLED" and terminal_state == "failed":
                        # This may be the guard marker published by an earlier
                        # reconciliation poll; keep looking for a late ID until
                        # SkyPilot's durable cancellation window expires.
                        terminal_requests[job] = (status, record)
                        continue
                    authoritative_state = cls._cache_terminal_state(job, terminal_state)
                    if authoritative_state == "failed":
                        job._record_failure(  # noqa: SLF001
                            f"SkyPilot launch request {job.request_id} reported {status}; "
                            "the workspace recorded the job as failed."
                        )
                    result[job] = authoritative_state
                    continue
                terminal_requests[job] = (status, record)
                continue
            terminal_state = cls._workspace_terminal_state(job)
            if terminal_state is None:
                missing.append(job)
                continue
            authoritative_state = cls._cache_terminal_state(job, terminal_state)
            if authoritative_state == "failed":
                job._record_failure(  # noqa: SLF001
                    f"SkyPilot launch request {job.request_id} is no longer retained; "
                    "the workspace recorded the job as failed."
                )
            result[job] = authoritative_state

        recoverable = [
            *missing,
            *(job for job, (status, _) in terminal_requests.items() if status == "FAILED"),
        ]
        recovered = cls._recover_managed_job_ids(sky, recoverable)
        recovered_set = set(recovered)
        resolved.extend(recovered)
        for job, (status, request_record) in terminal_requests.items():
            if status != "CANCELLED":
                continue
            try:
                published = publish_dependency_state(
                    job.workspace,
                    job.submission_id,
                    cast("str", job.job_id),
                    b"failed",
                )
            except (MisenError, OSError) as exc:
                msg = f"Could not publish a cancellation gate for {job.label}: {exc}"
                raise StatusQueryError(msg) from exc
            authoritative_state = cast("JobState", published.decode())
            if authoritative_state == "done":
                cls._remember_terminal_state(job, "done")
                result[job] = "done"
                continue
            try:
                # External api_cancel marks a request CANCELLED before its
                # handler is guaranteed to quiesce. Recover and cancel an ID
                # that is already visible while the failed worker gate makes
                # any late new-protocol worker exit without user code.
                recovered_cancelled = cls._recover_managed_job_ids(
                    sky,
                    [job],
                    refresh=False,
                    persist=False,
                )
            except StatusQueryError as exc:
                job.managed_job_id = None
                job._managed_job_id_persisted = False  # noqa: SLF001
                if exc.retryable:
                    logger.warning(
                        "Could not yet reconcile externally cancelled SkyPilot request %s: %s",
                        job.request_id,
                        exc,
                    )
                    recovered_cancelled = []
                else:
                    msg = f"Could not cancel managed job recovered from launch request {job.request_id}: {exc}"
                    raise StatusQueryError(msg, retryable=False) from exc
            except Exception as exc:
                job.managed_job_id = None
                job._managed_job_id_persisted = False  # noqa: SLF001
                msg = f"Could not cancel managed job recovered from launch request {job.request_id}: {exc}"
                raise StatusQueryError(msg) from exc
            if not recovered_cancelled:
                finished_at = _field(request_record, "finished_at")
                still_reconciling = (
                    isinstance(finished_at, (int, float))
                    and not isinstance(finished_at, bool)
                    and time.time() - finished_at < _CANCELLED_REQUEST_RECONCILE_S
                )
                if still_reconciling:
                    result[job] = "pending"
                    continue
                authoritative_state = cls._cache_terminal_state(job, "failed")
                if authoritative_state == "failed":
                    job._record_failure(  # noqa: SLF001
                        f"SkyPilot launch request {job.request_id} was cancelled before assigning a managed-job ID."
                    )
                result[job] = authoritative_state
                continue
            managed_job_id = cast("int", job.managed_job_id)
            try:
                sky.get(sky.jobs.cancel(job_ids=[managed_job_id]))
            except Exception as exc:
                job.managed_job_id = None
                job._managed_job_id_persisted = False  # noqa: SLF001
                msg = f"Could not cancel managed job recovered from launch request {job.request_id}: {exc}"
                raise StatusQueryError(msg) from exc
            job._remember_managed_job_id(managed_job_id)  # noqa: SLF001
            recovered_set.add(job)
            resolved.append(job)
        for job, (status, _) in terminal_requests.items():
            if job in recovered_set:
                continue
            if status == "CANCELLED":
                continue
            detail = ""
            try:
                sky.get(job.request_id)
            except Exception as exc:  # noqa: BLE001 - expected server-side failure detail
                detail = f": {type(exc).__name__}: {exc}"
            authoritative_state = cls._cache_terminal_state(job, "failed")
            if authoritative_state == "failed":
                job._record_failure(  # noqa: SLF001
                    f"SkyPilot launch request {job.request_id} reported {status}{detail}."
                )
            result[job] = authoritative_state
        for job in missing:
            if job in recovered_set:
                continue
            result[job] = "unknown"
        return resolved

    @classmethod
    def bulk_state(cls, jobs: Sequence[Job]) -> dict[Job, JobState]:
        """Resolve launch requests and query managed jobs in batches."""
        if not jobs:
            return {}
        skypilot_jobs = cast("Sequence[SkyPilotJob]", jobs)
        result: dict[Job, JobState] = {
            job: job._terminal_state  # noqa: SLF001
            for job in skypilot_jobs
            if job._terminal_state is not None  # noqa: SLF001
        }
        active_jobs = [job for job in skypilot_jobs if job._terminal_state is None]  # noqa: SLF001
        if not active_jobs:
            return result

        groups: dict[Any, list[SkyPilotJob]] = {}
        for job in active_jobs:
            groups.setdefault(job._api_session, []).append(job)  # noqa: SLF001
        if len(groups) > 1:
            for group in groups.values():
                result.update(cls.bulk_state(group))
            return result
        session = active_jobs[0]._api_session  # noqa: SLF001
        if session is not None:
            session.check_open()
        sky = (
            session.client
            if session is not None
            else (_load_skypilot() if active_session() is None else _load_external_skypilot())
        )
        for job in active_jobs:
            if job.managed_job_id is not None and not job._managed_job_id_persisted:  # noqa: SLF001
                job._remember_managed_job_id(job.managed_job_id)  # noqa: SLF001
        unresolved_requests = [job for job in active_jobs if job.managed_job_id is None]
        resolved_jobs = [job for job in active_jobs if job.managed_job_id is not None]
        if unresolved_requests:
            resolved_jobs.extend(cls._resolve_launch_requests(sky, unresolved_requests, result))
        if not resolved_jobs:
            return result

        managed_ids = sorted({cast("int", job.managed_job_id) for job in resolved_jobs})
        try:
            request_id = sky.jobs.queue_v2(
                # Managed-jobs controllers autostop. Refresh makes old handles
                # queryable instead of turning a stopped controller into an
                # indefinitely unknown Misen state.
                refresh=True,
                job_ids=managed_ids,
                fields=_QUEUE_FIELDS,
            )
            records = _queue_records(sky.get(request_id))
        except StatusQueryError:
            raise
        except Exception as exc:
            msg = f"Could not query SkyPilot managed jobs {managed_ids}: {exc}"
            raise StatusQueryError(msg) from exc
        by_job_id: dict[int, object] = {}
        for record in records:
            raw_job_id = _field(record, "job_id")
            raw_task_id = _field(record, "task_id")
            if isinstance(raw_job_id, int) and raw_task_id in (0, None):
                by_job_id[raw_job_id] = record

        for job in resolved_jobs:
            record = by_job_id.get(job.managed_job_id)
            state = _normalize_skypilot_state(_field(record, "status"))
            raw_status = _status_name(_field(record, "status"))

            if record is None or state == "unknown":
                terminal_state = cls._workspace_terminal_state(job)
                if terminal_state is not None:
                    state = cls._cache_terminal_state(job, terminal_state)
                    if state == "failed":
                        job._record_failure(  # noqa: SLF001
                            f"SkyPilot no longer reports managed job {job.managed_job_id}; "
                            "the workspace recorded the job as failed."
                        )
                    result[job] = state
                    continue

            # A failed jobs controller can stop reporting while its worker is
            # still finishing. Give any worker its full command timeout from
            # SkyPilot's durable failure timestamp before the controller
            # publishes a competing failure marker.
            if raw_status == "FAILED_CONTROLLER":
                terminal_state = cls._workspace_terminal_state(job)
                if terminal_state is None:
                    end_at = _field(record, "end_at")
                    grace_s = job.deadline_minutes * 60 + _CONTROLLER_FAILURE_KILL_GRACE_S
                    if (
                        isinstance(end_at, (int, float))
                        and not isinstance(end_at, bool)
                        and time.time() - end_at < grace_s
                    ):
                        result[job] = "running"
                        continue
                if terminal_state is not None:
                    state = terminal_state

            if state in {"done", "failed"}:
                state = cls._cache_terminal_state(job, state)
            if state == "failed":
                raw_status = raw_status or "FAILED"
                reason = _field(record, "failure_reason")
                detail = f": {reason}" if isinstance(reason, str) and reason else ""
                job._record_failure(  # noqa: SLF001
                    f"SkyPilot managed job {job.managed_job_id} reported {raw_status}{detail}."
                )
            result[job] = state
        return result


# Graph coordinator and executor
# ----------------------------------------------------------------------------

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


_runs: contextvars.ContextVar[tuple[int, list[GraphCoordinator]] | None] = contextvars.ContextVar(
    "misen_graph_runs", default=None
)


_HEALTH_INTERVAL = 10.0


_LEASE_INTERVAL = 10.0


_CONTROL_RECORD_LIMIT = 2_000_000


def _read(workspace: Workspace, run_id: str, name: str) -> dict[str, Any] | None:
    try:
        data = workspace.read_job_file(run_id, name)
    except FileNotFoundError:
        return None
    if len(data) > _CONTROL_RECORD_LIMIT:
        msg = "Oversized graph control record."
        raise StorageError(msg)
    try:
        result = msgspec.json.decode(data, type=dict[str, Any])
    except msgspec.DecodeError as exc:
        msg = "Malformed graph control record."
        raise StorageError(msg) from exc
    if result.get("version") != 1 or result.get("run_id") != run_id:
        msg = "Graph control record has an unsupported version or wrong run identity."
        raise StorageError(msg)
    return result


def _write(workspace: Workspace, run_id: str, filename: str, **data: Any) -> None:
    workspace.put_job_file(run_id, filename, msgspec.json.encode({"version": 1, "run_id": run_id, **data}))


def _async_call(function: Callable[..., Any], *args: Any) -> Future[Any]:
    """Run one bounded-by-capacity SDK operation off the scheduling loop."""
    future: Future[Any] = Future()

    def invoke() -> None:
        if not future.set_running_or_notify_cancel():
            return
        try:
            future.set_result(function(*args))
        except BaseException as exc:  # noqa: BLE001 -- propagated through the future to the coordinator
            future.set_exception(exc)

    threading.Thread(
        target=contextvars.copy_context().run, args=(invoke,), daemon=True, name="misen-capacity-operation"
    ).start()
    return future


class SkyPilotTaskJob(Job):
    """Logical handle that exists before an attempt or native allocation."""

    __slots__ = ("coordinator", "run_id", "stale_timeout_s", "workspace")

    def __init__(self, work_unit: WorkUnit, job_id: str, log_path: Path, workspace: Workspace, run_id: str) -> None:
        """Bind logical identity to a durable run, before any native allocation."""
        super().__init__(work_unit, job_id, log_path)
        self.workspace = workspace
        self.run_id = run_id
        self.coordinator: GraphCoordinator | None = None
        self.stale_timeout_s = 600.0

    def state(self) -> JobState:
        """Observe the graph; never advance scheduling from a polling call."""
        return self.bulk_state([self])[self]

    @classmethod
    def bulk_state(cls, jobs: Sequence[Job]) -> dict[Job, JobState]:
        """Read one coalesced index per run, or attached in-memory state."""
        runs: dict[tuple[int, str], RunState] = {}
        result: dict[Job, JobState] = {}
        for raw in jobs:
            job = cast("SkyPilotTaskJob", raw)
            key = (id(job.workspace), job.run_id)
            if key not in runs:
                runs[key] = (
                    job.coordinator.snapshot_state()
                    if job.coordinator is not None
                    else read_run_state(job.workspace, job.run_id)
                )
            state = runs[key].jobs[cast("str", job.job_id)]
            if runs[key].cleanup_errors:
                msg = f"Run {job.run_id} has unresolved cleanup: {' '.join(runs[key].cleanup_errors)}"
                raise StatusQueryError(msg, retryable=False)
            if (
                job.coordinator is None
                and state.state not in ("done", "failed")
                and time.time() - runs[key].heartbeat_at > job.stale_timeout_s
            ):
                job._record_failure(  # noqa: SLF001 -- bulk observation is the Job subclass state boundary
                    "Coordinator heartbeat expired; inspect the run and allocation records before resubmission."
                )
                result[job] = "unknown"
                continue
            if state.reason and state.state in ("failed", "unknown"):
                job._record_failure(state.reason)  # noqa: SLF001 -- same subclass state boundary
            result[job] = state.state
        return result

    def cancel(self) -> None:
        """Cancel this logical unit and its descendants, not borrowed capacity."""
        with self.workspace.lock("job", f"cancel-{self.run_id}").context(timeout=30):
            current = _read(self.workspace, self.run_id, "cancellations.json") or {}
            cancelled = set(current.get("job_ids", []))
            cancelled.add(self.job_id)
            _write(self.workspace, self.run_id, "cancellations.json", job_ids=sorted(cancelled))
        if self.coordinator is not None:
            self.coordinator.wakeup.set()


@dataclass
class _Allocation:
    worker_id: str
    profile: str
    launch: Future[Any]
    dedicated: bool = False
    native: Any = None
    health: Future[Any] | None = None
    generation: str | None = None
    job_id: str | None = None
    attempt_id: str | None = None
    last_health: float = 0
    started_at: float = 0
    retired: bool = False
    native_done: bool = False
    cancel_attempt_id: str | None = None
    execution_started_at: float | None = None
    cancel_requested: bool = False
    cancellation: Future[Any] | None = None
    cleanup_reported: bool = False


class GraphCoordinator:
    """Session-owned graph progress independent of UI and SDK health polling."""

    def __init__(
        self, executor: GraphSkyPilotExecutor, manifest: RunManifest, workspace: Workspace, backend: Any
    ) -> None:
        """Create one coordinator with explicit backend and workspace ownership."""
        self.executor = executor
        self.manifest = manifest
        self.workspace = workspace
        self.backend = backend
        self.graph = ReadyGraph(manifest.nodes)
        self.allocations: dict[str, _Allocation] = {}
        self.used_agents: set[str] = set()
        self.wakeup = threading.Event()
        self.finished = threading.Event()
        self.cancelled = threading.Event()
        self.lock = threading.RLock()
        self.errors: list[str] = []
        self.sequence = 0
        self.last_lease = 0.0
        self.started_at = time.monotonic()
        self.heartbeat_at = time.time()
        self.thread: threading.Thread | None = None
        self._last_state: bytes | None = None
        self._last_revision: tuple[Any, ...] | None = None
        self.ready: dict[str, deque[str]] = {name: deque() for name in executor.capacity}
        self.processed_cancellations: set[str] = set()

    def snapshot_state(self) -> RunState:
        """Return an immutable copy for handles on other threads."""
        with self.lock:
            status: Literal["running", "done", "failed", "interrupted"] = "running"
            if self.finished.is_set():
                if self.graph.complete:
                    status = (
                        "failed"
                        if self.errors or any(s.state == "failed" for s in self.graph.states.values())
                        else "done"
                    )
                else:
                    status = "interrupted"
            return msgspec.json.decode(
                msgspec.json.encode(
                    RunState(
                        self.manifest.run_id,
                        self.graph.states,
                        status,
                        list(self.errors),
                        heartbeat_at=self.heartbeat_at,
                    )
                ),
                type=RunState,
            )

    def persist(self) -> None:
        """Publish changed state and heartbeat without rewriting unchanged records."""
        revision = (self.graph.revision, tuple(self.errors), self.finished.is_set(), self.heartbeat_at)
        if revision == self._last_revision:
            return
        state = msgspec.json.encode(self.snapshot_state())
        if state != self._last_state:
            self.workspace.put_job_file(self.manifest.run_id, "run-state.json", state)
            self._last_state = state
        self._last_revision = revision

    def start(self) -> None:
        """Start scheduling in the owning session's context, independently of polling."""
        self.persist()
        self.thread = threading.Thread(
            target=contextvars.copy_context().run, args=(self.run,), name="misen-graph-coordinator", daemon=True
        )
        self.thread.start()

    def _lease(self, *, stop: bool = False) -> None:
        self.sequence += 1
        for worker in self.allocations.values():
            if worker.dedicated or worker.retired:
                continue
            _write(
                self.workspace,
                self.manifest.run_id,
                f"worker-{worker.worker_id}.lease.json",
                worker_id=worker.worker_id,
                sequence=self.sequence,
                stop=stop,
                cancel_attempt_id=worker.cancel_attempt_id,
            )
        self.last_lease = time.monotonic()
        self.heartbeat_at = time.time()

    def _attempt_record(self, worker: _Allocation, suffix: str) -> dict[str, Any] | None:
        if worker.attempt_id is None:
            return None
        record = _read(self.workspace, self.manifest.run_id, f"attempt-{worker.attempt_id}{suffix}.json")
        if record is not None and record.get("attempt_id") != worker.attempt_id:
            msg = "Attempt outcome has the wrong identity."
            raise StorageError(msg)
        return record

    def _finish_attempt(self, worker: _Allocation, record: dict[str, Any]) -> None:
        if worker.job_id is None or worker.attempt_id is None:
            return
        state = record.get("state")
        if state not in ("done", "failed", "unknown"):
            return
        if state != "done" and self.graph.states[worker.job_id].state == "done" and not worker.cleanup_reported:
            self.errors.append(
                f"Attempt {worker.attempt_id} committed its result but subsequent process cleanup failed."
            )
            worker.cleanup_reported = True
        self.graph.apply_result(
            worker.job_id,
            worker.attempt_id,
            success=state == "done",
            reason=None
            if state == "done"
            else str(
                record.get("reason")
                or record.get("error_message")
                or "Attempt failed; it will not be replayed automatically."
            ),
        )

    def _observe(self, worker: _Allocation) -> None:
        if worker.retired:
            return
        if worker.native is None and worker.launch.done():
            try:
                worker.native = worker.launch.result()
            except Exception as exc:  # noqa: BLE001 -- persist launch failure and cancel accepted handles
                worker.retired = True
                accepted = getattr(exc, "submitted_jobs", ())
                for native in accepted:
                    _async_call(self.backend.cancel, native)
                if accepted:
                    self.errors.append(
                        f"Accepted allocation {worker.worker_id} could not be recorded; cancellation requested."
                    )
                healthy = any(
                    other is not worker and other.profile == worker.profile and not other.retired
                    for other in self.allocations.values()
                )
                if worker.job_id is not None:
                    self.graph.finish(
                        worker.job_id, success=False, reason=f"Capacity launch failed: {type(exc).__name__}."
                    )
                elif not healthy:
                    self._fail_profile(worker.profile, f"Capacity launch failed: {type(exc).__name__}.")
                return
        if worker.job_id is not None:
            if self._attempt_record(worker, ".started") is not None:
                self.graph.running(worker.job_id, cast("str", worker.attempt_id))
                if worker.execution_started_at is None:
                    worker.execution_started_at = time.monotonic()
            result = self._attempt_record(worker, ".result")
            if result is not None:
                self._finish_attempt(worker, result)
        if not worker.dedicated:
            record = _read(self.workspace, self.manifest.run_id, f"worker-{worker.worker_id}.state.json")
            if record is not None:
                generation = record.get("generation")
                if record.get("worker_id") != worker.worker_id or not isinstance(generation, str) or not generation:
                    msg = "Invalid worker identity."
                    raise StorageError(msg)
                if worker.generation is not None and worker.generation != generation:
                    if worker.job_id is not None:
                        if self.graph.states[worker.job_id].state == "done":
                            self._finish_attempt(worker, {"state": "unknown"})
                        self.graph.finish(
                            worker.job_id,
                            success=False,
                            reason="Worker restarted during an uncertain attempt; automatic replay is disabled.",
                        )
                    worker.job_id = worker.attempt_id = None
                    worker.retired = True
                    if worker.native is not None:
                        worker.cancellation = _async_call(self.backend.cancel, worker.native)
                    if not any(
                        other is not worker and other.profile == worker.profile and not other.retired
                        for other in self.allocations.values()
                    ):
                        self._fail_profile(
                            worker.profile, "Worker generation changed; automatic replacement is disabled."
                        )
                    return
                worker.generation = generation
                if worker.attempt_id is not None:
                    outcome = self._attempt_record(worker, "")
                    if (
                        outcome is not None
                        and outcome.get("generation") == generation
                        and outcome.get("worker_id") == worker.worker_id
                    ):
                        self._finish_attempt(worker, outcome)
                        if record.get("state") in ("idle", "stopped"):
                            worker.job_id = worker.attempt_id = None
                            worker.cancel_attempt_id = None
                if record.get("state") == "stopped":
                    worker.retired = True
                    if worker.job_id is not None:
                        if self.graph.states[worker.job_id].state == "done":
                            self._finish_attempt(worker, {"state": "unknown"})
                        self.graph.finish(
                            worker.job_id, success=False, reason="Worker stopped before its attempt completed."
                        )
        now = time.monotonic()
        if worker.health is not None and worker.health.done():
            try:
                state = worker.health.result()
            except Exception as exc:  # noqa: BLE001 -- health uncertainty is logged, never fabricated failure
                logger.warning("Allocation health unavailable for %s: %s", worker.worker_id, type(exc).__name__)
            else:
                if state in ("done", "failed"):
                    worker.retired = True
                    worker.native_done = True
                    if worker.job_id is not None:
                        # A commit may have appeared since this iteration's
                        # first read. Native failure must not preempt that
                        # durable success or incorrectly fail its descendants.
                        result = self._attempt_record(worker, ".result")
                        if result is not None:
                            self._finish_attempt(worker, result)
                        elif worker.job_id not in self.graph.finished:
                            self.graph.finish(
                                worker.job_id,
                                success=False,
                                reason="Allocation ended without a committed attempt outcome; no automatic replay.",
                            )
                        if state == "failed":
                            self._finish_attempt(worker, {"state": "failed"})
                    if (
                        worker.generation is None
                        and not worker.dedicated
                        and not any(
                            other is not worker and other.profile == worker.profile and not other.retired
                            for other in self.allocations.values()
                        )
                    ):
                        self._fail_profile(worker.profile, "Worker allocation ended before agent startup.")
            worker.health = None
        if (
            worker.native is not None
            and not worker.retired
            and worker.health is None
            and now - worker.last_health >= _HEALTH_INTERVAL
        ):
            worker.health = _async_call(self.backend.state, worker.native)
            worker.last_health = now
        if (
            worker.generation is None
            and not worker.dedicated
            and now - worker.started_at > self.executor.setup_timeout_s
        ):
            worker.retired = True
            if not any(
                other is not worker and other.profile == worker.profile and not other.retired
                for other in self.allocations.values()
            ):
                self._fail_profile(worker.profile, "Worker provisioning or bootstrap exceeded setup_timeout_s.")
            if worker.native is not None:
                _async_call(self.backend.cancel, worker.native)
        if worker.dedicated and worker.job_id is not None and worker.job_id not in self.graph.finished:
            if worker.execution_started_at is None:
                expired = now - worker.started_at >= self.executor.setup_timeout_s
                reason = "Dedicated allocation provisioning or setup exceeded setup_timeout_s."
            else:
                expired = now - worker.execution_started_at >= self.graph.nodes[worker.job_id].resources["time"] * 60
                reason = "Dedicated task execution deadline exceeded."
            if expired:
                self.graph.finish(worker.job_id, success=False, reason=reason)
                worker.cancel_requested = True
        if (
            worker.cancel_requested
            and worker.native is not None
            and worker.cancellation is None
            and not worker.native_done
        ):
            worker.cancellation = _async_call(self.backend.cancel, worker.native)

    def _fail_profile(self, profile: str, reason: str) -> None:
        for node in self.graph.nodes.values():
            if node.profile == profile and node.job_id not in self.graph.finished:
                self.graph.finish(node.job_id, success=False, reason=reason)

    def _assign(self, worker: _Allocation, node: GraphWork) -> None:
        attempt_id = uuid.uuid4().hex
        self.graph.assign(node.job_id, attempt_id, worker.worker_id)
        worker.job_id, worker.attempt_id = node.job_id, attempt_id
        worker.execution_started_at = None
        worker.cleanup_reported = False
        _write(
            self.workspace,
            self.manifest.run_id,
            f"attempt-{attempt_id}.assignment.json",
            job_id=node.job_id,
            attempt_id=attempt_id,
            worker_id=worker.worker_id,
            generation=worker.generation,
            profile=worker.profile,
        )
        self.persist()  # durable identity before any payload is sent
        if worker.dedicated:
            worker.launch = _async_call(self.backend.launch_dedicated, node, attempt_id)
        else:
            _write(
                self.workspace,
                self.manifest.run_id,
                f"worker-{worker.worker_id}.command.json",
                worker_id=worker.worker_id,
                generation=worker.generation,
                attempt_id=attempt_id,
                job_id=node.job_id,
                argv=node.argv,
                env=node.env,
                log_path=node.log_path,
                execution_timeout_s=float(node.resources["time"] * 60),
                setup_timeout_s=self.executor.setup_timeout_s,
            )

    def _schedule(self) -> None:
        while self.graph.ready:
            key = self.graph.ready.popleft()
            self.ready[self.graph.nodes[key].profile].append(key)
        for name, queue in self.ready.items():
            profile = self.executor.capacity[name]
            while queue:
                key = queue[0]
                if key in self.graph.finished or key in self.graph.active:
                    queue.popleft()
                    continue
                node = self.graph.nodes[key]
                workers = [
                    worker for worker in self.allocations.values() if worker.profile == name and not worker.retired
                ]
                if profile.dedicated:
                    if len(workers) >= profile.max_workers:
                        break
                    placeholder: Future[Any] = Future()
                    worker = _Allocation(
                        uuid.uuid4().hex, name, placeholder, dedicated=True, started_at=time.monotonic()
                    )
                    self.allocations[worker.worker_id] = worker
                    self._assign(worker, node)
                    queue.popleft()
                    continue
                idle = next(
                    (worker for worker in workers if worker.generation is not None and worker.job_id is None), None
                )
                if idle is not None:
                    self._assign(idle, node)
                    queue.popleft()
                    continue
                if len(workers) < profile.max_workers:
                    agent = next(
                        (
                            agent
                            for agent in self.manifest.agents
                            if agent.profile == name and agent.worker_id not in self.used_agents
                        ),
                        None,
                    )
                    if agent is not None:
                        self.used_agents.add(agent.worker_id)
                        worker = _Allocation(agent.worker_id, name, Future(), started_at=time.monotonic())
                        self.allocations[worker.worker_id] = worker
                        self._lease()
                        worker.launch = _async_call(self.backend.launch_worker, agent)
                    elif not workers:
                        self._fail_profile(name, "No live workers remain; automatic replacement is disabled.")
                break

    def step(self) -> None:
        """Advance one iteration; exposed for deterministic fake-backend tests."""
        with self.lock:
            if time.monotonic() - self.last_lease >= _LEASE_INTERVAL:
                self._lease()
            for worker in list(self.allocations.values()):
                self._observe(worker)
            cancellations = _read(self.workspace, self.manifest.run_id, "cancellations.json") or {}
            for key in set(cancellations.get("job_ids", [])) - self.processed_cancellations:
                self.processed_cancellations.add(key)
                if key in self.graph.nodes and key not in self.graph.finished:
                    self.graph.finish(key, success=False, reason="Cancelled by caller.")
                    for worker in self.allocations.values():
                        if worker.job_id == key:
                            if worker.dedicated:
                                worker.cancel_requested = True
                                if worker.native is not None and worker.cancellation is None:
                                    worker.cancellation = _async_call(self.backend.cancel, worker.native)
                            elif not worker.dedicated:
                                worker.cancel_attempt_id = worker.attempt_id
                                self.sequence += 1
                                _write(
                                    self.workspace,
                                    self.manifest.run_id,
                                    f"worker-{worker.worker_id}.lease.json",
                                    worker_id=worker.worker_id,
                                    sequence=self.sequence,
                                    stop=False,
                                    cancel_attempt_id=worker.attempt_id,
                                )
            self._schedule()
            self.persist()

    def run(self) -> None:
        """Own scheduling until completion, cancellation, or the run deadline."""
        try:
            while not self.graph.complete:
                if (
                    self.cancelled.is_set()
                    or _read(self.workspace, self.manifest.run_id, "cancel-run.json") is not None
                ):
                    for key in self.graph.nodes.keys() - self.graph.finished:
                        self.graph.finish(key, success=False, reason="Run cancelled.")
                    break
                if time.monotonic() - self.started_at >= self.executor.max_run_minutes * 60:
                    for key in self.graph.nodes.keys() - self.graph.finished:
                        self.graph.finish(key, success=False, reason="Run deadline exceeded.")
                    break
                self.step()
                self.wakeup.wait(self.executor.poll_interval_s)
                self.wakeup.clear()
        except Exception as exc:
            logger.exception("Graph coordinator %s failed", self.manifest.run_id)
            self.errors.append(f"Coordinator failed: {type(exc).__name__}.")
            with self.lock:
                for key in self.graph.nodes.keys() - self.graph.finished:
                    self.graph.states[key].state = "unknown" if key in self.graph.active else "failed"
                    self.graph.states[key].reason = "Coordinator stopped; reconcile this run before any resubmission."
        finally:
            try:
                self._cleanup()
            except Exception as exc:
                self.errors.append(f"Graph cleanup failed: {type(exc).__name__}.")
                logger.exception("Could not finish graph cleanup for %s", self.manifest.run_id)
            finally:
                self.finished.set()
                try:
                    self.persist()
                except Exception:
                    logger.exception("Could not persist final graph state for %s", self.manifest.run_id)

    def _cleanup(self) -> None:
        """Stop only this run's agents/native jobs, with a finite wait."""
        deadline = time.monotonic() + self.executor.shutdown_timeout_s
        if not self.cancelled.is_set() and self.graph.complete:
            while any(worker.job_id is not None and not worker.retired for worker in self.allocations.values()):
                if time.monotonic() >= deadline:
                    self.errors.append(
                        "Final process/log draining exceeded the shutdown grace; cancellation requested."
                    )
                    break
                if time.monotonic() - self.last_lease >= _LEASE_INTERVAL:
                    self._lease()
                for worker in list(self.allocations.values()):
                    self._observe(worker)
                self.wakeup.wait(min(self.executor.poll_interval_s, max(0, deadline - time.monotonic())))
                self.wakeup.clear()
        try:
            self._lease(stop=True)
        except Exception as exc:  # noqa: BLE001 -- best-effort cleanup records unresolved lease revocation
            self.errors.append(f"Could not revoke worker leases: {type(exc).__name__}.")
        cancellations = []
        for worker in self.allocations.values():
            # Cancellation may race the first observation of an accepted launch.
            # Reconcile that future within the same shared cleanup deadline.
            if worker.native is None and not worker.retired:
                try:
                    worker.native = worker.launch.result(timeout=max(0.0, deadline - time.monotonic()))
                except Exception as exc:  # noqa: BLE001 -- retain accepted handles even when persistence failed
                    cancellations.extend(
                        (worker.worker_id, _async_call(self.backend.cancel, native))
                        for native in getattr(exc, "submitted_jobs", ())
                    )
            if worker.native is not None and not worker.native_done:
                cancellation = worker.cancellation or _async_call(self.backend.cancel, worker.native)
                cancellations.append((worker.worker_id, cancellation))
            elif worker.native is None:
                self.errors.append(
                    f"Unresolved launch for allocation {worker.worker_id}; inspect its allocation record."
                )
        for worker_id, future in cancellations:
            try:
                future.result(timeout=max(0.0, deadline - time.monotonic()))
            except Exception as exc:  # noqa: BLE001 -- record each unresolved native cleanup target
                self.errors.append(f"Cleanup unresolved for allocation {worker_id}: {type(exc).__name__}.")

    def close(self) -> None:
        """Cancel unfinished work and wait only through the shutdown grace."""
        if not self.graph.complete:
            self.cancelled.set()
        self.wakeup.set()
        if self.thread is not None:
            self.thread.join(self.executor.shutdown_timeout_s + 1)
        if not self.finished.is_set():
            msg = f"Graph {self.manifest.run_id} cleanup exceeded its deadline; remote leases remain finite."
            raise ExecutionError(msg)
        if self.errors:
            raise ExecutionError(" ".join(self.errors))


class GraphSkyPilotExecutor(Executor[SkyPilotTaskJob]):
    """Schedule ready work over explicitly bounded SkyPilot capacity profiles.

    This replaces eager per-work-unit managed dispatch. Attached runs require
    a live session; detached runs require a stable remote SkyPilot API and an
    explicit dedicated coordinator allocation. No automatic uncertain replay.
    """

    capacity: dict[str, SkyPilotCapacity] = msgspec.field(default_factory=dict)
    lifecycle: Literal["attached", "detached"] = "attached"
    coordinator: SkyPilotCapacity | None = None
    manage_api_server: bool = True
    api_server_namespace: str = "default"
    name_prefix: str = "misen"
    max_run_minutes: int = 1440
    setup_timeout_s: float = 600.0
    shutdown_timeout_s: float = 30.0
    poll_interval_s: float = 0.2
    _config_validation_errors: ClassVar[tuple[type[Exception], ...]] = (ValueError,)

    def __post_init__(self) -> None:
        """Validate the replacement API without starting SkyPilot or allocating compute."""
        namespace_directory(self.api_server_namespace)
        self.capacity = msgspec.convert(self.capacity, type=dict[str, SkyPilotCapacity])
        self.coordinator = msgspec.convert(self.coordinator, type=SkyPilotCapacity | None)
        if any(not _TOKEN.fullmatch(name) for name in self.capacity):
            msg = "Capacity names must be 1-64 letters, digits, underscores, or hyphens."
            raise ValueError(msg)
        if self.lifecycle not in ("attached", "detached"):
            msg = "lifecycle must be attached or detached."
            raise ValueError(msg)
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,19}", self.name_prefix):
            msg = "name_prefix must contain 1-20 lowercase letters, digits, or hyphens."
            raise ValueError(msg)
        if not self.snapshot or self.prewarm_envs:
            msg = "SkyPilot requires snapshot=True and prewarm_envs=False."
            raise ValueError(msg)
        if (
            isinstance(self.max_run_minutes, bool)
            or not isinstance(self.max_run_minutes, int)
            or self.max_run_minutes <= 0
        ):
            msg = "max_run_minutes must be a positive integer."
            raise ValueError(msg)
        for name in ("setup_timeout_s", "shutdown_timeout_s", "poll_interval_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                msg = f"{name} must be finite and positive."
                raise ValueError(msg)
        if self.lifecycle == "detached" and (self.manage_api_server or self.coordinator is None):
            msg = (
                "Detached runs require manage_api_server=False, a stable remote SkyPilot API, and coordinator capacity."
            )
            raise ValueError(msg)
        if self.coordinator is not None and (
            self.lifecycle != "detached"
            or not self.coordinator.dedicated
            or self.coordinator.nodes != 1
            or self.coordinator.borrowed
        ):
            msg = "Coordinator capacity is a detached-only, dedicated, run-owned single-node reservation."
            raise ValueError(msg)

    @contextlib.contextmanager
    def session(self) -> Iterator[Any]:
        """Keep the graph coordinator and isolated API lease alive together."""
        existing = _runs.get()
        if existing is not None and existing[0] == id(self):
            yield active_session()
            return
        owned: list[GraphCoordinator] = []
        api = managed_session(self.api_server_namespace) if self.manage_api_server else contextlib.nullcontext()
        with api as session:
            token = _runs.set((id(self), owned))
            original: BaseException | None = None
            try:
                yield session
            except BaseException as exc:
                original = exc
                raise
            finally:
                try:
                    errors: list[str] = []
                    for run in owned:
                        try:
                            run.close()
                        except ExecutionError as exc:
                            errors.append(str(exc))
                    if errors:
                        if original is None:
                            raise ExecutionError(" ".join(errors))
                        for error in errors:
                            original.add_note(error)
                finally:
                    _runs.reset(token)

    def submit(
        self, tasks: set[Task], workspace: Workspace, *, blocking: bool = False
    ) -> DependencyGraph[CompletedJob | SkyPilotTaskJob]:
        """Submit a graph; blocking calls automatically scope an attached run."""
        if blocking:
            with self.session():
                return super().submit(tasks, workspace, blocking=True)
        return super().submit(tasks, workspace, blocking=False)

    def _run_defaults_to_blocking(self) -> bool:
        return self.lifecycle == "attached"

    def attach(self, run_id: str, workspace: Workspace) -> DependencyGraph[SkyPilotTaskJob]:
        """Reconstruct handles for a trusted durable run without resubmitting it.

        This observes/cancels an existing run; it never takes over a lost
        coordinator or replays uncertain execution. The workspace contains
        executable pickle payloads and must belong to a trusted submission.

        Raises:
            ValueError: If the run identity is invalid.
            StorageError: If the manifest or work-unit identities do not match.
        """
        from misen.utils.graph import DependencyGraph

        if not _TOKEN.fullmatch(run_id):
            msg = "Invalid graph run identity."
            raise ValueError(msg)
        from misen.utils.work_unit import WorkUnit

        try:
            manifest = msgspec.json.decode(workspace.read_job_file(run_id, "run-manifest.json"), type=RunManifest)
            ReadyGraph(manifest.nodes)
        except (msgspec.DecodeError, ValueError) as exc:
            msg = "Run manifest is malformed."
            raise StorageError(msg) from exc
        if manifest.run_id != run_id or manifest.version != 1:
            msg = "Run manifest has a different identity."
            raise StorageError(msg)
        units = cloudpickle.loads(workspace.read_job_file(run_id, "run-work-units.pkl"))
        if (
            not isinstance(units, dict)
            or set(units) != {node.job_id for node in manifest.nodes}
            or not all(isinstance(unit, WorkUnit) for unit in units.values())
        ):
            msg = "Run work-unit identities do not match the manifest."
            raise StorageError(msg)
        graph: DependencyGraph[SkyPilotTaskJob] = DependencyGraph()
        indices = {}
        owner = _runs.get()
        active = next((run for run in owner[1] if run.manifest.run_id == run_id), None) if owner is not None else None
        for node in manifest.nodes:
            job = SkyPilotTaskJob(units[node.job_id], node.job_id, Path(node.log_path), workspace, run_id)
            job.coordinator = active
            job.stale_timeout_s = self.setup_timeout_s
            indices[node.job_id] = graph.add_node(job)
        for node in manifest.nodes:
            for parent in node.dependencies:
                graph.add_edge(indices[node.job_id], indices[parent])
        return graph

    def _profile(self, work_unit: WorkUnit) -> str:
        if work_unit.resources["accelerators"] and work_unit.resources["accelerator_type"] not in (
            "cuda",
            "rocm",
            "xpu",
        ):
            msg = "SkyPilot graph execution supports accelerator visibility isolation for cuda, rocm, and xpu only."
            raise ConfigError(msg)
        matches = [
            name
            for name, profile in self.capacity.items()
            if profile.fits(work_unit.resources)
            and (not work_unit.uses_dask_client or profile.nodes == work_unit.resources["nodes"])
        ]
        if not matches:
            msg = f"No configured SkyPilot capacity fits {work_unit.resources}; add an explicit bounded profile."
            raise ConfigError(msg)
        return min(
            matches,
            key=lambda name: (
                self.capacity[name].accelerator_count > work_unit.resources["accelerators"],
                self.capacity[name].dedicated,
                self.capacity[name].nodes,
                self.capacity[name].memory,
                self.capacity[name].cpus,
                name,
            ),
        )

    def _validate_submission(
        self, *, work_graph: DependencyGraph[WorkUnit], pending_work_units: Sequence[WorkUnit], workspace: Workspace
    ) -> None:
        del work_graph
        if not workspace.supports_job_file_reads() or workspace.bootstrap_transport() is None:
            msg = "SkyPilot graph runs require a remotely fetchable workspace with job-file coordination."
            raise ConfigError(msg)
        if workspace.get_temp_dir().is_absolute():
            msg = "SkyPilot requires a relative workspace cache_dir."
            raise ConfigError(msg)
        owner = _runs.get()
        if self.lifecycle == "attached" and (owner is None or owner[0] != id(self)):
            msg = "Nonblocking attached submissions require `with executor.session():`; use blocking=True otherwise."
            raise ConfigError(msg)
        for work_unit in pending_work_units:
            self._profile(work_unit)
        if self.lifecycle == "detached":
            sky = _load_skypilot()
            if sky.server.common.is_api_server_local():
                msg = "Detached graph execution requires an explicitly configured stable remote SkyPilot API."
                raise ConfigError(msg)
            try:
                health = sky.api_info()
            except Exception as exc:
                msg = "Could not verify remote SkyPilot API access for a detached coordinator."
                raise ConfigError(msg) from exc
            status = _field(health, "status")
            if getattr(status, "value", status) != "healthy":
                msg = "Detached graph execution requires an authenticated healthy remote SkyPilot API."
                raise ConfigError(msg)
            if _field(health, "service_account_token_enabled") is not True:
                msg = (
                    "Detached graph execution requires service accounts enabled on the remote SkyPilot API "
                    "(service_account_token_enabled=True); otherwise coordinator credentials are not injected."
                )
                raise ConfigError(msg)
            api_version = _field(health, "api_version")
            # The SDK skips api_server_access endpoint injection below API 42.
            minimum_api_access_version = 42
            if (
                isinstance(api_version, bool)
                or not isinstance(api_version, (str, int))
                or not str(api_version).isascii()
                or not str(api_version).isdecimal()
                or int(api_version) < minimum_api_access_version
            ):
                msg = "Detached coordinator credential injection requires remote SkyPilot API version >=42."
                raise ConfigError(msg)

    def _dispatch(
        self,
        work_unit: WorkUnit,
        dependencies: set[SkyPilotTaskJob],
        workspace: Workspace,
        snapshot: ProjectSnapshot | None,
    ) -> SkyPilotTaskJob:
        del work_unit, dependencies, workspace, snapshot
        msg = "SkyPilot dispatches graphs through its coordinator, not individual eager jobs."
        raise RuntimeError(msg)

    def _dispatch_work_graph(
        self,
        *,
        pending_work_units: Sequence[WorkUnit],
        jobs: dict[WorkUnit, CompletedJob | SkyPilotTaskJob],
        workspace: Workspace,
        snapshot: ProjectSnapshot | None,
        progress: Callable[[int], None],
    ) -> None:
        if snapshot is None:
            msg = "SkyPilot requires a published snapshot."
            raise SubmissionError(msg)
        prepared = {unit: snapshot.prepare_job(unit, workspace) for unit in pending_work_units}
        nodes = []
        for unit, (job_id, argv, env, log_path) in prepared.items():
            task_env = env | resource_environment(cpu_indices=list(range(unit.resources["cpus"])))
            # The agent intersects the inherited SkyPilot GPU reservation; never
            # guess physical device IDs on borrowed/shared machines.
            task_env["MISEN_ACCELERATOR_COUNT"] = str(unit.resources["accelerators"])
            task_env["MISEN_ACCELERATOR_TYPE"] = unit.resources["accelerator_type"]
            node = GraphWork(
                job_id,
                [prepared[parent][0] for parent in unit.dependencies if parent in prepared],
                self._profile(unit),
                argv,
                task_env,
                str(log_path),
                unit.resources,
                unit.uses_dask_client,
            )
            nodes.append(node)
            jobs[unit] = SkyPilotTaskJob(unit, job_id, log_path, workspace, snapshot.submission_id)
            cast("SkyPilotTaskJob", jobs[unit]).stale_timeout_s = self.setup_timeout_s
            progress(1)
        agents = []
        for name in {node.profile for node in nodes}:
            profile = self.capacity[name]
            if profile.dedicated:
                continue
            for _ in range(min(profile.max_workers, sum(node.profile == name for node in nodes))):
                worker_id = uuid.uuid4().hex
                fn = functools.partial(
                    run_worker_agent,
                    workspace,
                    snapshot.submission_id,
                    worker_id,
                    max_runtime_s=self.max_run_minutes * 60,
                    poll_interval_s=self.poll_interval_s,
                )
                agent_id, argv, env, log_path = _prepare_control(snapshot, workspace, fn)
                agents.append(AgentWork(worker_id, name, agent_id, argv, env, str(log_path)))
        manifest = RunManifest(snapshot.submission_id, snapshot.snapshot_key, nodes, agents)
        workspace.put_job_file(
            manifest.run_id,
            "run-work-units.pkl",
            cloudpickle.dumps({prepared[unit][0]: unit for unit in pending_work_units}),
        )
        workspace.put_job_file(manifest.run_id, "run-manifest.json", msgspec.json.encode(manifest))
        backend = _SkyCapacityBackend(self, manifest, workspace)
        run = GraphCoordinator(self, manifest, workspace, backend)
        run.persist()
        if self.lifecycle == "detached":
            fn = functools.partial(_run_remote, self, manifest, workspace)
            job_id, argv, env, log_path = _prepare_control(snapshot, workspace, fn)
            backend.launch_coordinator(job_id, argv, env, log_path)
        else:
            owner = _runs.get()
            if owner is None:
                msg = "Attached coordinator has no owning session."
                raise SubmissionError(msg)
            owner[1].append(run)
            for unit in pending_work_units:
                cast("SkyPilotTaskJob", jobs[unit]).coordinator = run
            run.start()


def _prepare_control(
    snapshot: ProjectSnapshot, workspace: Workspace, function: Callable[[], None]
) -> tuple[str, list[str], dict[str, str], Path]:
    """Use the normal trusted snapshot bootstrap for agent/coordinator payloads."""
    from misen.tasks import Task
    from misen.utils.work_unit import WorkUnit

    work_unit = WorkUnit(Task(_control_placeholder), set())
    job_id, argv, env, log_path = snapshot.prepare_job(work_unit, workspace)
    workspace.put_job_file(
        snapshot.submission_id, f"{job_id}.pkl", cloudpickle.dumps({"workspace": workspace, "fn": function})
    )
    return job_id, argv, env, log_path


@meta(id="misen-skypilot-control-v1")
def _control_placeholder() -> None:
    """Supply a stable internal work-unit identity; payload is replaced before launch."""


def _run_remote(executor: GraphSkyPilotExecutor, manifest: RunManifest, workspace: Workspace) -> None:
    import os

    if executor.lifecycle != "detached" or executor.manage_api_server:
        msg = "A remote coordinator requires detached lifecycle and an externally managed SkyPilot API."
        raise ExecutionError(msg)
    if (
        not os.environ.get("SKYPILOT_API_SERVER_ENDPOINT", "").strip()
        or not os.environ.get("SKYPILOT_SERVICE_ACCOUNT_TOKEN", "").strip()
    ):
        msg = "Remote coordinator is missing its injected SkyPilot endpoint or service-account token."
        raise ExecutionError(msg)
    sky = _load_skypilot()
    if sky.server.common.is_api_server_local():
        msg = "Remote coordinator refuses a local SkyPilot API endpoint; no local service will be started."
        raise ExecutionError(msg)
    try:
        health = sky.api_info()
    except Exception as exc:
        msg = "Remote coordinator could not authenticate to its injected SkyPilot API."
        raise ExecutionError(msg) from exc
    status = _field(health, "status")
    if getattr(status, "value", status) != "healthy":
        msg = "Remote coordinator requires an authenticated healthy remote SkyPilot API."
        raise ExecutionError(msg)
    with workspace.lock("job", f"coordinator-{manifest.run_id}").context(timeout=30):
        if _read(workspace, manifest.run_id, "coordinator-owner.json") is not None:
            msg = "Coordinator has already run; automatic takeover of uncertain work is disabled."
            raise ExecutionError(msg)
        _write(workspace, manifest.run_id, "coordinator-owner.json", epoch=uuid.uuid4().hex)
    run = GraphCoordinator(executor, manifest, workspace, _SkyCapacityBackend(executor, manifest, workspace))
    run.run()
    if run.errors:
        raise ExecutionError(" ".join(run.errors))


class _SkyCapacityBackend:
    """Native allocation operations; never invoked once per reusable work unit."""

    def __init__(self, executor: GraphSkyPilotExecutor, manifest: RunManifest, workspace: Workspace) -> None:

        self.executor = executor
        self.manifest = manifest
        self.workspace = workspace
        self.sky = _load_skypilot()
        self._native_records: dict[int, tuple[str, str]] = {}

    def _native(self, profile: SkyPilotCapacity, record: dict[str, Any], log_path: Path) -> Any:
        """Restore an accepted request without issuing another cloud submission."""
        from misen.tasks import Task
        from misen.utils.work_unit import WorkUnit

        request_id = record.get("request_id")
        native_id = record.get("native_job_id")
        if not isinstance(request_id, str) or not request_id:
            msg = "Allocation acceptance is uncertain; inspect its durable record before resubmitting."
            raise SubmissionError(msg)
        if native_id is not None and (isinstance(native_id, bool) or not isinstance(native_id, int) or native_id < 1):
            msg = "Allocation record contains an invalid native job identity."
            raise StorageError(msg)
        native: Any
        if profile.cluster:
            native = _ClusterAllocation(
                self.sky, profile.cluster, request_id, name=record["name"], job_id=native_id, log_path=log_path
            )
        else:
            native = SkyPilotJob(
                work_unit=WorkUnit(Task(_control_placeholder), set()),
                job_id=record["job_id"],
                managed_job_id=native_id,
                submission_id=self.manifest.run_id,
                deadline_minutes=record["time_minutes"],
                log_path=log_path,
                workspace=self.workspace,
                request_id=request_id,
                managed_job_name=record["name"],
            )
            session = native._api_session  # noqa: SLF001 -- graph owns allocation draining and reconciliation
            if session is not None:
                session.jobs.remove(native)
        self._native_records[id(native)] = (record["allocation_id"], request_id)
        return native

    def _remember_native(self, native: Any) -> None:
        """Persist resolved identities without overwriting a replaced allocation."""
        allocation_id, request_id = self._native_records[id(native)]
        native_id = native.job_id if isinstance(native, _ClusterAllocation) else native.managed_job_id
        if native_id is None:
            return
        if isinstance(native_id, bool) or not isinstance(native_id, int) or native_id < 1:
            msg = "SkyPilot returned an invalid native allocation identity."
            raise StorageError(msg)
        with self.workspace.lock("job", self._lock_key(allocation_id)).context(timeout=30):
            record = _read(self.workspace, self.manifest.run_id, f"allocation-{allocation_id}.json")
            name = native.name if isinstance(native, _ClusterAllocation) else native.managed_job_name
            if (
                record is None
                or record.get("allocation_id") != allocation_id
                or record.get("request_id") not in (None, request_id)
                or record.get("name") != name
            ):
                msg = "Allocation record disappeared or changed while resolving its native identity."
                raise StorageError(msg)
            if record.get("native_job_id") == native_id:
                return
            if record.get("native_job_id") is not None:
                msg = "Allocation record already contains a different native identity."
                raise StorageError(msg)
            record["native_job_id"] = native_id
            record["request_id"] = request_id
            record["launch_state"] = "accepted"
            self.workspace.put_job_file(
                self.manifest.run_id, f"allocation-{allocation_id}.json", msgspec.json.encode(record)
            )

    def _lock_key(self, allocation_id: str) -> str:
        """Bound the lock filename even when run and allocation IDs are long."""
        import hashlib

        identity = msgspec.json.encode((self.manifest.run_id, allocation_id))
        return "allocation-" + hashlib.sha256(identity).hexdigest()

    def _launch(
        self,
        profile: SkyPilotCapacity,
        allocation_id: str,
        job_id: str,
        argv: list[str],
        env: dict[str, str],
        *,
        log_path: Path,
        api_access: bool = False,
        uses_dask_client: bool = False,
        time_minutes: int | None = None,
    ) -> Any:
        options = profile.as_sky_options()
        alternatives = options.pop("infra", None)
        alternatives = alternatives if isinstance(alternatives, list) else [alternatives]
        resources = [self.sky.Resources(**options, **({"infra": infra} if infra else {})) for infra in alternatives]
        # Hash the full identity: truncating 'coordinator-<run>' discarded most
        # of the run ID and could collide with unrelated accepted jobs.
        name = f"{self.executor.name_prefix}-{self._lock_key(allocation_id)[-24:]}"
        deadline = self.executor.max_run_minutes if time_minutes is None else time_minutes
        command = _run_command(
            argv, env, log_path, time_minutes=deadline, profile=profile, uses_dask_client=uses_dask_client
        )
        task = self.sky.Task(
            name=name, run=command, num_nodes=profile.nodes, resources=resources, api_server_access=api_access
        )
        identity = {
            "allocation_id": allocation_id,
            "job_id": job_id,
            "name": name,
            "source": "cluster" if profile.cluster else "pool" if profile.pool else "owned",
            "cluster": profile.cluster,
            "pool": profile.pool,
            "profile": msgspec.to_builtins(profile),
            "time_minutes": deadline,
        }
        record_name = f"allocation-{allocation_id}.json"
        with self.workspace.lock("job", self._lock_key(allocation_id)).context(timeout=30):
            previous = _read(self.workspace, self.manifest.run_id, record_name)
            if previous is not None:
                if any(previous.get(key) != value for key, value in identity.items()):
                    msg = "Allocation identity was reused with different submission parameters."
                    raise SubmissionError(msg)
                return self._native(profile, previous, log_path)
            record = dict(identity, request_id=None, native_job_id=None, launch_state="submitting")
            _write(self.workspace, self.manifest.run_id, record_name, **record)
            try:
                request_id = (
                    self.sky.exec(task, cluster_name=profile.cluster)
                    if profile.cluster
                    else self.sky.jobs.launch(task, name=name, pool=profile.pool)
                )
            except Exception as exc:
                msg = "SkyPilot allocation submission failed or is uncertain; inspect the durable allocation record."
                raise SubmissionError(msg) from exc
            if not isinstance(request_id, str) or not request_id:
                msg = "SkyPilot returned no valid allocation request identity; submission may have been accepted."
                raise SubmissionError(msg)
            record.update(request_id=str(request_id), launch_state="accepted")
            native = self._native(profile, record, log_path)
            try:
                _write(self.workspace, self.manifest.run_id, record_name, **record)
            except Exception as exc:
                msg = "SkyPilot accepted the allocation, but its durable request record could not be persisted."
                raise SubmissionError(msg, submitted_jobs=(native,)) from exc
            return native

    def launch_worker(self, agent: AgentWork) -> Any:
        return self._launch(
            self.executor.capacity[agent.profile],
            agent.worker_id,
            agent.job_id,
            agent.argv,
            agent.env,
            log_path=Path(agent.log_path),
        )

    def launch_dedicated(self, node: GraphWork, attempt_id: str) -> Any:
        env = node.env | {"MISEN_RUN_ID": self.manifest.run_id, "MISEN_ATTEMPT_ID": attempt_id}
        return self._launch(
            self.executor.capacity[node.profile],
            attempt_id,
            node.job_id,
            node.argv,
            env,
            log_path=Path(node.log_path),
            uses_dask_client=node.uses_dask_client,
            time_minutes=node.resources["time"] + math.ceil(self.executor.setup_timeout_s / 60),
        )

    def launch_coordinator(self, job_id: str, argv: list[str], env: dict[str, str], log_path: Path) -> Any:
        profile = self.executor.coordinator
        if profile is None:
            msg = "Detached coordinator capacity is required."
            raise ConfigError(msg)
        if profile.borrowed or not profile.dedicated or profile.nodes != 1:
            msg = "Detached coordinators require dedicated run-owned single-node capacity."
            raise ConfigError(msg)
        if self.sky.server.common.is_api_server_local():
            msg = "Detached coordinators require an explicitly configured stable remote SkyPilot API."
            raise ConfigError(msg)
        native = self._launch(
            profile, "coordinator-" + self.manifest.run_id, job_id, argv, env, log_path=log_path, api_access=True
        )
        # Durable remote acknowledgement is necessary before the submitter exits.
        try:
            native_id = native._resolve_managed_job_id(self.sky)  # noqa: SLF001 -- low-level allocation handle
            if isinstance(native_id, bool) or not isinstance(native_id, int) or native_id < 1:
                msg = "SkyPilot did not acknowledge a managed coordinator job."
                raise ExecutionError(msg)  # noqa: TRY301 -- preserve accepted native handle on every acknowledgement failure
            self._remember_native(native)
        except Exception as exc:
            msg = (
                "Could not durably acknowledge the remote coordinator; inspect the accepted allocation before retrying."
            )
            raise SubmissionError(msg, submitted_jobs=(native,)) from exc
        return native

    @contextlib.contextmanager
    def _persist_after_operation(self, native: Any) -> Iterator[None]:
        """Retain resolved IDs even when status or cancellation subsequently fails."""
        try:
            yield
        except BaseException as exc:
            try:
                self._remember_native(native)
            except Exception:  # noqa: BLE001 -- preserve the original SDK failure and traceback
                exc.add_note("Additionally, the resolved allocation identity could not be persisted.")
            raise
        else:
            self._remember_native(native)

    def state(self, native: Any) -> JobState:
        with self._persist_after_operation(native):
            return native.state()

    def cancel(self, native: Any) -> None:
        with self._persist_after_operation(native):
            native.cancel()


class _ClusterAllocation:
    """A reserved cluster job: cleanup never stops or downs the cluster."""

    def __init__(
        self,
        sky: Any,
        cluster: str,
        request_id: str,
        *,
        name: str,
        job_id: int | None = None,
        log_path: Path | None = None,
    ) -> None:
        self.sky, self.cluster, self.request_id = sky, cluster, request_id
        self.job_id = job_id
        self.name = name
        self.log_path = log_path
        self.label = f"SkyPilot allocation {name} on cluster {cluster}"

    def _recover(self) -> int:
        """Recover exactly one named cluster job after request metadata expires."""
        records = self.sky.get(self.sky.queue(cluster_name=self.cluster, skip_finished=False, all_users=False))
        if not isinstance(records, list):
            msg = "SkyPilot returned an invalid cluster queue response."
            raise StatusQueryError(msg)
        matching = [record for record in records if _field(record, "job_name") == self.name]
        if len(matching) != 1:
            msg = "The cluster allocation request is unresolved and exact-name recovery is absent or ambiguous."
            raise StatusQueryError(msg)
        job_id = _field(matching[0], "job_id")
        if isinstance(job_id, bool) or not isinstance(job_id, int) or job_id < 1:
            msg = "SkyPilot returned an invalid recovered cluster job identity."
            raise StatusQueryError(msg, retryable=False)
        return job_id

    def _resolve(self) -> None:
        if self.job_id is None:
            try:
                result = self.sky.get(self.request_id)
            except Exception:  # noqa: BLE001 -- expired/failed request lookup requires exact-name reconciliation
                try:
                    self.job_id = self._recover()
                except Exception as recovery_exc:
                    msg = "Could not recover the cluster allocation's native identity; no broad cancellation is safe."
                    raise StatusQueryError(msg) from recovery_exc
                return
            job_id = result[0] if isinstance(result, tuple) and len(result) == 2 else None  # noqa: PLR2004
            if isinstance(job_id, bool) or not isinstance(job_id, int) or job_id < 1:
                msg = "SkyPilot returned an invalid cluster job identity."
                raise ExecutionError(msg)
            self.job_id = job_id

    def state(self) -> JobState:

        self._resolve()
        states = self.sky.get(self.sky.job_status(cluster_name=self.cluster, job_ids=[self.job_id]))
        if not isinstance(states, dict):
            msg = "SkyPilot returned an invalid cluster status response."
            raise StatusQueryError(msg)
        # The isolated API broker serializes JSON object keys as strings.
        return _normalize_skypilot_state(states.get(self.job_id, states.get(str(self.job_id))))

    def cancel(self) -> None:
        self._resolve()
        self.sky.get(self.sky.cancel(cluster_name=self.cluster, job_ids=[self.job_id]))


def _run_command(
    argv: list[str],
    env: dict[str, str],
    log_path: Path,
    *,
    time_minutes: int,
    profile: SkyPilotCapacity,
    uses_dask_client: bool,
) -> str:
    from misen.utils.dask_runtime import managed_ranked_cluster_script

    if isinstance(time_minutes, bool) or not isinstance(time_minutes, int) or time_minutes < 1:
        msg = "Allocation timeout must be a positive integer number of minutes."
        raise ValueError(msg)
    command = (
        managed_ranked_cluster_script(
            argv,
            environment=env,
            workers=profile.nodes,
            cpus=profile.cpus,
            memory_gib=profile.memory,
            startup_timeout=300,
            node_rank_env="SKYPILOT_NODE_RANK",
            node_ips_env="SKYPILOT_NODE_IPS",
            scheduler_port=8786,
        )
        if uses_dask_client
        else shlex.join(["env", *(f"{key}={value}" for key, value in env.items()), *argv])
    )
    lines = ["set -euo pipefail"]
    if profile.nodes > 1 and not uses_dask_client:
        lines.append('if [[ "${SKYPILOT_NODE_RANK:-0}" != "0" ]]; then exit 0; fi')
    lines.extend(
        (
            f"mkdir -p {shlex.quote(str(log_path.parent))}",
            (
                f"timeout --signal=TERM --kill-after=30s {time_minutes}m bash -c {shlex.quote(command)} "
                f"2>&1 | tee -a {shlex.quote(str(log_path))}"
            ),
        )
    )
    return "\n".join(lines)


class SkyPilotExecutor(GraphSkyPilotExecutor):
    """Ready-only graph execution over reusable and dedicated SkyPilot capacity."""


def _main() -> None:
    """Dispatch explicit child-only roles without loading SkyPilot for guards."""
    if len(sys.argv) > 1 and sys.argv[1] in ("--broker", "--server"):
        role = sys.argv[1]
        os.environ["_MISEN_SKYPILOT_PROCESS_ROLE"] = role
        if role == "--broker":
            del sys.argv[1]
            parser = argparse.ArgumentParser(description="Start an isolated SkyPilot broker.")
            parser.add_argument("directory")
            parser.add_argument("log_path")
            parser.parse_args()
        _broker_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "--worker-guard":
        del sys.argv[1]
        os.environ.pop("_MISEN_SKYPILOT_PROCESS_ROLE", None)
        _guard_main()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("role", choices=("--broker", "--server", "--worker-guard"))
        parser.parse_args()
        parser.error("an explicit child-process role is required")


if __name__ == "__main__":
    _main()
elif __name__ == "__mp_main__" and os.environ.get("_MISEN_SKYPILOT_PROCESS_ROLE") in ("--broker", "--server"):
    # Spawned SDK workers need the same compatibility paths as their parent.
    # Ordinary imports and worker guards must not load the SDK.
    _load_isolated_sdk()
