"""Graph coordinator, executor implementation, and capacity backend."""

from __future__ import annotations

import contextlib
import contextvars
import functools
import hashlib
import logging
import math
import os
import re
import shlex
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import cloudpickle
import msgspec

from misen.exceptions import (
    ConfigError,
    ExecutionError,
    StatusQueryError,
    StorageError,
    SubmissionError,
)
from misen.executor import CompletedJob, Executor, Job, JobState
from misen.task_metadata import meta
from misen.utils.resource_env import resource_environment

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from misen.tasks import Task
    from misen.utils.graph import DependencyGraph
    from misen.utils.snapshot import ProjectSnapshot
    from misen.utils.work_unit import WorkUnit
    from misen.workspace import Workspace

from .api import _active_session, managed_session, namespace_directory
from .jobs import SkyPilotJob, _field, _load_external_skypilot, _normalize_skypilot_state
from .models import (
    AgentWork,
    GraphWork,
    ReadyGraph,
    RunManifest,
    RunState,
    SkyPilotCapacity,
    profile_dependency_widths,
    read_run_state,
)
from .worker import run_worker_agent

logger = logging.getLogger("misen.executors.skypilot")

# Graph coordinator and executor
# ----------------------------------------------------------------------------

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


_runs: contextvars.ContextVar[_ExecutionSession | None] = contextvars.ContextVar("misen_graph_runs", default=None)


@contextlib.contextmanager
def _bound_active_session(session: Any) -> Iterator[Any]:
    """Temporarily bind one exact managed session, including explicit external mode."""
    token = _active_session.set(session)
    try:
        yield session
    finally:
        _active_session.reset(token)


_HEALTH_INTERVAL = 10.0


_LEASE_INTERVAL = 10.0


_AGENT_LEASE_TIMEOUT_S = 60.0


_CONTROL_RECORD_LIMIT = 2_000_000


def _agent_lifetime_s(executor: GraphSkyPilotExecutor) -> float:
    """Bound one reusable allocation beyond the graph deadline and cleanup grace."""
    return (
        executor.max_run_minutes * 60 + executor.setup_timeout_s + executor.shutdown_timeout_s + _AGENT_LEASE_TIMEOUT_S
    )


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
    backend: Any = None
    workspace: Workspace | None = None
    control_id: str | None = None
    runtime_key: str | None = None
    profile_key: str | None = None
    owner_run_id: str | None = None
    lease_sequence: int = 0
    last_lease: float = 0
    launch_reconciled: bool = False
    launch_error: BaseException | None = field(default=None, repr=False)
    late_cancel_registered: bool = False
    cleanup_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    cleanup_cancellations: dict[int, Future[Any]] = field(default_factory=dict, repr=False)
    lease_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    lease_stopped: bool = False
    maintenance: bool = False
    expires_at: float | None = None


def _resolve_completed_launch(worker: _Allocation) -> tuple[bool, bool, BaseException | None]:
    """Adopt one completed launch exactly once.

    Returns ``(complete, newly_reconciled, error)``.  Keeping reconciliation
    separate from scheduling lets teardown retain ownership after an allocation
    has otherwise been retired.
    """
    with worker.cleanup_lock:
        if worker.launch_reconciled:
            return True, False, worker.launch_error
        if not worker.launch.done():
            return False, False, None
        try:
            worker.native = worker.launch.result()
        except BaseException as exc:  # noqa: BLE001 -- launch failures are reconciled by their owner
            worker.launch_error = exc
        worker.launch_reconciled = True
        return True, True, worker.launch_error


def _request_handle_cancellation(worker: _Allocation, backend: Any, native: Any) -> Future[Any]:
    """Start and retain exactly one cancellation future per accepted handle."""
    target = id(native)
    with worker.cleanup_lock:
        existing = worker.cleanup_cancellations.get(target)
        if existing is not None:
            return existing
        future = _async_call(backend.cancel, native)
        worker.cleanup_cancellations[target] = future
        if native is worker.native:
            worker.cancellation = future
        return future


def _request_submitted_cancellations(
    worker: _Allocation, backend: Any, error: BaseException | None
) -> list[Future[Any]]:
    """Retain cleanup for handles attached to a post-acceptance failure."""
    if error is None:
        return []
    return [_request_handle_cancellation(worker, backend, native) for native in getattr(error, "submitted_jobs", ())]


def _register_late_launch_cancellation(worker: _Allocation, backend: Any) -> None:
    """Cancel a launch that becomes accepted after the cleanup deadline."""
    with worker.cleanup_lock:
        if worker.late_cancel_registered:
            return
        worker.late_cancel_registered = True

    def cancel_when_resolved(_future: Future[Any]) -> None:
        try:
            _complete, _new, error = _resolve_completed_launch(worker)
            _request_submitted_cancellations(worker, backend, error)
            if worker.native is not None and not worker.native_done:
                cancellation = _request_handle_cancellation(worker, backend, worker.native)

                def report_failure(completed: Future[Any]) -> None:
                    try:
                        completed.result()
                    except BaseException:  # no caller remains to receive a late failure
                        logger.exception("Late allocation cancellation failed for %s", worker.worker_id)

                cancellation.add_done_callback(report_failure)
        except BaseException:  # callbacks must not escape into Future internals
            logger.exception("Could not reconcile late allocation launch for %s", worker.worker_id)

    worker.launch.add_done_callback(cancel_when_resolved)


def _prepare_launch_cleanup(worker: _Allocation, backend: Any, deadline: float) -> bool:
    """Reconcile and cancel a launch, waiting only until the shared deadline.

    ``False`` means acceptance is still unresolved.  A callback then owns any
    later accepted handle, while the caller reports the unresolved launch.
    """
    complete, _new, error = _resolve_completed_launch(worker)
    if not complete:
        completed = threading.Event()
        worker.launch.add_done_callback(lambda _future: completed.set())
        completed.wait(max(0.0, deadline - time.monotonic()))
        complete, _new, error = _resolve_completed_launch(worker)
    if not complete:
        _register_late_launch_cancellation(worker, backend)
        return False
    _request_submitted_cancellations(worker, backend, error)
    if worker.native is not None and not worker.native_done:
        _request_handle_cancellation(worker, backend, worker.native)
    return True


def _await_tracked_cancellations(
    workers: Sequence[_Allocation], deadline: float, errors: list[str], *, message_prefix: str
) -> None:
    """Await every cancellation known before the shared deadline expires."""
    seen: set[int] = set()
    while True:
        pending: list[tuple[str, Future[Any]]] = []
        for worker in workers:
            with worker.cleanup_lock:
                pending.extend(
                    (worker.worker_id, future)
                    for future in worker.cleanup_cancellations.values()
                    if id(future) not in seen
                )
        if not pending:
            return
        for worker_id, future in pending:
            seen.add(id(future))
            try:
                future.result(timeout=max(0.0, deadline - time.monotonic()))
            except BaseException as exc:  # noqa: BLE001 -- continue draining every exact cleanup target
                errors.append(f"{message_prefix} {worker_id}: {type(exc).__name__}.")


@dataclass(frozen=True)
class _FleetKey:
    """Exact compatibility identity for one reusable worker environment."""

    workspace_id: int
    runtime_key: str
    profile_name: str
    profile_key: str


class _AgentFleet:
    """Own warm allocations for exactly one explicit executor session.

    Coordinators borrow an allocation only while they have ready work. Idle
    workers retain a finite session lease and can be reused by a later graph;
    closing the session revokes every lease and cancels every native job.
    """

    def __init__(self, executor: GraphSkyPilotExecutor) -> None:
        self.executor = executor
        self.lock = threading.RLock()
        self.allocations: dict[str, _Allocation] = {}
        self.errors: list[str] = []
        self.stopped = threading.Event()
        self.wakeup = threading.Event()
        self.thread: threading.Thread | None = None

    def key(self, workspace: Workspace, runtime_key: str, profile_name: str, profile: SkyPilotCapacity) -> _FleetKey:
        profile_data = msgspec.json.encode(
            (
                "misen-agent-v2",
                msgspec.to_builtins(profile),
                self.executor.max_run_minutes,
                self.executor.setup_timeout_s,
                self.executor.shutdown_timeout_s,
                self.executor.poll_interval_s,
                _AGENT_LEASE_TIMEOUT_S,
            )
        )
        return _FleetKey(id(workspace), runtime_key, profile_name, hashlib.sha256(profile_data).hexdigest())

    def _start(self) -> None:
        if self.thread is not None or self.stopped.is_set():
            return
        self.thread = threading.Thread(
            target=contextvars.copy_context().run,
            args=(self._run,),
            name="misen-skypilot-agent-fleet",
            daemon=True,
        )
        self.thread.start()

    @staticmethod
    def _matches(worker: _Allocation, key: _FleetKey) -> bool:
        return (
            worker.workspace is not None
            and id(worker.workspace) == key.workspace_id
            and worker.runtime_key == key.runtime_key
            and worker.profile == key.profile_name
            and worker.profile_key == key.profile_key
        )

    @staticmethod
    def _retirement_complete(worker: _Allocation) -> bool:
        """Whether retired native capacity is safe to stop charging to its limit."""
        if not worker.retired or not worker.launch_reconciled:
            return False
        with worker.cleanup_lock:
            futures = list(worker.cleanup_cancellations.values())
            cancellation = worker.cancellation
        if worker.native is not None and not worker.native_done:
            if cancellation is None or not cancellation.done():
                return False
            try:
                cancellation.result()
            except BaseException:  # noqa: BLE001 -- failed cancellation remains capacity
                return False
            worker.native_done = True
        for future in futures:
            if not future.done():
                return False
            try:
                future.result()
            except BaseException:  # noqa: BLE001 -- failed cancellation remains capacity
                return False
        return worker.native is None or worker.native_done

    def ensure(
        self,
        key: _FleetKey,
        agents: Sequence[AgentWork],
        backend: Any,
        workspace: Workspace,
        control_id: str,
    ) -> None:
        """Launch missing compatible capacity without assigning logical work."""
        self._start()
        with self.lock:
            if self.stopped.is_set():
                return
            # An idle worker with another environment cannot safely execute
            # this graph. Retire it before counting the profile's hard limit.
            for worker in self.allocations.values():
                if (
                    worker.profile == key.profile_name
                    and not worker.retired
                    and worker.owner_run_id is None
                    and not self._matches(worker, key)
                ):
                    self._retire(worker)
            limit = self.executor.capacity[key.profile_name].max_workers
            live_profile = sum(
                worker.profile == key.profile_name and (not worker.retired or not self._retirement_complete(worker))
                for worker in self.allocations.values()
            )
            known_ids = set(self.allocations)
            for agent in agents:
                if live_profile >= limit:
                    break
                if agent.worker_id in known_ids:
                    continue
                worker = _Allocation(
                    agent.worker_id,
                    agent.profile,
                    _async_call(backend.launch_worker, agent),
                    started_at=time.monotonic(),
                    backend=backend,
                    workspace=workspace,
                    control_id=control_id,
                    runtime_key=key.runtime_key,
                    profile_key=key.profile_key,
                    expires_at=time.monotonic() + _agent_lifetime_s(self.executor),
                )
                self.allocations[worker.worker_id] = worker
                live_profile += 1
        self.wakeup.set()

    def acquire(self, key: _FleetKey, run_id: str, *, required_runtime_s: float) -> _Allocation | None:
        """Lease one ready compatible allocation, preferring this run's idle worker."""
        with self.lock:
            if self.stopped.is_set():
                return None
            now = time.monotonic()
            # A coordinator keeps its allocation leased between nodes. Check
            # that worker first, then fall back to unowned compatible capacity.
            # Both paths revalidate the finite native lifetime before every
            # assignment.
            for owner in (run_id, None):
                for worker in self.allocations.values():
                    if (
                        not self._matches(worker, key)
                        or worker.owner_run_id != owner
                        or worker.generation is None
                        or worker.job_id is not None
                        or worker.retired
                        or worker.maintenance
                    ):
                        continue
                    if worker.expires_at is not None and worker.expires_at - now <= required_runtime_s:
                        self._retire(worker)
                        continue
                    worker.owner_run_id = run_id
                    return worker
            return None

    def release(self, worker: _Allocation, run_id: str) -> bool:
        """Return a clean worker to the fleet; uncertain workers are retired."""
        with self.lock:
            if worker.owner_run_id != run_id:
                return False
            if (
                worker.retired
                or worker.job_id is not None
                or worker.attempt_id is not None
                or worker.generation is None
            ):
                self._retire(worker)
                return False
            worker.owner_run_id = None
            worker.cancel_attempt_id = None
            worker.execution_started_at = None
            worker.cleanup_reported = False
        self.wakeup.set()
        return True

    def exhausted(self, key: _FleetKey, agents: Sequence[AgentWork]) -> bool:
        """Whether every bootstrap for this key failed with no live capacity."""
        with self.lock:
            matching = [worker for worker in self.allocations.values() if self._matches(worker, key)]
            return (
                bool(agents)
                and all(agent.worker_id in self.allocations for agent in agents)
                and not any(not worker.retired for worker in matching)
            )

    def write_lease(
        self,
        worker: _Allocation,
        *,
        owner_run_id: str | None,
        stop: bool = False,
        force: bool = False,
    ) -> bool:
        """Serialize one worker's monotonic lease without holding the fleet lock during I/O."""
        with worker.lease_lock:
            with self.lock:
                if stop:
                    if not force and not (
                        worker.owner_run_id == owner_run_id or (worker.owner_run_id is None and worker.retired)
                    ):
                        return False
                    worker.lease_stopped = True
                    worker.retired = True
                    worker.owner_run_id = None
                elif (
                    self.stopped.is_set()
                    or worker.lease_stopped
                    or worker.retired
                    or worker.owner_run_id != owner_run_id
                ):
                    return False
                if worker.workspace is None or worker.control_id is None:
                    return False
                worker.lease_sequence += 1
                sequence = worker.lease_sequence
                workspace = worker.workspace
                control_id = worker.control_id
                cancel_attempt_id = worker.cancel_attempt_id
            _write(
                workspace,
                control_id,
                f"worker-{worker.worker_id}.lease.json",
                worker_id=worker.worker_id,
                sequence=sequence,
                stop=stop,
                cancel_attempt_id=cancel_attempt_id,
            )
            with self.lock:
                if worker.lease_sequence == sequence:
                    worker.last_lease = time.monotonic()
            return True

    def _retire(self, worker: _Allocation) -> None:
        worker.retired = True
        worker.owner_run_id = None
        worker.cancel_requested = True
        if worker.native is not None and not worker.native_done and worker.cancellation is None:
            _request_handle_cancellation(worker, worker.backend, worker.native)

    def _step_worker(self, worker: _Allocation) -> None:
        if worker.owner_run_id is not None:
            return
        complete, newly_reconciled, launch_error = _resolve_completed_launch(worker)
        if complete and newly_reconciled and launch_error is not None:
            _request_submitted_cancellations(worker, worker.backend, launch_error)
            worker.retired = True
            return
        if worker.retired:
            if worker.native is not None and not worker.native_done:
                _request_handle_cancellation(worker, worker.backend, worker.native)
            return
        if complete and launch_error is not None:
            worker.retired = True
            return
        if worker.workspace is None or worker.control_id is None:
            self._retire(worker)
            return
        record = _read(worker.workspace, worker.control_id, f"worker-{worker.worker_id}.state.json")
        if record is not None:
            generation = record.get("generation")
            if record.get("worker_id") != worker.worker_id or not isinstance(generation, str) or not generation:
                self._retire(worker)
                return
            if worker.generation is not None and worker.generation != generation:
                self._retire(worker)
                return
            worker.generation = generation
            if record.get("state") == "stopped":
                self._retire(worker)
                return
        now = time.monotonic()
        if worker.health is not None and worker.health.done():
            try:
                state = worker.health.result()
            except Exception as exc:  # noqa: BLE001 -- transient status loss does not fabricate failure
                logger.warning("Idle allocation health unavailable for %s: %s", worker.worker_id, type(exc).__name__)
            else:
                if state in ("done", "failed"):
                    worker.native_done = True
                    self._retire(worker)
                    return
            worker.health = None
        if worker.native is not None and worker.health is None and now - worker.last_health >= _HEALTH_INTERVAL:
            worker.health = _async_call(worker.backend.state, worker.native)
            worker.last_health = now
        if worker.generation is None and now - worker.started_at > self.executor.setup_timeout_s:
            self._retire(worker)
            return
        if now - worker.last_lease >= _LEASE_INTERVAL:
            self.write_lease(worker, owner_run_id=None)

    def step(self) -> None:
        """Advance bootstrap/health state for idle workers."""
        with self.lock:
            workers = [
                worker for worker in self.allocations.values() if worker.owner_run_id is None and not worker.maintenance
            ]
            for worker in workers:
                worker.maintenance = True
        for worker in workers:
            try:
                self._step_worker(worker)
            except Exception as exc:  # noqa: BLE001 -- a poisoned worker must not kill the fleet thread
                with self.lock:
                    self.errors.append(f"Fleet worker {worker.worker_id} failed: {type(exc).__name__}.")
                    self._retire(worker)
            finally:
                with self.lock:
                    worker.maintenance = False

    def _run(self) -> None:
        while not self.stopped.is_set():
            self.step()
            self.wakeup.wait(self.executor.poll_interval_s)
            self.wakeup.clear()

    def _revoke_lease(self, worker: _Allocation) -> None:
        """Write a terminal lease off the fleet ownership lock."""
        self.write_lease(worker, owner_run_id=None, stop=True, force=True)

    def close(self) -> None:
        """Stop every agent and cancel its native job within one shared deadline."""
        deadline = time.monotonic() + self.executor.shutdown_timeout_s
        self.stopped.set()
        self.wakeup.set()
        if self.thread is not None:
            self.thread.join(min(self.executor.poll_interval_s + 1, max(0.0, deadline - time.monotonic())))
            if self.thread.is_alive():
                self.errors.append("Fleet monitor did not stop before the shutdown deadline.")
        acquired = self.lock.acquire(timeout=max(0.0, deadline - time.monotonic()))
        if not acquired:
            self.errors.append("Fleet ownership lock did not become available before the shutdown deadline.")
            raise ExecutionError(" ".join(self.errors))
        try:
            workers = list(self.allocations.values())
        finally:
            self.lock.release()

        # Lease I/O can stall independently of native cancellation.  Start all
        # revocations concurrently, outside the ownership lock, then reconcile
        # every launch against the same deadline.
        revocations = [(worker.worker_id, _async_call(self._revoke_lease, worker)) for worker in workers]
        for worker in workers:
            if not _prepare_launch_cleanup(worker, worker.backend, deadline):
                self.errors.append(
                    f"Unresolved launch for fleet allocation {worker.worker_id}; late acceptance will be cancelled."
                )
        for worker_id, future in revocations:
            try:
                future.result(timeout=max(0.0, deadline - time.monotonic()))
            except Exception as exc:  # noqa: BLE001 -- native cancellation remains the hard fallback
                self.errors.append(f"Could not revoke fleet lease {worker_id}: {type(exc).__name__}.")
        _await_tracked_cancellations(workers, deadline, self.errors, message_prefix="Fleet cleanup unresolved for")
        if self.errors:
            raise ExecutionError(" ".join(self.errors))


@dataclass
class _ExecutionSession:
    """One local lifetime boundary for API state, graph threads, and agents."""

    executor_id: int
    api: Any
    fleet: _AgentFleet
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    runs: list[GraphCoordinator] = field(default_factory=list)
    runtime_keys: dict[str, str] = field(default_factory=dict, repr=False)
    _client: Any = field(default=None, init=False, repr=False)

    def api_client(self) -> Any:
        """Bind one lazy API client to this executor session, never to ambient context."""
        if self._client is None:
            self._client = self.api.client if self.api is not None else _load_external_skypilot()
        return self._client

    def runtime_token(self, runtime_key: str) -> str:
        """Return a session-local opaque token without persisting dotenv digests."""
        return self.runtime_keys.setdefault(runtime_key, uuid.uuid4().hex)

    def __getitem__(self, index: int) -> Any:
        """Retain the historical private tuple view used by integrations."""
        if index == 0:
            return self.executor_id
        if index == 1:
            return self.runs
        raise IndexError(index)


class GraphCoordinator:
    """Session-owned graph progress independent of UI and SDK health polling."""

    def __init__(
        self,
        executor: GraphSkyPilotExecutor,
        manifest: RunManifest,
        workspace: Workspace,
        backend: Any,
        fleet: _AgentFleet | None = None,
    ) -> None:
        """Create one coordinator with explicit backend and workspace ownership."""
        self.executor = executor
        self.manifest = manifest
        self.workspace = workspace
        self.backend = backend
        self.fleet = fleet
        self.control_id = manifest.control_id or manifest.run_id
        self.runtime_key = manifest.runtime_key or manifest.snapshot_key
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

    def _fleet_key(self, profile_name: str) -> _FleetKey:
        if self.fleet is None:
            msg = "A fleet compatibility key requires an attached execution session."
            raise RuntimeError(msg)
        return self.fleet.key(
            self.workspace,
            self.runtime_key,
            profile_name,
            self.executor.capacity[profile_name],
        )

    def _worker_backend(self, worker: _Allocation) -> Any:
        return worker.backend if worker.backend is not None else self.backend

    def _worker_workspace(self, worker: _Allocation) -> Workspace:
        return worker.workspace if worker.workspace is not None else self.workspace

    def _worker_control_id(self, worker: _Allocation) -> str:
        return worker.control_id or self.control_id

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

    def _lease_worker(self, worker: _Allocation, *, stop: bool = False) -> bool:
        """Publish one lease through its sole sequence owner."""
        if worker.dedicated or (worker.retired and not stop):
            return False
        if self.fleet is not None:
            return self.fleet.write_lease(
                worker,
                owner_run_id=self.manifest.run_id,
                stop=stop,
            )
        with worker.lease_lock:
            if worker.lease_stopped:
                return stop
            if stop:
                worker.lease_stopped = True
            worker.lease_sequence += 1
            _write(
                self._worker_workspace(worker),
                self._worker_control_id(worker),
                f"worker-{worker.worker_id}.lease.json",
                worker_id=worker.worker_id,
                sequence=worker.lease_sequence,
                stop=stop,
                cancel_attempt_id=worker.cancel_attempt_id,
            )
            worker.last_lease = time.monotonic()
            return True

    def _lease(self) -> None:
        self.sequence += 1
        for worker in self.allocations.values():
            self._lease_worker(worker)
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
        complete, newly_reconciled, launch_error = _resolve_completed_launch(worker)
        if complete and newly_reconciled and launch_error is not None:
            exc = launch_error
            worker.retired = True
            accepted = tuple(getattr(exc, "submitted_jobs", ()))
            _request_submitted_cancellations(worker, self._worker_backend(worker), exc)
            if accepted:
                self.errors.append(
                    f"Accepted allocation {worker.worker_id} could not be recorded; cancellation requested."
                )
            healthy = any(
                other is not worker and other.profile == worker.profile and not other.retired
                for other in self.allocations.values()
            )
            if worker.job_id is not None:
                self.graph.finish(worker.job_id, success=False, reason=f"Capacity launch failed: {type(exc).__name__}.")
            elif not healthy:
                self._fail_profile(worker.profile, f"Capacity launch failed: {type(exc).__name__}.")
            return
        if worker.retired:
            if worker.native is not None and not worker.native_done:
                _request_handle_cancellation(worker, self._worker_backend(worker), worker.native)
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
            record = _read(
                self._worker_workspace(worker),
                self._worker_control_id(worker),
                f"worker-{worker.worker_id}.state.json",
            )
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
                        _request_handle_cancellation(worker, self._worker_backend(worker), worker.native)
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
            worker.health = _async_call(self._worker_backend(worker).state, worker.native)
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
                _request_handle_cancellation(worker, self._worker_backend(worker), worker.native)
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
            _request_handle_cancellation(worker, self._worker_backend(worker), worker.native)

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
        # The immutable assignment is the durable fence before any payload is
        # sent. ``step()`` coalesces all logical-state changes into one index
        # write after scheduling the ready batch.
        if worker.dedicated:
            worker.launch = _async_call(self.backend.launch_dedicated, node, attempt_id)
        else:
            _write(
                self._worker_workspace(worker),
                self._worker_control_id(worker),
                f"worker-{worker.worker_id}.command.json",
                worker_id=worker.worker_id,
                generation=worker.generation,
                attempt_id=attempt_id,
                job_id=node.job_id,
                target_run_id=self.manifest.run_id,
                argv=node.argv,
                env=node.env,
                log_path=node.log_path,
                execution_timeout_s=float(node.resources["time"] * 60),
                setup_timeout_s=self.executor.setup_timeout_s,
                direct=node.direct,
                payload_name=node.payload_name,
            )

    def _schedule(self) -> None:
        while self.graph.ready:
            key = self.graph.ready.popleft()
            self.ready[self.graph.nodes[key].profile].append(key)
        if self.fleet is not None:
            self.fleet.step()
            for name, profile in self.executor.capacity.items():
                if profile.dedicated:
                    continue
                agents = [agent for agent in self.manifest.agents if agent.profile == name]
                if agents:
                    self.fleet.ensure(
                        self._fleet_key(name),
                        agents,
                        self.backend,
                        self.workspace,
                        self.control_id,
                    )
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
                if self.fleet is not None:
                    idle = self.fleet.acquire(
                        self._fleet_key(name),
                        self.manifest.run_id,
                        required_runtime_s=(
                            self.executor.setup_timeout_s
                            + min(node.resources["time"], self.executor.max_run_minutes) * 60
                            + self.executor.shutdown_timeout_s
                        ),
                    )
                    if idle is not None:
                        self.allocations[idle.worker_id] = idle
                        self._lease_worker(idle)
                        self.last_lease = time.monotonic()
                        self.heartbeat_at = time.time()
                        self._assign(idle, node)
                        queue.popleft()
                        continue
                    agents = [agent for agent in self.manifest.agents if agent.profile == name]
                    if self.fleet.exhausted(self._fleet_key(name), agents):
                        self._fail_profile(name, "No live workers remain; automatic replacement is disabled.")
                    break
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
                        worker = _Allocation(
                            agent.worker_id,
                            name,
                            Future(),
                            started_at=time.monotonic(),
                            backend=self.backend,
                            workspace=self.workspace,
                            control_id=self.control_id,
                            runtime_key=self.runtime_key,
                        )
                        self.allocations[worker.worker_id] = worker
                        self._lease_worker(worker)
                        self.last_lease = time.monotonic()
                        self.heartbeat_at = time.time()
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
                                    _request_handle_cancellation(worker, self._worker_backend(worker), worker.native)
                            elif not worker.dedicated:
                                worker.cancel_attempt_id = worker.attempt_id
                                self.sequence += 1
                                self._lease_worker(worker)
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
        """Release clean session agents and stop every other owned allocation."""
        deadline = time.monotonic() + self.executor.shutdown_timeout_s
        drained = True
        if not self.cancelled.is_set() and self.graph.complete:
            while any(worker.job_id is not None and not worker.retired for worker in self.allocations.values()):
                if time.monotonic() >= deadline:
                    drained = False
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
        reusable: set[str] = set()
        if self.fleet is not None and not self.cancelled.is_set() and self.graph.complete and drained:
            for worker in self.allocations.values():
                if not worker.dedicated and self.fleet.release(worker, self.manifest.run_id):
                    reusable.add(worker.worker_id)
        workers = list(self.allocations.values())
        for worker in workers:
            if worker.worker_id in reusable:
                continue
            if not worker.dedicated:
                try:
                    self._lease_worker(worker, stop=True)
                except Exception as exc:  # noqa: BLE001 -- native cancellation remains the hard fallback
                    self.errors.append(f"Could not revoke worker lease {worker.worker_id}: {type(exc).__name__}.")
            if not _prepare_launch_cleanup(worker, self._worker_backend(worker), deadline):
                self.errors.append(
                    f"Unresolved launch for allocation {worker.worker_id}; late acceptance will be cancelled."
                )
        _await_tracked_cancellations(workers, deadline, self.errors, message_prefix="Cleanup unresolved for allocation")

    def close(self, *, deadline: float | None = None) -> None:
        """Cancel unfinished work and wait only through the shutdown grace."""
        deadline = time.monotonic() + self.executor.shutdown_timeout_s + 1 if deadline is None else deadline
        if not self.graph.complete:
            self.cancelled.set()
        self.wakeup.set()
        if self.thread is not None:
            self.thread.join(max(0.0, deadline - time.monotonic()))
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
        """Keep API state, graph coordinators, and a reusable agent fleet together."""
        existing = _runs.get()
        if existing is not None and existing.executor_id == id(self):
            with _bound_active_session(existing.api) as session:
                yield session
            return
        api = managed_session(self.api_server_namespace) if self.manage_api_server else _bound_active_session(None)
        with api as session:
            execution = _ExecutionSession(id(self), session, _AgentFleet(self))
            token = _runs.set(execution)
            original: BaseException | None = None
            try:
                yield session
            except BaseException as exc:
                original = exc
                raise
            finally:
                try:
                    cleanup_failures: list[BaseException] = []
                    runs = list(execution.runs)
                    run_deadline = time.monotonic() + self.shutdown_timeout_s + 1
                    # Revoke every run before waiting for any one coordinator;
                    # a slow first close must not leave later graphs executing.
                    for run in runs:
                        try:
                            if isinstance(run, GraphCoordinator) and not run.graph.complete:
                                run.cancelled.set()
                            if isinstance(run, GraphCoordinator):
                                run.wakeup.set()
                        except BaseException as exc:  # noqa: BLE001 -- defer until all cleanup is attempted
                            cleanup_failures.append(exc)
                    for run in runs:
                        try:
                            if isinstance(run, GraphCoordinator):
                                run.close(deadline=run_deadline)
                            else:
                                run.close()
                        except BaseException as exc:  # noqa: BLE001 -- do not skip later runs
                            cleanup_failures.append(exc)
                    try:
                        execution.fleet.close()
                    except BaseException as exc:  # noqa: BLE001 -- the fleet cleanup remains mandatory
                        cleanup_failures.append(exc)
                    if cleanup_failures:
                        if original is not None:
                            for failure in cleanup_failures:
                                detail = str(failure) or type(failure).__name__
                                original.add_note(detail)
                        elif all(isinstance(failure, ExecutionError) for failure in cleanup_failures):
                            raise ExecutionError(" ".join(str(failure) for failure in cleanup_failures))
                        else:
                            primary = next(
                                failure for failure in cleanup_failures if not isinstance(failure, ExecutionError)
                            )
                            for failure in cleanup_failures:
                                if failure is not primary:
                                    detail = str(failure) or type(failure).__name__
                                    primary.add_note(f"Additional SkyPilot session cleanup failure: {detail}")
                            raise primary
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
        active = next((run for run in owner.runs if run.manifest.run_id == run_id), None) if owner is not None else None
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
            sky = _load_external_skypilot()
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
        ambient_owner = _runs.get()
        owner = ambient_owner if ambient_owner is not None and ambient_owner.executor_id == id(self) else None
        if self.lifecycle == "attached" and owner is None:
            msg = "Attached coordinator has no owning session."
            raise SubmissionError(msg)
        control_id = owner.session_id if self.lifecycle == "attached" and owner is not None else snapshot.submission_id
        raw_runtime_key = getattr(snapshot, "runtime_key", snapshot.snapshot_key)
        runtime_key = (
            owner.runtime_token(raw_runtime_key)
            if self.lifecycle == "attached" and owner is not None
            else snapshot.submission_id
        )
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
                direct=True,
                payload_name=f"{job_id}.pkl",
            )
            nodes.append(node)
            jobs[unit] = SkyPilotTaskJob(unit, job_id, log_path, workspace, snapshot.submission_id)
            cast("SkyPilotTaskJob", jobs[unit]).stale_timeout_s = self.setup_timeout_s
            progress(1)
        agents = []
        profile_widths = profile_dependency_widths(nodes)
        for name in {node.profile for node in nodes}:
            profile = self.capacity[name]
            if profile.dedicated:
                continue
            for _ in range(min(profile.max_workers, profile_widths[name])):
                worker_id = uuid.uuid4().hex
                fn = functools.partial(
                    run_worker_agent,
                    workspace,
                    control_id,
                    worker_id,
                    lease_timeout_s=_AGENT_LEASE_TIMEOUT_S,
                    max_runtime_s=_agent_lifetime_s(self),
                    poll_interval_s=self.poll_interval_s,
                )
                agent_id, argv, env, log_path = _prepare_control(snapshot, workspace, fn)
                agents.append(AgentWork(worker_id, name, agent_id, argv, env, str(log_path)))
        manifest = RunManifest(
            snapshot.submission_id,
            snapshot.snapshot_key,
            nodes,
            agents,
            control_id=control_id,
            runtime_key=runtime_key,
        )
        workspace.put_job_file(
            manifest.run_id,
            "run-work-units.pkl",
            cloudpickle.dumps({prepared[unit][0]: unit for unit in pending_work_units}),
        )
        workspace.put_job_file(manifest.run_id, "run-manifest.json", msgspec.json.encode(manifest))
        sky = owner.api_client() if owner is not None else _load_external_skypilot()
        backend = _SkyCapacityBackend(
            self,
            manifest,
            workspace,
            api_session=owner.api if owner is not None else None,
            sky=sky,
        )
        run = GraphCoordinator(
            self,
            manifest,
            workspace,
            backend,
            owner.fleet if self.lifecycle == "attached" and owner is not None else None,
        )
        run.persist()
        if self.lifecycle == "detached":
            fn = functools.partial(_run_remote, self, manifest, workspace)
            job_id, argv, env, log_path = _prepare_control(snapshot, workspace, fn)
            backend.launch_coordinator(job_id, argv, env, log_path)
        else:
            if owner is None:
                msg = "Attached coordinator has no owning session."
                raise SubmissionError(msg)
            owner.runs.append(run)
            for unit in pending_work_units:
                cast("SkyPilotTaskJob", jobs[unit]).coordinator = run
            run.start()


def _prepare_control(
    snapshot: ProjectSnapshot, workspace: Workspace, function: Callable[[], None]
) -> tuple[str, list[str], dict[str, str], Path]:
    """Stage one control payload through the normal trusted snapshot bootstrap."""
    from misen.tasks import Task
    from misen.utils.work_unit import WorkUnit

    payload = cloudpickle.dumps(function)
    return snapshot.prepare_job(WorkUnit(Task(_control_placeholder, payload=payload), set()), workspace)


@meta(id="misen-skypilot-control-v2")
def _control_placeholder(payload: bytes | None = None) -> None:
    """Invoke a trusted internal role through a stable work-unit identity."""
    if payload is not None:
        function = cloudpickle.loads(payload)
        if not callable(function):
            msg = "SkyPilot control payload did not decode to a callable."
            raise TypeError(msg)
        function()


def _run_remote(executor: GraphSkyPilotExecutor, manifest: RunManifest, workspace: Workspace) -> None:

    if executor.lifecycle != "detached" or executor.manage_api_server:
        msg = "A remote coordinator requires detached lifecycle and an externally managed SkyPilot API."
        raise ExecutionError(msg)
    if (
        not os.environ.get("SKYPILOT_API_SERVER_ENDPOINT", "").strip()
        or not os.environ.get("SKYPILOT_SERVICE_ACCOUNT_TOKEN", "").strip()
    ):
        msg = "Remote coordinator is missing its injected SkyPilot endpoint or service-account token."
        raise ExecutionError(msg)
    sky = _load_external_skypilot()
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
    run = GraphCoordinator(
        executor,
        manifest,
        workspace,
        _SkyCapacityBackend(executor, manifest, workspace, api_session=None, sky=sky),
    )
    run.run()
    if run.errors:
        raise ExecutionError(" ".join(run.errors))


class _SkyCapacityBackend:
    """Native allocation operations; never invoked once per reusable work unit."""

    def __init__(
        self,
        executor: GraphSkyPilotExecutor,
        manifest: RunManifest,
        workspace: Workspace,
        *,
        api_session: Any,
        sky: Any,
    ) -> None:

        self.executor = executor
        self.manifest = manifest
        self.workspace = workspace
        self.api_session = api_session
        self.sky = sky
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
            with _bound_active_session(self.api_session):
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
                try:
                    return self._native(profile, previous, log_path)
                except SubmissionError:
                    raise
                except Exception as exc:
                    msg = "The durably accepted allocation could not be restored; no new submission was issued."
                    raise SubmissionError(msg) from exc
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
            try:
                _write(self.workspace, self.manifest.run_id, record_name, **record)
            except Exception as exc:
                try:
                    native = self._native(profile, record, log_path)
                except Exception:  # noqa: BLE001 -- preserve the durable-storage failure as the primary error
                    exc.add_note(
                        "Additionally, an in-memory cancellation handle could not be constructed; "
                        "the pre-submission record prevents automatic replay."
                    )
                    msg = (
                        "SkyPilot accepted the allocation, but neither its request record nor a native handle survived."
                    )
                    raise SubmissionError(msg) from exc
                msg = "SkyPilot accepted the allocation, but its durable request record could not be persisted."
                raise SubmissionError(msg, submitted_jobs=(native,)) from exc
            try:
                return self._native(profile, record, log_path)
            except Exception as exc:
                msg = (
                    "SkyPilot accepted the allocation and its request identity is durable, "
                    "but the native handle could not be constructed."
                )
                raise SubmissionError(msg) from exc

    def launch_worker(self, agent: AgentWork) -> Any:
        return self._launch(
            self.executor.capacity[agent.profile],
            agent.worker_id,
            agent.job_id,
            agent.argv,
            agent.env,
            log_path=Path(agent.log_path),
            time_minutes=math.ceil(_agent_lifetime_s(self.executor) / 60),
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
