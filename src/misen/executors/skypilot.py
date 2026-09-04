"""SkyPilot managed-job executor.

The executor treats SkyPilot as a control plane only. Misen's workspace is
still the data plane: a remote-capable workspace transports the immutable
project snapshot, job payloads, results, locks, and logs, while SkyPilot
provisions compute and owns each durable job lifecycle. Misen submits one
managed job per work unit and enforces dependencies through durable workspace
markers. Independent jobs are submitted eagerly and execute concurrently when
backend or pool capacity permits.
"""

from __future__ import annotations

import importlib
import logging
import math
import re
import shlex
import time
from enum import Enum
from functools import cache
from typing import TYPE_CHECKING, Any, ClassVar, cast

import msgspec

from misen.exceptions import ConfigError, ExecutionError, MisenError, StatusQueryError, StorageError, SubmissionError
from misen.executor import Executor, Job, JobState, _JobRecord
from misen.task_metadata import AcceleratorType
from misen.utils.dask_runtime import (
    DEFAULT_DASK_SCHEDULER_PORT,
    DEFAULT_DASK_STARTUP_TIMEOUT,
    MAX_DASK_SCHEDULER_PORT,
    MIN_DASK_SCHEDULER_PORT,
    managed_ranked_cluster_script,
)
from misen.utils.job_dependencies import dependency_state_name, publish_dependency_state
from misen.utils.resource_env import resource_environment
from misen.utils.runtime_events import work_unit_label

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from misen.utils.graph import DependencyGraph
    from misen.utils.snapshot import ProjectSnapshot
    from misen.utils.work_unit import WorkUnit
    from misen.workspace import Workspace

__all__ = ("SkyPilotExecutor", "SkyPilotJob")

logger = logging.getLogger(__name__)

_SKYPILOT_INSTALL = 'uv pip install "misen[skypilot]"'
_SKYPILOT_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_SKYPILOT_POOL_NAME = re.compile(r"[a-zA-Z](?:[-_.a-zA-Z0-9]*[a-zA-Z0-9])?")
_QUEUE_FIELDS = ("job_id", "task_id", "status", "failure_reason", "end_at")
_RECOVERY_QUEUE_FIELDS = ("job_id", "task_id", "job_name")
_ACTIVE_REQUEST_STATES = frozenset({"PENDING", "WAITING", "RUNNING"})
_CANCELLED_REQUEST_RECONCILE_S = 60
_CONTROLLER_FAILURE_KILL_GRACE_S = 30
_SKYPILOT_NODE_RANK_ENV = "SKYPILOT_NODE_RANK"
_SKYPILOT_NODE_IPS_ENV = "SKYPILOT_NODE_IPS"

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


@cache
def _pre_pool_key_type(executor_type: type[msgspec.Struct]) -> type[msgspec.Struct]:
    """Build a hash-only structural twin of an executor before ``pool`` existed."""
    fields = tuple(field for field in executor_type.__struct_fields__ if field != "pool")
    return msgspec.defstruct(
        executor_type.__name__,
        fields,
        module=executor_type.__module__,
        namespace={"__qualname__": executor_type.__qualname__},
        frozen=True,
    )


def _load_skypilot() -> Any:
    """Load the optional SkyPilot SDK on first use."""
    try:
        return importlib.import_module("sky")
    except ModuleNotFoundError as exc:
        if exc.name != "sky":
            raise
        msg = f"SkyPilotExecutor requires SkyPilot >=0.13; install it with `{_SKYPILOT_INSTALL}`."
        raise ConfigError(msg) from exc


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

    def state(self) -> JobState:
        """Return this managed job's normalized SkyPilot state."""
        return type(self).bulk_state([self]).get(self, "unknown")

    def cancel(self) -> None:
        """Cancel an unresolved launch request or its assigned managed job."""
        sky = _load_skypilot()
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

        sky = _load_skypilot()
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


class SkyPilotExecutor(Executor[SkyPilotJob]):
    """Run Misen work units as dependency-aware SkyPilot managed jobs.

    ``infra`` is passed through to SkyPilot and accepts any infrastructure
    understood by the installed SkyPilot version (for example ``"aws"``,
    ``"azure/eastus"``, ``"k8s/my-context"``, ``"ssh/my-pool"``, or
    ``"slurm/my-cluster"``), or an ordered list of alternatives. SkyPilot
    remains responsible for provider dependencies, credentials, configuration,
    and backend-specific feature checks. CPU and memory requests are forwarded
    as per-node minimums.

    ``job_recovery`` selects SkyPilot's infrastructure-recovery strategy.
    This adapter deliberately accepts only the strategy name, leaving
    application-error restarts disabled so task side effects are not repeated
    implicitly.

    ``pool`` routes every managed job to an existing SkyPilot worker pool.
    Pool workers can be reused across jobs, preserving Misen's worker-local
    environment cache and avoiding repeated instance provisioning. The pool's
    worker shape must be compatible with every submitted work unit. Pending
    work units must be dependency-independent because pool workers are
    exclusive and a descendant admitted before its parent could deadlock.

    Misen accelerator types describe programming backends (``cuda``, ``tpu``,
    etc.), whereas SkyPilot requires concrete hardware names. Configure the
    candidate mapping explicitly with ``accelerators``. If tasks declare a
    minimum per-device memory, also provide each candidate's capacity in
    ``accelerator_memory`` so the executor can filter safely.

    Without a pool, arbitrary Misen DAGs are supported by submitting one
    managed-job launch request per work unit. Requests are accepted eagerly,
    while a worker-side workspace gate delays user code until its parents
    finish. This preserves concurrency up to available backend capacity, plus
    durable dependency gates for jobs that reach worker code, at the cost of
    provisioning descendants while they wait. A pre-worker failure requires
    status observation for prompt propagation; otherwise descendants fail at
    their cumulative timeout. Multi-node tasks normally execute the Misen
    worker once on rank zero. A work unit requesting ``DASK_CLIENT`` instead
    receives one private Dask worker per node, with its scheduler and Misen
    coordinator on rank zero.
    """

    infra: str | list[str] = "aws"
    instance_type: str | None = None
    accelerators: dict[AcceleratorType, list[str]] = msgspec.field(default_factory=dict)
    accelerator_memory: dict[str, int] = msgspec.field(default_factory=dict)
    use_spot: bool = False
    image_id: str | None = None
    disk_size: int | None = None
    max_hourly_cost: float | None = None
    job_recovery: str | None = None
    dask_startup_timeout: int = DEFAULT_DASK_STARTUP_TIMEOUT
    dask_scheduler_port: int = DEFAULT_DASK_SCHEDULER_PORT
    name_prefix: str = "misen"
    pool: str | None = None
    _job_class: ClassVar[type[Job] | None] = SkyPilotJob
    _config_validation_errors: ClassVar[tuple[type[Exception], ...]] = (ValueError,)

    def __post_init__(self) -> None:
        """Normalize configuration and reject modes unsafe on remote workers."""
        self.accelerators = msgspec.convert(self.accelerators, type=dict[AcceleratorType, list[str]])
        self.accelerator_memory = msgspec.convert(self.accelerator_memory, type=dict[str, int])
        infras = [self.infra] if isinstance(self.infra, str) else self.infra
        if not infras or any(not isinstance(infra, str) or not infra.strip() for infra in infras):
            msg = "infra must be a non-empty SkyPilot infrastructure string or list of strings."
            raise ValueError(msg)
        normalized_infras = [infra.strip() for infra in infras]
        if len(set(normalized_infras)) != len(normalized_infras):
            msg = "infra must not contain duplicate alternatives."
            raise ValueError(msg)
        self.infra = normalized_infras[0] if isinstance(self.infra, str) else normalized_infras
        if not self.snapshot:
            msg = "SkyPilotExecutor requires snapshot=True; live project paths are not visible on remote workers."
            raise ValueError(msg)
        if self.prewarm_envs:
            msg = "SkyPilotExecutor requires prewarm_envs=False; submit-host environments are not worker-visible."
            raise ValueError(msg)
        for field_name in ("instance_type", "image_id", "job_recovery"):
            value = getattr(self, field_name)
            if value is not None:
                if not value.strip():
                    msg = f"{field_name} must be a non-empty string when set."
                    raise ValueError(msg)
                setattr(self, field_name, value.strip())
        if not _SKYPILOT_NAME.fullmatch(self.name_prefix) or len(self.name_prefix) > 30:  # noqa: PLR2004
            msg = "name_prefix must be at most 30 lowercase letters, digits, or single hyphen-separated words."
            raise ValueError(msg)
        if self.pool is not None and (
            not isinstance(self.pool, str) or _SKYPILOT_POOL_NAME.fullmatch(self.pool) is None
        ):
            msg = (
                "pool must start with a letter, end with a letter or digit, and contain only letters, digits, "
                "periods, underscores, or hyphens."
            )
            raise ValueError(msg)
        for accelerator_type, models in self.accelerators.items():
            if not models or any(not model.strip() for model in models):
                msg = f"accelerators[{accelerator_type!r}] must contain one or more non-empty model names."
                raise ValueError(msg)
            normalized_models = [model.strip() for model in models]
            if len(set(normalized_models)) != len(normalized_models):
                msg = f"accelerators[{accelerator_type!r}] must not contain duplicate model names."
                raise ValueError(msg)
            self.accelerators[accelerator_type] = normalized_models
        if any(not model.strip() for model in self.accelerator_memory):
            msg = "accelerator_memory keys must be non-empty SkyPilot model names."
            raise ValueError(msg)
        normalized_memory = {model.strip(): memory for model, memory in self.accelerator_memory.items()}
        if len(normalized_memory) != len(self.accelerator_memory):
            msg = "accelerator_memory must not contain duplicate model names after trimming whitespace."
            raise ValueError(msg)
        self.accelerator_memory = normalized_memory
        if any(isinstance(memory, bool) or memory < 1 for memory in self.accelerator_memory.values()):
            msg = "accelerator_memory values must be positive integer GiB capacities."
            raise ValueError(msg)
        if self.disk_size is not None and (isinstance(self.disk_size, bool) or self.disk_size < 1):
            msg = "disk_size must be a positive integer number of GiB."
            raise ValueError(msg)
        if self.max_hourly_cost is not None and (
            isinstance(self.max_hourly_cost, bool)
            or not math.isfinite(self.max_hourly_cost)
            or self.max_hourly_cost <= 0
        ):
            msg = "max_hourly_cost must be positive."
            raise ValueError(msg)
        if (
            isinstance(self.dask_startup_timeout, bool)
            or not isinstance(self.dask_startup_timeout, int)
            or self.dask_startup_timeout < 1
        ):
            msg = "dask_startup_timeout must be a positive integer number of seconds."
            raise ValueError(msg)
        if (
            isinstance(self.dask_scheduler_port, bool)
            or not isinstance(self.dask_scheduler_port, int)
            or not MIN_DASK_SCHEDULER_PORT <= self.dask_scheduler_port <= MAX_DASK_SCHEDULER_PORT
        ):
            msg = "dask_scheduler_port must be an integer between 1024 and 65535."
            raise ValueError(msg)

    def _validate_submission(
        self,
        *,
        work_graph: DependencyGraph[WorkUnit],
        pending_work_units: Sequence[WorkUnit],
        workspace: Workspace,
    ) -> None:
        """Reject local-only storage and validate every remote resource request."""
        del work_graph
        pending_set = set(pending_work_units)
        if self.pool is not None and any(work_unit.dependencies & pending_set for work_unit in pending_work_units):
            msg = (
                "SkyPilot pools currently require dependency-independent pending work units: pool workers are "
                "exclusive, so a descendant admitted before its parent could wait on that parent indefinitely. "
                "Run dependent stages separately, ensure their parents are already cached, or omit executor.pool."
            )
            raise ConfigError(msg)
        transport = workspace.bootstrap_transport()
        if transport is None:
            msg = (
                "SkyPilotExecutor requires a remotely fetchable workspace transport; "
                "use CloudWorkspace with worker IAM/service-account access."
            )
            raise ConfigError(msg)
        if workspace.get_temp_dir().is_absolute():
            msg = (
                "SkyPilotExecutor requires a relative workspace cache_dir so worker payload/log paths are valid "
                "on ephemeral hosts (for CloudWorkspace, use cache_dir='.cache/misen')."
            )
            raise ConfigError(msg)
        if not workspace.supports_job_file_reads():
            msg = "SkyPilotExecutor requires a workspace that supports submission-file coordination reads."
            raise ConfigError(msg)
        sky = _load_skypilot()
        for work_unit in pending_work_units:
            try:
                self._resource_options(sky, work_unit)
            except SubmissionError:
                raise
            except Exception as exc:
                msg = f"Invalid SkyPilot resources for {work_unit_label(work_unit)}: {exc}"
                raise SubmissionError(msg) from exc

    def _job_key_identity(self) -> object:
        """Preserve pre-pool reattachment keys when the pool is unset."""
        if self.pool is not None:
            return self
        legacy_type = _pre_pool_key_type(type(self))
        return legacy_type(*(getattr(self, field) for field in legacy_type.__struct_fields__))

    def _accelerator_models(self, work_unit: WorkUnit) -> Sequence[str | None]:
        """Resolve a generic accelerator request to concrete SkyPilot models."""
        requested = work_unit.resources
        if not requested["accelerators"]:
            return [None]

        accelerator_type = requested["accelerator_type"]
        models = list(self.accelerators.get(accelerator_type, ()))
        if not models:
            msg = (
                f"No SkyPilot accelerator models are configured for {accelerator_type!r}; "
                f"set executor.accelerators.{accelerator_type}."
            )
            raise SubmissionError(msg)
        if minimum_memory := requested["accelerator_memory"]:
            models = [model for model in models if self.accelerator_memory.get(model, 0) >= minimum_memory]
            if not models:
                msg = (
                    f"No configured {accelerator_type!r} SkyPilot accelerator has the requested "
                    f"minimum {minimum_memory} GiB/device; configure matching accelerator_memory capacities."
                )
                raise SubmissionError(msg)
        return models

    def _resource_options(self, sky: Any, work_unit: WorkUnit) -> object:
        """Translate one generic Misen request into SkyPilot resource choices."""
        requested = work_unit.resources
        infras = [self.infra] if isinstance(self.infra, str) else self.infra
        models = self._accelerator_models(work_unit)
        validate_locally = sky.server.common.is_api_server_local()

        common_options: dict[str, object] = {
            "cpus": f"{requested['cpus']}+",
            "memory": f"{requested['memory']}+",
            "use_spot": self.use_spot,
        }
        for key in ("instance_type", "image_id", "disk_size", "max_hourly_cost", "job_recovery"):
            if (value := getattr(self, key)) is not None:
                common_options[key] = value

        options: list[object] = []
        for infra in infras:
            for model in models:
                resource_options = {"infra": infra, **common_options}
                if model is not None:
                    resource_options["accelerators"] = {model: requested["accelerators"]}
                option = sky.Resources(**resource_options)
                # Full validation consults local Kubernetes contexts, SSH node
                # pools, and Slurm configuration. A remote API server owns
                # those settings and validates them in jobs.launch instead.
                if validate_locally:
                    option.validate()
                options.append(option)
        return options[0] if len(options) == 1 else options

    @staticmethod
    def _run_command(
        argv: list[str],
        env: dict[str, str],
        log_path: Path,
        *,
        time_minutes: int,
        nodes: int,
        cpus: int,
        memory_gib: int,
        uses_dask_client: bool,
        dask_startup_timeout: int,
        dask_scheduler_port: int,
    ) -> str:
        """Render one bounded rank-aware SkyPilot worker command."""
        command = (
            managed_ranked_cluster_script(
                argv,
                environment=env,
                workers=nodes,
                cpus=cpus,
                memory_gib=memory_gib,
                startup_timeout=dask_startup_timeout,
                node_rank_env=_SKYPILOT_NODE_RANK_ENV,
                node_ips_env=_SKYPILOT_NODE_IPS_ENV,
                scheduler_port=dask_scheduler_port,
            )
            if uses_dask_client
            else shlex.join(["env", *(f"{key}={value}" for key, value in env.items()), *argv])
        )
        lines = ["set -o pipefail"]
        if nodes > 1 and not uses_dask_client:
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

    def _dispatch(
        self,
        work_unit: WorkUnit,
        dependencies: set[SkyPilotJob],
        workspace: Workspace,
        snapshot: ProjectSnapshot | None,
    ) -> SkyPilotJob:
        """Submit one asynchronous managed-job request with durable parent gates."""
        if snapshot is None:  # guarded by __post_init__; retain the boundary invariant
            msg = "SkyPilotExecutor cannot dispatch without a project snapshot."
            raise SubmissionError(msg)

        resources = work_unit.resources
        dependency_jobs = {
            dependency.work_unit: (dependency.submission_id, cast("str", dependency.job_id))
            for dependency in dependencies
        }
        job_id, argv, env, log_path = snapshot.prepare_job(
            work_unit=work_unit,
            workspace=workspace,
            dependency_jobs=dependency_jobs,
        )
        env = env | resource_environment()
        deadline_minutes = resources["time"] + max(
            (dependency.deadline_minutes for dependency in dependencies),
            default=0,
        )
        name = f"{self.name_prefix}-{snapshot.submission_id.lower()}-{job_id.lower()}"
        sky = _load_skypilot()
        try:
            task = sky.Task(
                name=name,
                run=self._run_command(
                    argv,
                    env,
                    log_path,
                    time_minutes=deadline_minutes,
                    nodes=resources["nodes"],
                    cpus=resources["cpus"],
                    memory_gib=resources["memory"],
                    uses_dask_client=work_unit.uses_dask_client,
                    dask_startup_timeout=self.dask_startup_timeout,
                    dask_scheduler_port=self.dask_scheduler_port,
                ),
                num_nodes=resources["nodes"],
                resources=self._resource_options(sky, work_unit),
                api_server_access=False,
            )
            request_id = sky.jobs.launch(task, name=name, pool=self.pool)
        except SubmissionError:
            raise
        except Exception as exc:
            msg = f"SkyPilot failed to submit managed job {name!r}: {exc}"
            raise SubmissionError(msg) from exc

        if not isinstance(request_id, str) or not request_id:
            msg = f"SkyPilot returned an invalid launch request ID for job {name!r}: {request_id!r}."
            raise SubmissionError(msg)

        logger.info("Submitted SkyPilot launch request for %s (request_id=%s).", name, request_id)
        return SkyPilotJob(
            work_unit=work_unit,
            job_id=job_id,
            managed_job_id=None,
            submission_id=snapshot.submission_id,
            deadline_minutes=deadline_minutes,
            log_path=log_path,
            workspace=workspace,
            request_id=request_id,
            managed_job_name=name,
        )
