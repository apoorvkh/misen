"""Native SkyPilot allocation handles and SDK result normalization."""

from __future__ import annotations

import importlib
import logging
import time
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

from misen.exceptions import (
    ConfigError,
    ExecutionError,
    MisenError,
    StatusQueryError,
    StorageError,
)
from misen.executor import Job, JobState, _JobRecord
from misen.utils.job_dependencies import dependency_state_name, publish_dependency_state

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from misen.utils.work_unit import WorkUnit
    from misen.workspace import Workspace

from .api import active_session

logger = logging.getLogger("misen.executors.skypilot")

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
