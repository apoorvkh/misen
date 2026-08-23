"""Stable failures for Misen-owned subsystems.

Ordinary API contract errors use Python's built-ins, and user-task exceptions
remain unchanged when they can cross the execution boundary directly.
"""

from __future__ import annotations

from collections.abc import Iterable  # noqa: TC003 -- public hints resolve at runtime
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path  # noqa: TC003 -- public hints resolve at runtime
from typing import ClassVar, Protocol

__all__ = [
    "CacheError",
    "CliUsageError",
    "ConfigError",
    "ErrorCode",
    "ExecutionError",
    "ExperimentReferenceError",
    "HashError",
    "JobFailedError",
    "JobFailure",
    "LockUnavailableError",
    "MisenError",
    "SerializationError",
    "SnapshotError",
    "StatusQueryError",
    "StorageError",
    "SubmissionError",
    "SubmittedJob",
    "WorkspaceError",
]


class ErrorCode(StrEnum):
    """Stable, machine-readable categories for user-facing error handling."""

    MISEN = "misen"
    CACHE = "cache"
    CLI_USAGE = "cli_usage"
    CONFIG = "config"
    HASH = "hash"
    LOCK_UNAVAILABLE = "lock_unavailable"
    SERIALIZATION = "serialization"
    WORKSPACE = "workspace"
    STORAGE = "storage"
    SNAPSHOT = "snapshot"
    EXECUTION = "execution"
    SUBMISSION = "submission"
    STATUS_QUERY = "status_query"
    JOB_FAILED = "job_failed"
    EXPERIMENT_REFERENCE = "experiment_reference"


class MisenError(Exception):
    """Base class for stable, Misen-owned domain failures."""

    code: ClassVar[ErrorCode] = ErrorCode.MISEN

    def __init_subclass__(cls, *, code: ErrorCode | None = None) -> None:
        """Attach an explicitly declared diagnostic code to a subclass."""
        super().__init_subclass__()
        if code is not None:
            cls.code = code


class CacheError(MisenError, code=ErrorCode.CACHE):
    """Raised on cache misses or missing prerequisite cache entries."""


class CliUsageError(MisenError, code=ErrorCode.CLI_USAGE):
    """Raised when command-line input is valid syntax but cannot be acted on."""


class ConfigError(MisenError, code=ErrorCode.CONFIG):
    """Raised when Misen configuration cannot be resolved or applied."""


class HashError(MisenError, code=ErrorCode.HASH):
    """Raised when ``stable_hash`` cannot hash a value."""


class LockUnavailableError(MisenError, code=ErrorCode.LOCK_UNAVAILABLE):
    """Raised when a lock cannot be acquired or ownership is lost."""


class SerializationError(MisenError, code=ErrorCode.SERIALIZATION):
    """Raised on serializer save/load failures."""


class WorkspaceError(MisenError, code=ErrorCode.WORKSPACE):
    """Base class for workspace access and lifecycle failures."""


class StorageError(WorkspaceError, code=ErrorCode.STORAGE):
    """Raised when a workspace's backing store cannot complete an operation."""


class SnapshotError(MisenError, code=ErrorCode.SNAPSHOT):
    """Raised when a project snapshot cannot be created or materialized."""


class ExecutionError(MisenError, code=ErrorCode.EXECUTION):
    """Base class for executor and job lifecycle failures."""


class SubmittedJob(Protocol):
    """Typed view of the job details exposed by a partial submission."""

    job_id: str | None
    log_path: Path | None

    @property
    def label(self) -> str:
        """Return a human-readable job label."""
        ...

    def state(self) -> str:
        """Return the backend's current state for this job."""
        ...

    def wait(self, poll_s: float = 0.5) -> None:
        """Wait until this job reaches a terminal state."""
        ...


class SubmissionError(ExecutionError, code=ErrorCode.SUBMISSION):
    """Raised when submission fails, retaining any already-accepted jobs."""

    submitted_jobs: tuple[SubmittedJob, ...]

    def __init__(self, message: str, *, submitted_jobs: Iterable[SubmittedJob] = ()) -> None:
        """Initialize a submission failure with any already-live jobs."""
        super().__init__(message)
        self.submitted_jobs = tuple(submitted_jobs)


class StatusQueryError(ExecutionError, code=ErrorCode.STATUS_QUERY):
    """Raised when an executor cannot determine job status reliably."""

    retryable: bool

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        """Initialize a status failure and record whether polling may recover."""
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class JobFailure:
    """Serializable facts describing one failed job."""

    label: str
    job_id: str | None = None
    log_path: str | None = None
    reason: str | None = None


class JobFailedError(ExecutionError, code=ErrorCode.JOB_FAILED):
    """Raised when one or more submitted jobs finish unsuccessfully."""

    failures: tuple[JobFailure, ...]

    def __init__(self, message: str, *, failures: Iterable[JobFailure] = ()) -> None:
        """Initialize an execution error with structured failed-job facts."""
        super().__init__(message)
        self.failures = tuple(failures)


class ExperimentReferenceError(MisenError, code=ErrorCode.EXPERIMENT_REFERENCE):
    """Raised when an experiment reference cannot be resolved."""
