"""Tests for the public Misen exception contract."""
# ruff: noqa: D103, S101

from __future__ import annotations

from typing import get_type_hints

import misen
from misen.exceptions import (
    ErrorCode,
    ExecutionError,
    JobFailedError,
    JobFailure,
    MisenError,
    StatusQueryError,
    StorageError,
    SubmissionError,
    SubmittedJob,
    WorkspaceError,
)


def test_domain_hierarchy_and_codes() -> None:
    assert issubclass(StorageError, WorkspaceError)
    assert issubclass(WorkspaceError, MisenError)
    assert issubclass(StatusQueryError, ExecutionError)
    assert issubclass(ExecutionError, MisenError)
    assert StorageError.code is ErrorCode.STORAGE
    assert StatusQueryError.code is ErrorCode.STATUS_QUERY


def test_job_failed_error_retains_structured_failures() -> None:
    failure = JobFailure(label="train", job_id="42", log_path="train.log", reason="exit 1")
    error = JobFailedError("One job failed", failures=[failure])

    assert str(error) == "One job failed"
    assert error.failures == (failure,)
    assert error.code is ErrorCode.JOB_FAILED


def test_structured_exception_annotations_are_runtime_resolvable() -> None:
    assert get_type_hints(SubmissionError)["submitted_jobs"] == tuple[SubmittedJob, ...]
    assert "submitted_jobs" in get_type_hints(SubmissionError.__init__)
    assert "failures" in get_type_hints(JobFailedError.__init__)


def test_public_package_exports_domain_exceptions() -> None:
    for name in (
        "ErrorCode",
        "CliUsageError",
        "ExecutionError",
        "ExperimentReferenceError",
        "JobFailedError",
        "JobFailure",
        "LockUnavailableError",
        "SnapshotError",
        "StatusQueryError",
        "StorageError",
        "SubmittedJob",
        "SubmissionError",
        "WorkspaceError",
    ):
        assert getattr(misen, name) is not None
