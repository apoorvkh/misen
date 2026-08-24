"""Durable worker-side dependency gates for remote executors."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, TypeVar

from misen.exceptions import ExecutionError, StorageError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from misen.workspace import Workspace

__all__ = ["dependency_state_name", "run_with_dependencies"]

_DONE = b"done"
_FAILED = b"failed"
_POLL_SECONDS = 1.0
_STORAGE_RETRY_DELAYS = (0.25, 0.5, 1.0, 2.0)
_T = TypeVar("_T")


def dependency_state_name(job_id: str) -> str:
    """Return the submission-scoped state filename for one Misen job."""
    return f"{job_id}.state"


def _retry_storage(operation: Callable[[], _T]) -> _T:
    """Retry transient workspace I/O with a small bounded backoff."""
    for delay in _STORAGE_RETRY_DELAYS:
        try:
            return operation()
        except FileNotFoundError:
            raise
        except (StorageError, OSError):
            time.sleep(delay)
    return operation()


def _read(workspace: Workspace, submission_id: str, job_id: str) -> bytes:
    return _retry_storage(lambda: workspace.read_job_file(submission_id, dependency_state_name(job_id)))


def _publish(workspace: Workspace, submission_id: str, job_id: str, state: bytes) -> None:
    _retry_storage(lambda: workspace.put_job_file(submission_id, dependency_state_name(job_id), state))


def _await_dependencies(
    workspace: Workspace,
    submission_id: str,
    dependencies: Sequence[tuple[str, str]],
) -> None:
    """Block until every dependency succeeds or one fails."""
    pending = dict(dependencies)
    while pending:
        for dependency_id, label in tuple(pending.items()):
            try:
                state = _read(workspace, submission_id, dependency_id)
            except FileNotFoundError:
                continue
            if state == _FAILED:
                msg = f"Dependency {label} (job_id={dependency_id}) failed."
                raise ExecutionError(msg)
            if state != _DONE:
                msg = f"Dependency {label} (job_id={dependency_id}) published an invalid state marker."
                raise ExecutionError(msg)
            pending.pop(dependency_id)
        if pending:
            time.sleep(_POLL_SECONDS)


def run_with_dependencies(
    execute: Callable[[], None],
    *,
    workspace: Workspace,
    submission_id: str,
    job_id: str,
    dependencies: Sequence[tuple[str, str]],
) -> None:
    """Wait for dependency markers, run once, and publish a terminal marker.

    Remote schedulers without native DAG edges can submit every job eagerly:
    workers consume these small workspace markers before entering user code.
    The markers also make an infrastructure retry idempotent once an earlier
    attempt completed successfully.
    """
    completed = False
    try:
        try:
            previous_state = _read(workspace, submission_id, job_id)
            if previous_state == _DONE:
                return
            if previous_state != _FAILED:
                msg = f"Job {job_id} has an invalid persisted state marker."
                raise ExecutionError(msg)
        except FileNotFoundError:
            pass
        _await_dependencies(workspace, submission_id, dependencies)
        execute()
        completed = True
        _publish(workspace, submission_id, job_id, _DONE)
    except BaseException as exc:
        if completed:
            try:
                if _read(workspace, submission_id, job_id) == _DONE:
                    return
            except (FileNotFoundError, StorageError, OSError):
                pass
        try:
            _publish(workspace, submission_id, job_id, _FAILED)
        except Exception as marker_exc:  # noqa: BLE001 -- retain the original worker failure
            exc.add_note(f"Additionally, publishing the failed dependency marker failed: {marker_exc}")
        raise
