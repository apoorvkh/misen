"""User-facing exception rendering for Misen command-line entry points."""

from __future__ import annotations

import os
import sys
import traceback
from typing import TYPE_CHECKING

from misen.exceptions import ErrorCode, MisenError, SubmissionError

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TextIO

__all__ = ["render_cli_error", "run_cli"]

_FALSE_VALUES = frozenset({"", "0", "false", "no", "off"})
_ERROR_LABELS = {
    ErrorCode.MISEN: "error",
    ErrorCode.CLI_USAGE: "usage error",
    ErrorCode.CONFIG: "configuration error",
    ErrorCode.HASH: "hashing error",
    ErrorCode.LOCK_UNAVAILABLE: "lock unavailable",
    ErrorCode.STATUS_QUERY: "job status error",
    ErrorCode.JOB_FAILED: "job failed",
}
_USAGE_ERRORS = frozenset({ErrorCode.CLI_USAGE, ErrorCode.EXPERIMENT_REFERENCE})


def _render_structured_details(error: MisenError, stream: TextIO) -> None:
    """Render typed recovery details that do not belong in ``str(error)``."""
    if not isinstance(error, SubmissionError) or not error.submitted_jobs:
        return
    stream.write("  already submitted jobs:\n")
    for job in error.submitted_jobs:
        details = [
            f"{name}={value}" for name, value in (("job_id", job.job_id), ("log", job.log_path)) if value is not None
        ]
        suffix = f" ({', '.join(details)})" if details else ""
        stream.write(f"    - {job.label}{suffix}\n")


def render_cli_error(
    error: MisenError,
    *,
    stream: TextIO | None = None,
    debug: bool | None = None,
) -> int:
    """Render one expected domain failure and return its process exit code.

    Set ``MISEN_DEBUG=1`` to retain the complete chained traceback while
    diagnosing an expected failure. Unexpected exceptions are deliberately
    not handled here and therefore always keep Python's normal traceback.
    """
    target = stream or sys.stderr
    if debug is None:
        debug = os.getenv("MISEN_DEBUG", "").strip().lower() not in _FALSE_VALUES

    exit_code = 2 if error.code in _USAGE_ERRORS else 1
    if debug:
        traceback.print_exception(error, file=target)
        _render_structured_details(error, target)
        return exit_code

    detail = str(error).strip() or "The operation could not be completed."
    label = _ERROR_LABELS.get(error.code, f"{error.code.value.replace('_', ' ')} error")
    target.write(f"misen: {label}: {detail}\n")
    for note in getattr(error, "__notes__", ()):
        target.write(f"  {note}\n")
    _render_structured_details(error, target)
    return exit_code


def run_cli(action: Callable[[], int | None], *, stream: TextIO | None = None) -> int:
    """Run a CLI action with the shared expected-error policy."""
    target = stream or sys.stderr
    try:
        result = action()
    except MisenError as exc:
        return render_cli_error(exc, stream=target)
    except KeyboardInterrupt:
        target.write("misen: interrupted\n")
        return 130
    return 0 if result is None else result
