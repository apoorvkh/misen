"""Shared helpers for ``misen`` CLI commands."""


def system_exit_code(exc: SystemExit) -> int:
    """Normalize ``SystemExit.code`` into a stable integer exit code."""
    if exc.code is None:
        return 0
    try:
        return int(exc.code)
    except (TypeError, ValueError):
        return 1
