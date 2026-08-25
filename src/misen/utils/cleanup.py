"""Cleanup helpers that preserve an active exception."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@contextmanager
def _cleanup_on_exit(cleanup: Callable[[], None], label: str, *, on_error: bool = False) -> Iterator[None]:
    """Run cleanup without letting its failure replace an active one."""
    primary: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        primary = exc
        raise
    finally:
        if not on_error or primary is not None:
            try:
                cleanup()
            except BaseException as exc:
                if primary is None:
                    raise
                primary.add_note(f"Additionally, {label} failed: {type(exc).__name__}: {exc}")
