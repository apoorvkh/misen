"""Lock abstractions used by workspace/task/result coordination."""

import logging
import threading
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Protocol, Self

import flufl.lock._lockfile as flufl

from misen.exceptions import LockUnavailableError

__all__ = ["LockLike", "LockUnavailableError", "NFSLock"]

logger = logging.getLogger(__name__)


class LockLike(Protocol):
    """Protocol describing a lock with context-manager support."""

    def acquire(self, *, blocking: bool = True, timeout: int | None = None) -> None:
        """Acquire lock."""

    def release(self) -> None:
        """Release lock."""

    def context(self, *, blocking: bool = True, timeout: int | None = None) -> AbstractContextManager[Self]:
        """Return context manager that acquires/releases lock."""

    def is_locked(self) -> bool:
        """Return whether lock is currently held."""


class NFSLock:
    """NFS-safe lock implementation backed by ``flufl.lock`` lock files."""

    __slots__ = ("_lock", "_refresh_interval", "_stop", "_thread")

    _lock: flufl.Lock
    _refresh_interval: int | None
    _stop: threading.Event
    _thread: threading.Thread | None

    def __init__(
        self,
        lockfile: Path,
        lifetime: int = 15,
        refresh_interval: int | None = None,
    ) -> None:
        """Initialize the lock.

        Args:
            lockfile: Path to the lock file.
            lifetime: Lock lifetime in seconds for the underlying lock file.
            refresh_interval: Optional refresh interval in seconds.
        """
        self._lock = flufl.Lock(lockfile=str(lockfile), lifetime=lifetime)

        self._refresh_interval = refresh_interval
        if self._refresh_interval is not None:
            self._stop = threading.Event()
            self._thread = None

    def _refresh_loop(self) -> None:
        """Refresh lock lease until stop event is set."""
        try:
            while not self._stop.wait(self._refresh_interval):
                self._lock.refresh()
        except flufl.NotLockedError:
            pass

    def acquire(self, *, blocking: bool = True, timeout: int | None = None) -> None:
        """Acquire lock, optionally waiting up to timeout.

        Args:
            blocking: Whether to block until lock is available.
            timeout: Maximum wait time in seconds when blocking.

        Raises:
            LockUnavailableError: If acquisition fails within ``timeout``.
        """
        timeout = timeout if blocking else 0
        try:
            self._lock.lock(timeout=timeout)
        except flufl.TimeOutError as e:
            msg = f"Could not acquire lock {self._lock.lockfile!r} within {timeout}s."
            raise LockUnavailableError(msg) from e

        if self._refresh_interval is not None:
            if self._thread is not None and self._thread.is_alive():
                return

            self._stop.clear()
            self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
            self._thread.start()

    def release(self) -> None:
        """Release lock and stop refresh thread if active."""
        if self._refresh_interval is not None:
            self._stop.set()
            thread = self._thread
            if thread is not None and thread.is_alive():
                # Give the refresh loop up to one full interval to notice the
                # stop event and exit cleanly. If it doesn't, we log and move
                # on -- ``_stop`` is still set, so the loop will terminate on
                # its next iteration rather than refresh a released lock.
                join_timeout = max(self._refresh_interval, 1)
                thread.join(timeout=join_timeout)
                if thread.is_alive():
                    logger.warning(
                        "Refresh thread for lock %s did not exit within %ss; releasing anyway.",
                        self._lock.lockfile,
                        join_timeout,
                    )
            self._thread = None

        self._lock.unlock()

    @contextmanager
    def context(self, *, blocking: bool = True, timeout: int | None = None) -> Iterator[Self]:
        """Context manager that acquires/releases lock.

        Args:
            blocking: Whether acquisition should block.
            timeout: Optional acquisition timeout in seconds.

        Yields:
            ``self`` while lock is held.
        """
        self.acquire(blocking=blocking, timeout=timeout)
        try:
            yield self
        finally:
            self.release()

    def is_locked(self) -> bool:
        """Return whether underlying lock is held."""
        return self._lock.is_locked
