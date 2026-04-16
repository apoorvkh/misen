"""Lock abstractions used by workspace/task/result coordination."""

import threading
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Protocol, Self

import flufl.lock._lockfile as flufl

__all__ = ["LockLike", "LockUnavailableError", "NFSLock"]


class LockUnavailableError(TimeoutError):
    """Raised when lock acquisition exceeds timeout."""

    # TODO: raise this in NFSLock?


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
        """
        timeout = timeout if blocking else 0
        self._lock.lock(timeout=timeout)

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
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=0.2)
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
