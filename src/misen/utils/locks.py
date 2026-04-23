"""Lock abstractions used by workspace/task/result coordination."""

import logging
import os
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol, Self

import flufl.lock._lockfile as flufl

from misen.exceptions import LockUnavailableError

__all__ = ["LockLike", "LockUnavailableError", "NFSLock"]

logger = logging.getLogger(__name__)

_CLOCK_OFFSET_SAMPLES = 5
_clock_offset_cache: dict[int, timedelta] = {}
_clock_offset_cache_lock = threading.Lock()


def _measure_clock_offset(directory: Path) -> timedelta:
    """Measure ``(filesystem_time - local_time)`` for ``directory``'s mount.

    Creates several short-lived probe files and compares each file's
    server-assigned ``mtime`` against a local-clock midpoint bracketing
    the probe's creation. Returns the median sample as a ``timedelta``.

    On NFS the server stamps ``mtime`` at create-time using its own clock,
    so reading the mtime back gives a snapshot of the server's view of
    "now" -- modulo RPC round-trip, which we dampen by bracketing and
    taking the median over several samples.
    """
    samples: list[int] = []
    for _ in range(_CLOCK_OFFSET_SAMPLES):
        t_before_ns = time.time_ns()
        fd, probe = tempfile.mkstemp(prefix=".misen.clockprobe.", suffix=".tmp", dir=str(directory))
        t_after_ns = time.time_ns()
        try:
            fs_mtime_ns = os.fstat(fd).st_mtime_ns
        finally:
            os.close(fd)
            Path(probe).unlink()
        local_mid_ns = (t_before_ns + t_after_ns) // 2
        samples.append(fs_mtime_ns - local_mid_ns)

    samples.sort()
    median_ns = samples[len(samples) // 2]
    return timedelta(microseconds=median_ns // 1000)


def _get_clock_offset(lockfile: Path) -> timedelta:
    """Return the cached filesystem-vs-local clock offset for ``lockfile``.

    Keyed by the containing directory's device id (``st_dev``), so each
    filesystem is probed at most once per process. Falls back to a zero
    offset with a warning if the probe fails.
    """
    directory = lockfile.parent
    try:
        dev = directory.stat().st_dev
    except OSError as e:
        logger.warning("Cannot key clock offset for %s (%s); using zero offset.", directory, e)
        return timedelta(0)

    with _clock_offset_cache_lock:
        cached = _clock_offset_cache.get(dev)
    if cached is not None:
        return cached

    try:
        offset = _measure_clock_offset(directory)
    except OSError as e:
        logger.warning("Cannot measure clock offset against %s (%s); using zero offset.", directory, e)
        offset = timedelta(0)
    else:
        logger.debug("Measured clock offset for %s: %s.", directory, offset)

    with _clock_offset_cache_lock:
        return _clock_offset_cache.setdefault(dev, offset)


class _ClockOffsetLock(flufl.Lock):
    """``flufl.lock.Lock`` whose internal clock is shifted by a fixed offset.

    Overrides the ``_now()`` hook (flufl.lock 9.1+) so all lock-expiration
    arithmetic happens in the filesystem's time frame. Processes on
    clock-skewed machines then agree on lock validity via shared NFS
    lockfiles.

    The offset is resolved *lazily* on first ``_now()`` call, not at
    construction, so building an ``NFSLock`` on a slow or unreachable
    mount doesn't block. Measurement cost is also amortized by the
    per-filesystem cache in :func:`_get_clock_offset`.
    """

    _clock_offset: timedelta | None

    def __init__(self, lockfile: str, lifetime: timedelta | int) -> None:
        """Initialize lock; clock offset is measured on first use."""
        self._clock_offset = None
        super().__init__(lockfile=lockfile, lifetime=lifetime)

    def _now(self) -> datetime:
        """Return ``datetime.now()`` shifted by the filesystem clock offset."""
        if self._clock_offset is None:
            self._clock_offset = _get_clock_offset(Path(self.lockfile))
        return datetime.now() + self._clock_offset  # noqa: DTZ005


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

    _lock: _ClockOffsetLock
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
        self._lock = _ClockOffsetLock(lockfile=str(lockfile), lifetime=lifetime)

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
