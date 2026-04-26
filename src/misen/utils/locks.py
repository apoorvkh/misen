"""Lock abstractions used by workspace/task/result coordination."""

import logging
import os
import socket
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, suppress
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Self, cast, runtime_checkable

import flufl.lock._lockfile as flufl
import msgspec
import obstore as _obs
from obstore.exceptions import AlreadyExistsError, PreconditionError

from misen.exceptions import LockUnavailableError

if TYPE_CHECKING:
    from obstore import PutMode, PutResult

__all__ = ["LockLike", "LockUnavailableError", "NFSLock", "ObjectStoreLock"]

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


@runtime_checkable
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
        # flufl.lock's stale-break check runs only after its timeout check,
        # so ``timeout=0`` never breaks expired lockfiles. Pass ``timeout=1``
        # for non-blocking acquires so the break-check path fires at least
        # once; on real contention this costs at most ~1s of waiting.
        flufl_timeout = timeout if blocking else 1
        try:
            self._lock.lock(timeout=flufl_timeout)
        except flufl.TimeOutError as e:
            reported = timeout if blocking else 0
            msg = f"Could not acquire lock {self._lock.lockfile!r} within {reported}s."
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


def _owner_id() -> str:
    """Return a unique-per-acquisition owner identifier for an object-store lock."""
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"


def _now_ms() -> int:
    """Return current wall-clock time in milliseconds since the epoch."""
    return time.time_ns() // 1_000_000


class ObjectStoreLock(LockLike):
    """Distributed lock backed by object-store conditional writes.

    Implements :class:`LockLike` on top of obstore's conditional ``put``
    semantics:

    - ``put-if-absent`` (``mode="create"``) for fresh acquisition.
    - ``put-if-match`` (``UpdateVersion``) for refresh and expired-lease
      takeover.
    - ``put-if-match`` with an expired payload for release.

    The lock state is a single JSON object at ``key`` carrying owner
    identity and an absolute-millisecond expiry. A holder process runs a
    refresh thread that periodically extends the expiry via put-if-match;
    if the conditional update fails, the holder has lost the lock and the
    refresh thread terminates.

    S3, GCS, and Azure all implement the conditional-write primitives this
    lock relies on, so the same code path coordinates correctly across
    machines for every supported provider.
    """

    __slots__ = (
        "_key",
        "_lifetime_ms",
        "_owner",
        "_refresh_interval",
        "_stop",
        "_store",
        "_thread",
        "_token",
    )

    def __init__(
        self,
        store: Any,
        key: str,
        lifetime: int = 30,
        refresh_interval: int | None = 20,
    ) -> None:
        """Initialize the lock.

        Args:
            store: Underlying obstore store handle.
            key: Object-store key (path) used for the lock object.
            lifetime: Lease lifetime in seconds. Holders must refresh before
                this elapses.
            refresh_interval: Seconds between refresh attempts. ``None``
                disables refresh; the lock then expires when its lifetime
                elapses unless explicitly released.
        """
        self._store = store
        self._key = key
        self._lifetime_ms = lifetime * 1000
        self._refresh_interval = refresh_interval
        self._owner = _owner_id()
        self._token: PutResult | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _payload(self, *, expiry_ms: int | None = None) -> bytes:
        """Return JSON-encoded lock metadata for the current lease."""
        if expiry_ms is None:
            expiry_ms = _now_ms() + self._lifetime_ms
        return msgspec.json.encode({"owner": self._owner, "expiry_ms": expiry_ms})

    @staticmethod
    def _update_mode(token: "PutResult") -> "PutMode":
        """Return obstore ``put`` ``mode`` argument for put-if-match.

        obstore expects an :class:`obstore.UpdateVersion` ``TypedDict`` here
        (only ``e_tag`` / ``version`` keys present, omitted when ``None``).
        """
        mode: dict[str, str] = {}
        e_tag = token.get("e_tag")
        version = token.get("version")
        if e_tag is not None:
            mode["e_tag"] = e_tag
        if version is not None:
            mode["version"] = version
        if not mode:
            msg = "Object store did not return an update token for conditional writes."
            raise LockUnavailableError(msg)
        return cast("PutMode", mode)

    def _try_create(self) -> bool:
        """Attempt fresh put-if-absent. Returns True if we now hold the lock."""
        try:
            token = _obs.put(self._store, self._key, self._payload(), mode="create")
        except AlreadyExistsError:
            return False
        self._update_mode(token)
        self._token = token
        return True

    def _try_takeover(self) -> bool:
        """Attempt to take over an expired lease. Returns True on success."""
        try:
            existing = _obs.get(self._store, self._key)
            existing_meta = existing.meta
            existing_body = bytes(existing.bytes())
        except FileNotFoundError:
            return self._try_create()

        try:
            existing_state = msgspec.json.decode(existing_body)
        except msgspec.DecodeError:
            existing_state = {"expiry_ms": 0}

        if existing_state.get("expiry_ms", 0) > _now_ms():
            return False

        candidate = cast("PutResult", {"e_tag": existing_meta.get("e_tag"), "version": existing_meta.get("version")})
        mode = self._update_mode(candidate)
        try:
            self._token = _obs.put(self._store, self._key, self._payload(), mode=mode)
        except (PreconditionError, AlreadyExistsError):
            return False
        return True

    def acquire(self, *, blocking: bool = True, timeout: int | None = None) -> None:
        """Acquire the lock, optionally waiting up to ``timeout`` seconds.

        Args:
            blocking: Whether to block until the lock can be acquired.
            timeout: Maximum wait time in seconds when blocking. ``None``
                means wait indefinitely.

        Raises:
            LockUnavailableError: If the lock cannot be acquired within
                the requested time budget.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        backoff = 0.1
        while True:
            if self._try_create() or self._try_takeover():
                self._start_refresh()
                return
            if not blocking:
                msg = f"Could not acquire lock {self._key!r}."
                raise LockUnavailableError(msg)
            if deadline is not None and time.monotonic() >= deadline:
                msg = f"Could not acquire lock {self._key!r} within {timeout}s."
                raise LockUnavailableError(msg)
            time.sleep(backoff)
            backoff = min(backoff * 2, 2.0)

    def _start_refresh(self) -> None:
        """Start the background refresh thread if refresh is enabled."""
        if self._refresh_interval is None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._thread.start()

    def _refresh_loop(self) -> None:
        """Refresh the lease until release or until refresh fails."""
        while not self._stop.wait(self._refresh_interval):
            token = self._token
            if token is None:
                return
            try:
                self._token = _obs.put(self._store, self._key, self._payload(), mode=self._update_mode(token))
            except (PreconditionError, AlreadyExistsError, FileNotFoundError):
                logger.warning("Lost lease on lock %s during refresh.", self._key)
                self._token = None
                return

    def release(self) -> None:
        """Release the lock. Idempotent."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            join_timeout = max(self._refresh_interval or 1, 1)
            thread.join(timeout=join_timeout)
        self._thread = None
        token = self._token
        self._token = None
        if token is None:
            return
        with suppress(AlreadyExistsError, FileNotFoundError, PreconditionError):
            _obs.put(self._store, self._key, self._payload(expiry_ms=0), mode=self._update_mode(token))

    def is_locked(self) -> bool:
        """Return whether this lock instance currently holds the lease."""
        return self._token is not None

    @contextmanager
    def context(self, *, blocking: bool = True, timeout: int | None = None) -> Iterator[Self]:
        """Context manager that acquires/releases the lock."""
        self.acquire(blocking=blocking, timeout=timeout)
        try:
            yield self
        finally:
            self.release()
