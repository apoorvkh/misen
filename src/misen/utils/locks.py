import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import ContextManager, Protocol

import flufl.lock._lockfile as flufl
from typing_extensions import Self

__all__ = ["LockUnavailableError", "LockLike", "NFSLock"]


class LockUnavailableError(TimeoutError): ...  # TODO: raise this in NFSLock?


class LockLike(Protocol):
    def acquire(self, blocking: bool = True, timeout: int | None = None) -> None: ...
    def release(self) -> None: ...
    def context(self, blocking: bool = True, timeout: int | None = None) -> ContextManager[Self]: ...
    def is_locked(self) -> bool: ...


class NFSLock:
    def __init__(
        self,
        lockfile: Path,
        lifetime: int = 15,
        refresh_interval: int | None = None,
    ):
        self._lock = flufl.Lock(lockfile=str(lockfile), lifetime=lifetime)

        self._refresh_interval = refresh_interval
        if self._refresh_interval is not None:
            self._stop = threading.Event()
            self._thread: threading.Thread | None = None

    def _refresh_loop(self) -> None:
        while not self._stop.wait(self._refresh_interval):
            try:
                self._lock.refresh()
            except flufl.NotLockedError:
                break

    def acquire(self, blocking: bool = True, timeout: int | None = None) -> None:
        timeout = timeout if blocking else 0
        self._lock.lock(timeout=timeout)

        if self._refresh_interval is not None:
            if self._thread is not None and self._thread.is_alive():
                return

            self._stop.clear()
            self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
            self._thread.start()

    def release(self) -> None:
        if self._refresh_interval is not None:
            self._stop.set()
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=0.2)
            self._thread = None

        self._lock.unlock()

    @contextmanager
    def context(self, blocking: bool = True, timeout: int | None = None) -> Iterator[Self]:
        self.acquire(blocking=blocking, timeout=timeout)
        try:
            yield self
        finally:
            self.release()

    def is_locked(self) -> bool:
        return self._lock.is_locked
