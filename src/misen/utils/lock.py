import threading
from pathlib import Path

from flufl.lock._lockfile import Interval, Lock, NotLockedError

__all__ = ["NFSLock"]

# TODO: Lock class or Protocol?


class NFSLock(Lock):
    def __init__(
        self,
        lockfile: Path,
        lifetime: int = 15,
        refresh_interval: int | None = None,
        block: bool = True,
        timeout: int | None = None,
    ):
        if block is False:
            timeout = 1  # ?
        super().__init__(lockfile=str(lockfile), lifetime=lifetime, default_timeout=timeout)

        self._refresh_interval = refresh_interval
        if self._refresh_interval is not None:
            self._stop = threading.Event()
            self._thread: threading.Thread | None = None

    def _refresh_loop(self) -> None:
        while not self._stop.wait(self._refresh_interval):
            try:
                self.refresh()
            except NotLockedError:
                break

    def lock(self, timeout: Interval | None = None) -> None:
        super().lock(timeout=timeout)

        if self._refresh_interval is not None:
            if self._thread is not None and self._thread.is_alive():
                return

            self._stop.clear()
            self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
            self._thread.start()

    def unlock(self, *, unconditionally: bool = False) -> None:
        if self._refresh_interval is not None:
            self._stop.set()
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=0.2)
            self._thread = None

        super().unlock(unconditionally=unconditionally)
