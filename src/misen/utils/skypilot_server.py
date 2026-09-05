"""Own a local SkyPilot API server for one foreground execution session.

SkyPilot 0.13 has a shared local state directory. An owned server therefore
requires exclusive local use, including the SDK's server-creation lock. A
small guardian owns that lock and the server process tree, and watches a pipe
from Misen so even an abruptly killed client leaves no local server behind.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import select
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

from misen.exceptions import ConfigError, ExecutionError

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import FrameType

    from misen.executors.skypilot import SkyPilotJob

logger = logging.getLogger(__name__)
_START_TIMEOUT_S = 90
_STOP_TIMEOUT_S = 20
_TERM_GRACE_S = 5
_SESSION_LOCK = threading.RLock()
_active_session: ManagedSkyPilotSession | None = None


def active_session() -> ManagedSkyPilotSession | None:
    """Return the process-wide SDK session, if one is open."""
    return _active_session


@contextlib.contextmanager
def managed_session() -> Iterator[None]:
    """Nest sessions on the owning thread without changing SDK configuration."""
    global _active_session  # noqa: PLW0603
    with _SESSION_LOCK:
        existing = _active_session
        if existing is not None and existing.owner_thread != threading.get_ident():
            msg = "A managed SkyPilot session is already open in another thread."
            raise ConfigError(msg)
        if existing is None:
            _active_session = ManagedSkyPilotSession()
        session = _active_session
    if existing is not None:
        yield
        return
    error: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            if session is not None:
                session.close(error)
        finally:
            with _SESSION_LOCK:
                _active_session = None


class ManagedSkyPilotSession:
    """Lazily start one owned server and retain accepted launch requests."""

    def __init__(self) -> None:
        """Create a session without importing SkyPilot or starting processes."""
        self.owner_thread = threading.get_ident()
        self.jobs: list[SkyPilotJob] = []
        self.closed = False
        self.process: subprocess.Popen[bytes] | None = None
        self.log_path = Path.home() / ".sky" / f"misen-api-server-{os.getpid()}-{time.time_ns()}.log"
        self._sky: Any = None
        self._original_start: Any = None
        self._lock = threading.RLock()

    def check_open(self) -> None:
        """Prevent stale job handles from restarting an unowned SDK server."""
        if self.closed:
            msg = "This SkyPilot job's API session is closed; resubmit inside executor.session() to reattach."
            raise ExecutionError(msg)

    def ensure_started(self, sky: Any) -> None:
        """Start a server once, and prohibit SDK automatic daemon creation."""
        with self._lock:
            self.check_open()
            if self.process is not None:
                if self.process.poll() is not None:
                    msg = f"Misen's SkyPilot API server exited unexpectedly; see {self.log_path}."
                    raise ExecutionError(msg)
                return
            if os.name != "posix":
                msg = "manage_api_server requires Linux or macOS."
                raise ConfigError(msg)
            common = sky.server.common
            if not common.is_api_server_local():
                msg = "manage_api_server requires a local SkyPilot endpoint; unset it for a remote API server."
                raise ConfigError(msg)
            if sky.skypilot_config.get_nested(("jobs", "controller", "consolidation_mode"), default_value=False):
                msg = (
                    "manage_api_server requires jobs.controller.consolidation_mode=false in SkyPilot config; "
                    "the remote jobs/pool controller must outlive the local API server."
                )
                raise ConfigError(msg)
            endpoint = common.get_server_url()
            self._sky = sky
            self._original_start = common.check_server_healthy_or_start_fn
            # The SDK wrappers look up this function at call time. Checking
            # health only also prevents a mid-session crash from spawning a
            # detached replacement (or deadlocking on the guardian's lock).
            common.check_server_healthy_or_start_fn = self._check_server
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                self.log_path.touch(mode=0o600, exist_ok=False)
                with self.log_path.open("wb") as log:
                    self.process = subprocess.Popen(  # noqa: S603
                        [sys.executable, "-m", "misen.utils.skypilot_server", endpoint],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=log,
                        start_new_session=True,
                    )
                self._wait_ready(endpoint)
            except BaseException as exc:
                try:
                    self._stop()
                except Exception as cleanup_error:  # noqa: BLE001
                    exc.add_note(f"SkyPilot server cleanup also failed: {cleanup_error}")
                raise
            logger.info("Started Misen-owned SkyPilot API server (%s; log=%s).", endpoint, self.log_path)

    def _check_server(self, *_args: Any, **_kwargs: Any) -> None:
        """Replace SDK autostart with a health check during owned sessions."""
        self.check_open()
        self._sky.server.common.check_server_healthy()

    def _wait_ready(self, endpoint: str) -> None:
        process = self.process
        if process is None or process.stdout is None:
            msg = "SkyPilot server guardian was not started."
            raise ExecutionError(msg)
        deadline = time.monotonic() + _START_TIMEOUT_S
        launched = False
        while time.monotonic() < deadline:
            if process.poll() is not None:
                msg = (
                    f"Could not start an exclusive SkyPilot API server; see {self.log_path}. "
                    "Another local SkyPilot server may already be running. Stop it explicitly before retrying."
                )
                raise ConfigError(msg)
            if not launched:
                readable, _, _ = select.select([process.stdout], [], [], 0.1)
                if not readable:
                    continue
                try:
                    launched = isinstance(json.loads(process.stdout.readline()).get("pid"), int)
                except ValueError:
                    launched = False
                continue
            try:
                with urlopen(f"{endpoint}/api/health", timeout=1) as response:  # noqa: S310
                    healthy = json.load(response).get("status") == "healthy"
            except (URLError, OSError, ValueError):
                healthy = False
            if healthy and process.poll() is None:
                return
            time.sleep(0.1)
        msg = f"Timed out starting Misen's SkyPilot API server; see {self.log_path}."
        raise ExecutionError(msg)

    def close(self, original_error: BaseException | None = None) -> None:
        """Drain accepted launches, then stop the server even on error."""
        failures: list[Exception] = []
        try:
            if self.process is not None and self.process.poll() is None:
                for job in self.jobs:
                    if job.managed_job_id is None and job._terminal_state is None:  # noqa: SLF001
                        try:
                            job._resolve_managed_job_id(self._sky)  # noqa: SLF001
                        except Exception as exc:  # noqa: BLE001
                            failures.append(exc)
        finally:
            self.closed = True
            try:
                self._stop()
            except Exception as exc:  # noqa: BLE001
                failures.append(exc)
        if failures:
            msg = "SkyPilot session cleanup failed: " + "; ".join(str(exc) for exc in failures)
            if original_error is not None:
                original_error.add_note(msg)
            else:
                raise ExecutionError(msg) from failures[0]

    def _stop(self) -> None:
        try:
            if self.process is not None:
                if self.process.stdin is not None:
                    self.process.stdin.close()
                try:
                    self.process.wait(timeout=_STOP_TIMEOUT_S)
                except subprocess.TimeoutExpired as exc:
                    msg = f"SkyPilot server cleanup did not finish; see {self.log_path}."
                    raise ExecutionError(msg) from exc
                finally:
                    if self.process.stdout is not None:
                        self.process.stdout.close()
                logger.info("Stopped Misen-owned SkyPilot API server.")
        finally:
            if self._original_start is not None:
                self._sky.server.common.check_server_healthy_or_start_fn = self._original_start
                self._original_start = None


def _stop_tree(process: subprocess.Popen[bytes]) -> None:
    """Reap only the child tree and process group started by this guardian."""
    import psutil

    descendants = []
    with contextlib.suppress(psutil.NoSuchProcess):
        descendants = psutil.Process(process.pid).children(recursive=True)
    # Give the server supervisor a chance to stop its request workers before
    # terminating the remaining group; signalling the workers simultaneously
    # can break its shutdown queue before it drains.
    process.terminate()
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=_TERM_GRACE_S)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    for child in descendants:
        with contextlib.suppress(psutil.NoSuchProcess):
            child.kill()
    process.wait(timeout=_TERM_GRACE_S)
    psutil.wait_procs(descendants, timeout=_TERM_GRACE_S)


def _guard_server(endpoint: str) -> None:
    """Run in a helper process; pipe EOF also covers client SIGKILL/crashes."""
    import filelock
    import psutil
    from sky.skylet import constants

    def exit_on_signal(_signum: int, _frame: FrameType | None) -> None:
        raise SystemExit(1)

    signal.signal(signal.SIGTERM, exit_on_signal)
    signal.signal(signal.SIGINT, exit_on_signal)
    lock = filelock.FileLock(Path(constants.API_SERVER_CREATION_LOCK_PATH).expanduser())
    with lock.acquire(timeout=0):
        for process in psutil.process_iter(["cmdline", "uids"]):
            args = process.info["cmdline"] or []
            if "sky.server.server" in args:
                msg = "Another local SkyPilot API server is already running."
                raise RuntimeError(msg)
        port = urlsplit(endpoint).port or 46580
        # A bound but not-yet-healthy server must never be mistaken for ours.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
        env = dict(os.environ)
        env[constants.ENV_VAR_IS_SKYPILOT_SERVER] = "true"
        _supervise_server([sys.executable, "-m", "sky.server.server", "--host=127.0.0.1", f"--port={port}"], env=env)


def _supervise_server(command: list[str], *, env: dict[str, str]) -> None:
    """Keep the server tree alive only while the owning client's pipe is open."""
    child = subprocess.Popen(  # noqa: S603
        command,
        stdin=subprocess.DEVNULL,
        stdout=sys.stderr,
        stderr=sys.stderr,
        start_new_session=True,
        env=env,
    )
    try:
        sys.stdout.write(json.dumps({"pid": child.pid}) + "\n")
        sys.stdout.flush()
        while child.poll() is None:
            readable, _, _ = select.select([sys.stdin], [], [], 0.2)
            if readable and not os.read(sys.stdin.fileno(), 1):
                return
        msg = f"SkyPilot API server exited with status {child.returncode}."
        raise RuntimeError(msg)
    finally:
        _stop_tree(child)


if __name__ == "__main__":
    _guard_server(sys.argv[1])
