"""Scoped clients for independent, persistent SkyPilot namespaces.

The SDK and API server run in child processes. No SkyPilot imports, endpoint
changes, or environment changes are needed in the user's Python process.
An authenticated local socket leases the namespace server until its last
client disconnects, including when a client crashes.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import select
import subprocess
import sys
import threading
import time
import uuid
from contextvars import ContextVar
from multiprocessing.connection import Client
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from misen.exceptions import ConfigError, ExecutionError

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from multiprocessing.connection import Connection

    from misen.executors.skypilot import SkyPilotJob

logger = logging.getLogger(__name__)
_START_TIMEOUT_S = 120
_STOP_TIMEOUT_S = 25
_active_session: ContextVar[ManagedSkyPilotSession | None] = ContextVar("misen_skypilot_session", default=None)
_NAMESPACE_PATTERN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}")


def namespace_directory(namespace: str) -> Path:
    """Resolve persistent state outside project snapshots and disposable caches."""
    if not isinstance(namespace, str) or not _NAMESPACE_PATTERN.fullmatch(namespace) or namespace in {".", ".."}:
        msg = "api_server_namespace must be 1-64 letters, digits, periods, underscores, or hyphens."
        raise ValueError(msg)
    state_home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return (state_home / "misen" / "skypilot" / namespace).expanduser().resolve()


def active_session() -> ManagedSkyPilotSession | None:
    """Return this context's session; jobs retain it for polling in other threads."""
    return _active_session.get()


@contextlib.contextmanager
def managed_session(namespace: str = "default") -> Iterator[ManagedSkyPilotSession]:
    """Nest matching sessions and isolate different namespaces and threads."""
    directory = namespace_directory(namespace)
    existing = active_session()
    if existing is not None and existing.directory == directory:
        existing.check_open()
        yield existing
        return
    session = ManagedSkyPilotSession(directory)
    token = _active_session.set(session)
    error: BaseException | None = None
    try:
        yield session
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            session.close(error)
        finally:
            _active_session.reset(token)


class ManagedSkyPilotSession:
    """A lazy, scoped lease on a namespace's API server and isolated SDK."""

    def __init__(self, directory: Path) -> None:
        """Initialize a session without importing SkyPilot or starting processes."""
        self.directory = directory
        self.jobs: list[SkyPilotJob] = []
        self.closed = False
        self.endpoint: str | None = None
        self.log_path: Path | None = None
        self._connection: Connection | None = None
        self._lock = threading.RLock()
        self.client = _SkyClient(self)

    def check_open(self) -> None:
        """Prevent closed handles from silently creating another server."""
        if self.closed:
            msg = "This SkyPilot job's API session is closed; resubmit inside executor.session() to reattach."
            raise ExecutionError(msg)

    def ensure_started(self) -> None:
        """Connect to a live namespace broker or launch one under its own lock."""
        try:
            from filelock import FileLock
        except ModuleNotFoundError as exc:
            msg = "Isolated API sessions require `misen[skypilot-managed]`."
            raise ConfigError(msg) from exc

        with self._lock:
            self.check_open()
            if self._connection is not None:
                return
            if os.name != "posix":
                msg = "manage_api_server requires Linux or macOS."
                raise ConfigError(msg)
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            with FileLock(self.directory / "session.lock", timeout=_START_TIMEOUT_S):
                descriptor_path = self.directory / "server.json"
                if descriptor_path.exists():
                    try:
                        self._connect(json.loads(descriptor_path.read_text()))
                    except (OSError, EOFError, ExecutionError, ValueError, KeyError):
                        # A previous last client may still be shutting down.
                        # The new broker takes a namespace lifetime lock before
                        # touching any SkyPilot state.
                        pass
                    else:
                        return
                self._start()

    def _connect(self, descriptor: dict[str, Any]) -> None:
        connection = Client(descriptor["address"], family="AF_UNIX", authkey=bytes.fromhex(descriptor["authkey"]))
        try:
            connection.send_bytes(b'{"op":"acquire"}')
            if not connection.poll(_START_TIMEOUT_S):
                msg = "Timed out acquiring a SkyPilot namespace session."
                raise ExecutionError(msg)  # noqa: TRY301
            reply = json.loads(connection.recv_bytes())
            if reply.get("result") != "acquired":
                msg = "SkyPilot namespace server is shutting down; retry the submission."
                raise ExecutionError(msg)  # noqa: TRY301
        except BaseException:
            connection.close()
            raise
        self._connection = connection
        self.endpoint = descriptor["endpoint"]
        self.log_path = Path(descriptor["log_path"])

    def _start(self) -> None:
        identity_path = self.directory / "identity"
        if not identity_path.exists():
            identity_path.touch(mode=0o600, exist_ok=False)
            identity_path.write_text(uuid.uuid4().hex[:8])
        identity = identity_path.read_text().strip()
        if not re.fullmatch(r"[a-f0-9]{8}", identity):
            msg = f"Invalid SkyPilot namespace identity in {identity_path}."
            raise ConfigError(msg)
        config_path = self.directory / "config.yaml"
        if not config_path.exists():
            config_path.touch(mode=0o600, exist_ok=False)
            config_path.write_text('{"jobs": {"controller": {"consolidation_mode": false}}}\n')
        self.log_path = self.directory / f"server-{time.time_ns()}.log"
        self.log_path.touch(mode=0o600, exist_ok=False)
        env = _isolated_environment(self.directory, identity, config_path)
        process: subprocess.Popen[bytes] | None = None
        try:
            with self.log_path.open("wb") as log:
                process = subprocess.Popen(  # noqa: S603
                    [sys.executable, "-m", "misen.utils.skypilot_broker", str(self.directory), str(self.log_path)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=log,
                    start_new_session=True,
                    env=env,
                    cwd=self.directory,
                )
            if process.stdout is None:
                msg = "SkyPilot broker did not provide a startup pipe."
                raise ExecutionError(msg)
            readable, _, _ = select.select([process.stdout], [], [], _START_TIMEOUT_S)
            if not readable:
                msg = f"Timed out starting Misen's SkyPilot server; see {self.log_path}."
                raise ExecutionError(msg)
            line = process.stdout.readline()
            if not line:
                msg = f"SkyPilot broker exited during startup; see {self.log_path}."
                raise ExecutionError(msg)
            descriptor = json.loads(line)
            if "error" in descriptor:
                raise ConfigError(descriptor["error"])
            self._connect(descriptor)
            logger.info("Connected to Misen's SkyPilot namespace at %s (state=%s).", self.endpoint, self.directory)
        finally:
            if process is not None:
                # EOF on the startup pipe before a socket lease is acquired
                # also handles the creator being killed during startup.
                if process.stdin is not None:
                    process.stdin.close()
                if process.stdout is not None:
                    process.stdout.close()
                threading.Thread(target=process.wait, daemon=True, name="misen-skypilot-reaper").start()

    def call(self, operation: str, **arguments: Any) -> Any:
        """Call the isolated SDK over authenticated JSON, never Python pickle."""
        with self._lock:
            self.ensure_started()
            return self._exchange({"op": operation, "args": arguments})

    def _exchange(self, message: dict[str, Any], *, timeout: float | None = None) -> Any:
        connection = self._connection
        if connection is None:
            msg = "SkyPilot session has no broker connection."
            raise ExecutionError(msg)
        try:
            connection.send_bytes(json.dumps(message).encode())
            if timeout is not None and not connection.poll(timeout):
                msg = f"SkyPilot server shutdown timed out; see {self.log_path}."
                raise ExecutionError(msg)
            reply = json.loads(connection.recv_bytes())
        except (OSError, EOFError) as exc:
            msg = f"Lost Misen's SkyPilot server connection; see {self.log_path}."
            raise ExecutionError(msg) from exc
        if "error" in reply:
            raise ExecutionError(reply["error"])
        return _decode_result(reply["result"])

    def check(self, infra_list: Sequence[str], *, verbose: bool = False) -> Any:
        """Check credentials and enable the selected clouds in this namespace.

        Run once before using a fresh namespace, and again after changing cloud
        credentials. This does not provision workers or modify other namespaces.
        """
        if (
            isinstance(infra_list, str)
            or not infra_list
            or any(not isinstance(item, str) or not item for item in infra_list)
        ):
            msg = "infra_list must be a nonempty sequence of infrastructure names, e.g. ['aws']."
            raise ValueError(msg)
        return self.call("check", infra_list=list(infra_list), verbose=verbose)

    def pool_apply(self, pool_name: str, config: str | Path) -> None:
        """Create/update a pool in this namespace, waiting for SkyPilot's result."""
        self.call("pool_apply", pool_name=pool_name, config=str(Path(config).expanduser().resolve()))

    def pool_down(self, pool_name: str) -> None:
        """Terminate only the named pool in this namespace."""
        self.call("pool_down", pool_name=pool_name)

    def pool_status(self) -> Any:
        """Return this namespace's pool statuses."""
        return self.call("pool_status")

    def close(self, original_error: BaseException | None = None) -> None:
        """Persist accepted launches and release the lease; the last release stops the server."""
        failures: list[Exception] = []
        with self._lock:
            if self.closed:
                return
            try:
                if self._connection is not None:
                    for job in self.jobs:
                        if job.managed_job_id is None and job._terminal_state is None:  # noqa: SLF001
                            try:
                                job._resolve_managed_job_id(self.client)  # noqa: SLF001
                            except Exception as exc:  # noqa: BLE001
                                failures.append(exc)
                    try:
                        self._exchange({"op": "release"}, timeout=_STOP_TIMEOUT_S)
                    except Exception as exc:  # noqa: BLE001
                        failures.append(exc)
            finally:
                self.closed = True
                if self._connection is not None:
                    self._connection.close()
            if failures:
                msg = "SkyPilot session cleanup failed: " + "; ".join(str(exc) for exc in failures)
                if original_error is not None:
                    original_error.add_note(msg)
                else:
                    raise ExecutionError(msg) from failures[0]


def _isolated_environment(directory: Path, identity: str, config_path: Path) -> dict[str, str]:
    """Keep credentials available while replacing only the child's SkyPilot control settings."""
    # Provider credentials (AWS_*, GOOGLE_*, AZURE_*, etc.) remain available;
    # ambient SkyPilot control/auth flags belong to the ordinary namespace.
    env = {key: value for key, value in os.environ.items() if not key.startswith(("SKYPILOT_", "SKY_", "IS_SKYPILOT_"))}
    env.update(
        SKY_RUNTIME_DIR=str(directory),
        SKYPILOT_USER_ID=identity,
        SKYPILOT_GLOBAL_CONFIG=str(config_path),
        SKYPILOT_PROJECT_CONFIG=str(config_path),
        SKYPILOT_API_COOKIE_FILE=str(directory / "cookies.txt"),
    )
    return env


def _decode_result(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"__tuple__"}:
            return tuple(_decode_result(item) for item in value["__tuple__"])
        return {key: _decode_result(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_result(item) for item in value]
    return value


class _Resources:
    def __init__(self, session: ManagedSkyPilotSession, **kwargs: Any) -> None:
        self.session = session
        self.options = kwargs

    def validate(self) -> None:
        self.session.call("validate_resources", options=self.options)


class _Task:
    def __init__(self, **kwargs: Any) -> None:
        self.options = kwargs


class _SkyJobs:
    def __init__(self, session: ManagedSkyPilotSession) -> None:
        self.session = session

    def launch(self, task: _Task, **kwargs: Any) -> str:
        options = dict(task.options)
        resources = options["resources"]
        if isinstance(resources, _Resources):
            resources = [resources]
        options["resources"] = [resource.options for resource in resources]
        return self.session.call("launch", task=options, **kwargs)

    def queue_v2(self, **kwargs: Any) -> str:
        return self.session.call("queue_v2", **kwargs)

    def cancel(self, **kwargs: Any) -> str:
        return self.session.call("cancel", **kwargs)


class _SkyClient:
    """The small SDK surface used by SkyPilotExecutor, without importing sky."""

    Task = _Task

    def __init__(self, session: ManagedSkyPilotSession) -> None:
        self.session = session
        self.jobs = _SkyJobs(session)
        self.server = SimpleNamespace(common=SimpleNamespace(is_api_server_local=lambda: True))

    def Resources(self, **kwargs: Any) -> _Resources:  # noqa: N802
        return _Resources(self.session, **kwargs)

    def get(self, request_id: str) -> Any:
        return self.session.call("get", request_id=request_id)

    def api_status(self, **kwargs: Any) -> Any:
        return self.session.call("api_status", **kwargs)

    def api_info(self) -> Any:
        return self.session.call("api_info")
