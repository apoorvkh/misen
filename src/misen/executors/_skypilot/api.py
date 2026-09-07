"""Scoped SkyPilot SDK proxy, broker, and API-server lifecycle."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import queue as queue_module
import re
import runpy
import secrets
import select
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextvars import ContextVar
from enum import Enum
from multiprocessing.connection import Client, Listener
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from urllib.error import URLError
from urllib.request import urlopen

from misen.exceptions import (
    ConfigError,
    ExecutionError,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from multiprocessing.connection import Connection
    from typing import TextIO


if TYPE_CHECKING:
    from .jobs import SkyPilotJob

logger = logging.getLogger("misen.executors.skypilot")

# Isolated API sessions and JSON client
# ----------------------------------------------------------------------------

_START_TIMEOUT_S = 120


_STOP_TIMEOUT_S = 25


_CALL_TIMEOUT_S = 120


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

    def __init__(self, directory: Path, *, _call_timeout_s: float = _CALL_TIMEOUT_S) -> None:
        """Initialize a session without importing SkyPilot or starting processes."""
        if (
            isinstance(_call_timeout_s, bool)
            or not isinstance(_call_timeout_s, (int, float))
            or not math.isfinite(_call_timeout_s)
            or _call_timeout_s <= 0
        ):
            msg = "SkyPilot broker call timeout must be finite and positive."
            raise ValueError(msg)
        self.directory = directory
        self.jobs: list[SkyPilotJob] = []
        self.closed = False
        self.endpoint: str | None = None
        self.log_path: Path | None = None
        self._connection: Connection | None = None
        self._call_timeout_s = _call_timeout_s
        self._connection_error: str | None = None
        self._closing = False
        self._closing_thread: int | None = None
        self._close_deadline: float | None = None
        self._lock = threading.RLock()
        self.client = _SkyClient(self)

    def check_open(self) -> None:
        """Prevent closed handles from silently creating another server."""
        if self.closed:
            msg = "This SkyPilot job's API session is closed; resubmit inside executor.session() to reattach."
            raise ExecutionError(msg)
        if self._connection_error is not None:
            raise ExecutionError(self._connection_error)
        if self._closing and self._closing_thread != threading.get_ident():
            msg = "SkyPilot session is closing and cannot accept new calls."
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
            if self._closing:
                msg = "SkyPilot session is closing and cannot start another broker."
                raise ExecutionError(msg)
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
            self.check_open()
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
                    [
                        sys.executable,
                        "-m",
                        "misen.executors.skypilot",
                        "--broker",
                        str(self.directory),
                        str(self.log_path),
                    ],
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
            return self._exchange({"op": operation, "args": arguments}, timeout=self._call_timeout_s)

    def _invalidate_connection(self, message: str) -> None:
        """Discard an uncertain RPC stream; never consume its eventual stale reply."""
        connection, self._connection = self._connection, None
        self._connection_error = message
        if connection is not None:
            # Closing an fd in another thread need not wake a blocked read.
            # Shutdown the duplicated Unix socket first to interrupt both RPC
            # directions; the original connection still owns its descriptor.
            with contextlib.suppress(OSError, TypeError, ValueError):
                with socket.socket(fileno=os.dup(connection.fileno())) as stream:
                    stream.shutdown(socket.SHUT_RDWR)
            connection.close()

    def _exchange(self, message: dict[str, Any], *, timeout: float | None = None) -> Any:
        connection = self._connection
        if connection is None:
            msg = "SkyPilot session has no broker connection."
            raise ExecutionError(msg)
        timeout = self._call_timeout_s if timeout is None else timeout
        if self._close_deadline is not None:
            timeout = min(timeout, max(0.0, self._close_deadline - time.monotonic()))
        completed = threading.Event()
        received: list[bytes | BaseException] = []

        def exchange() -> None:
            try:
                connection.send_bytes(json.dumps(message).encode())
                received.append(connection.recv_bytes())
            except BaseException as exc:  # noqa: BLE001 -- forward transport failures to the calling thread
                received.append(exc)
            finally:
                completed.set()

        threading.Thread(target=exchange, daemon=True, name="misen-skypilot-rpc").start()
        try:
            if not completed.wait(timeout):
                operation = "server shutdown" if message.get("op") == "release" else f"broker {message.get('op')} call"
                msg = f"SkyPilot {operation} timed out; connection discarded; see {self.log_path}."
                self._invalidate_connection(msg)
                raise ExecutionError(msg)
            response = received[0]
            if isinstance(response, BaseException):
                raise response
            reply = json.loads(response)
        except (OSError, EOFError) as exc:
            msg = f"Lost Misen's SkyPilot server connection; see {self.log_path}."
            self._invalidate_connection(msg)
            raise ExecutionError(msg) from exc
        except (ValueError, UnicodeError) as exc:
            msg = f"Invalid reply from Misen's SkyPilot server; see {self.log_path}."
            self._invalidate_connection(msg)
            raise ExecutionError(msg) from exc
        except (KeyboardInterrupt, SystemExit):
            self._invalidate_connection("SkyPilot broker call was interrupted; its connection was discarded.")
            raise
        if not isinstance(reply, dict) or ("error" not in reply and "result" not in reply):
            msg = f"Invalid reply from Misen's SkyPilot server; see {self.log_path}."
            self._invalidate_connection(msg)
            raise ExecutionError(msg)
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
        """Drain/release within one deadline, discarding blocked connections safely."""
        failures: list[Exception] = []
        if self.closed:
            return
        if not self._closing:
            self._closing = True
            self._closing_thread = threading.get_ident()
            self._close_deadline = time.monotonic() + _STOP_TIMEOUT_S
        acquired = self._lock.acquire(timeout=_STOP_TIMEOUT_S)
        if not acquired:
            msg = "SkyPilot session cleanup timed out waiting for an active call; broker connection discarded."
            self.closed = True
            self._invalidate_connection(msg)
            if original_error is not None:
                original_error.add_note(msg)
                return
            raise ExecutionError(msg)
        try:
            if self.closed:
                return
            try:
                if self._connection is not None:
                    for job in self.jobs:
                        if self._connection is None:
                            break
                        if job.managed_job_id is None and job._terminal_state is None:  # noqa: SLF001
                            try:
                                job._resolve_managed_job_id(self.client)  # noqa: SLF001
                            except Exception as exc:  # noqa: BLE001
                                failures.append(exc)
                    if self._connection is not None:
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
        finally:
            self._lock.release()


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


def _task_options(task: _Task) -> dict[str, Any]:
    """Serialize resource candidates identically for managed and cluster jobs."""
    options = dict(task.options)
    resources = options.get("resources")
    if resources is not None:
        if isinstance(resources, _Resources):
            resources = [resources]
        options["resources"] = [dict(resource.options) for resource in resources]
    return options


def _cluster_arguments(cluster_name: str, job_ids: Sequence[int] | None = None) -> dict[str, Any]:
    """Require one explicit cluster and, when provided, explicit positive job IDs."""
    if (
        not isinstance(cluster_name, str)
        or not cluster_name
        or cluster_name.strip() != cluster_name
        or any(char in cluster_name for char in "*?[]\x00\n\r")
    ):
        msg = "A single explicit SkyPilot cluster name is required."
        raise ValueError(msg)
    arguments: dict[str, Any] = {"cluster_name": cluster_name}
    if job_ids is not None:
        if (
            not isinstance(job_ids, (list, tuple))
            or not job_ids
            or any(type(job_id) is not int or job_id < 1 for job_id in job_ids)
        ):
            msg = "Cluster operations require a nonempty sequence of positive job IDs."
            raise ValueError(msg)
        arguments["job_ids"] = list(job_ids)
    return arguments


class _SkyJobs:
    def __init__(self, session: ManagedSkyPilotSession) -> None:
        self.session = session

    def launch(self, task: _Task, **kwargs: Any) -> str:
        return self.session.call("launch", task=_task_options(task), **kwargs)

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

    def exec(self, task: _Task, cluster_name: str) -> str:
        return self.session.call("cluster_exec", task=_task_options(task), **_cluster_arguments(cluster_name))

    def job_status(self, cluster_name: str, job_ids: Sequence[int]) -> str:
        return self.session.call("cluster_job_status", **_cluster_arguments(cluster_name, job_ids))

    def cancel(self, cluster_name: str, job_ids: Sequence[int]) -> str:
        return self.session.call("cluster_cancel", **_cluster_arguments(cluster_name, job_ids))

    def queue(self, cluster_name: str, *, skip_finished: bool = False, all_users: bool = False) -> str:
        if type(skip_finished) is not bool or all_users is not False:
            msg = "Cluster queue requires a boolean skip_finished and all_users=False."
            raise ValueError(msg)
        return self.session.call(
            "cluster_queue", **_cluster_arguments(cluster_name), skip_finished=skip_finished, all_users=False
        )

    def api_status(self, **kwargs: Any) -> Any:
        return self.session.call("api_status", **kwargs)

    def api_info(self) -> Any:
        return self.session.call("api_info")


# Child-only SDK broker and server supervisor
# ----------------------------------------------------------------------------

_GRACE_S = 5


def _stop_tree(process: subprocess.Popen[bytes]) -> None:
    """Reap only our own child and its descendants, including detached workers."""
    import psutil

    descendants = []
    with contextlib.suppress(psutil.NoSuchProcess):
        descendants = psutil.Process(process.pid).children(recursive=True)
    with contextlib.suppress(ProcessLookupError):
        process.terminate()
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=_GRACE_S)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    for child in descendants:
        with contextlib.suppress(psutil.NoSuchProcess):
            child.kill()
    process.wait(timeout=_GRACE_S)
    psutil.wait_procs(descendants, timeout=_GRACE_S)


def _choose_port() -> int:
    """Probe both the HTTP port and SkyPilot's derived internal queue port."""
    for _ in range(100):
        port = 20000 + secrets.randbelow(20000)
        queue_port = 50000 + (port - 46569) % 10000
        try:
            with socket.socket() as http, socket.socket() as internal:
                http.bind(("127.0.0.1", port))
                internal.bind(("127.0.0.1", queue_port))
        except OSError:
            continue
        return port
    msg = "Could not find free SkyPilot HTTP and request-queue ports."
    raise RuntimeError(msg)


def _load_isolated_sdk() -> Any:
    """Require native runtime isolation and relocate the remaining identity file."""
    try:
        import sky
        from sky.skylet import runtime_utils
        from sky.utils import cluster_utils, common_utils
    except ModuleNotFoundError as exc:
        msg = "Isolated API sessions require `misen[skypilot-managed]` (SkyPilot nightly)."
        raise RuntimeError(msg) from exc
    if not hasattr(runtime_utils, "runtime_tilde_path"):
        msg = (
            "This SkyPilot build lacks runtime isolation; "
            "install `misen[skypilot-managed]` instead of `misen[skypilot]`."
        )
        raise RuntimeError(msg)
    # Upstream's runtime isolation does not yet cover this one legacy path.
    # SKYPILOT_USER_ID is set before imports; only the server startup writes it.
    common_utils.USER_HASH_FILE = runtime_utils.expanduser("~/.sky/user_hash")
    # Generated SSH shortcuts are runtime output, not user credentials. Keep
    # pool/controller provisioning from rewriting the ordinary SSH config.
    ssh = cluster_utils.SSHConfigHelper
    ssh.ssh_conf_path = runtime_utils.expanduser("~/.ssh/config")
    ssh.ssh_conf_lock_path = runtime_utils.expanduser("~/.sky/locks/.ssh_config.lock")
    ssh.ssh_conf_per_cluster_lock_path = runtime_utils.expanduser("~/.sky/locks/.ssh_config_{}.lock")
    ssh.ssh_cluster_path = runtime_utils.expanduser("~/.sky/generated/ssh/{}")
    ssh.ssh_cluster_key_path = runtime_utils.expanduser("~/.sky/generated/ssh-keys/{}.key")
    if sky.skypilot_config.get_nested(("jobs", "controller", "consolidation_mode"), default_value=False):
        msg = "Misen namespaces require jobs.controller.consolidation_mode=false so cloud jobs outlive local sessions."
        raise RuntimeError(msg)
    if sky.skypilot_config.get_nested(("db",), default_value=None):
        msg = "Misen namespaces cannot use an external/shared SkyPilot database."
        raise RuntimeError(msg)
    return sky


def _encode_result(value: Any) -> Any:
    """Serialize the executor's SDK responses without pickling backend objects."""
    if isinstance(value, Enum):
        return _encode_result(value.value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return {"__tuple__": [_encode_result(item) for item in value]}
    if isinstance(value, list):
        return [_encode_result(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _encode_result(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return _encode_result(value.model_dump(mode="json"))
    msg = f"Unsupported SkyPilot response type: {type(value).__name__}"
    raise TypeError(msg)


def _api_status(**arguments: Any) -> Any:
    """Query our endpoint without SkyPilot's native command-line process probe.

    Our server runs through this module's isolation bootstrap, so the SDK's
    search for ``-m sky.server.server`` incorrectly reports it as stopped.
    Use the pinned SDK's authenticated transport and payloads, without changing
    its process detection or exposing any global start/stop operations.
    """
    from sky.client import common as client_common
    from sky.server import common as server_common
    from sky.server.requests import payloads

    body = payloads.RequestStatusBody(**arguments)
    response = server_common.make_authenticated_request(
        "GET",
        "/api/status",
        params=server_common.request_body_to_params(body),
        timeout=(client_common.API_SERVER_REQUEST_CONNECTION_TIMEOUT_SECONDS, None),
    )
    server_common.handle_request_error(response)
    return [payloads.RequestPayload(**request) for request in response.json()]


def _dispatch(sky: Any, operation: str, arguments: dict[str, Any]) -> Any:
    """Expose only the SDK operations required by Misen and namespace pools."""
    if operation == "validate_resources":
        sky.Resources(**arguments["options"]).validate()
        return None
    if operation in {"launch", "cluster_exec"}:
        arguments = dict(arguments)
        task = _task_from_options(sky, arguments.pop("task"))
        if operation == "launch":
            return str(sky.jobs.launch(task, **arguments))

        if set(arguments) != {"cluster_name"}:
            msg = "Cluster execution accepts only an explicit cluster name."
            raise ValueError(msg)
        return str(sky.exec(task, **_cluster_arguments(arguments["cluster_name"])))
    if operation in {"cluster_job_status", "cluster_cancel"}:
        if set(arguments) != {"cluster_name", "job_ids"} or arguments["job_ids"] is None:
            msg = "Cluster status and cancellation require explicit job IDs."
            raise ValueError(msg)
        options = _cluster_arguments(arguments["cluster_name"], arguments["job_ids"])
        method = sky.job_status if operation == "cluster_job_status" else sky.cancel
        return str(method(**options))
    if operation == "cluster_queue":
        if (
            set(arguments) != {"cluster_name", "skip_finished", "all_users"}
            or type(arguments["skip_finished"]) is not bool
            or arguments["all_users"] is not False
        ):
            msg = "Cluster queue is scoped to the current user's explicitly named cluster."
            raise ValueError(msg)
        return str(
            sky.queue(
                **_cluster_arguments(arguments["cluster_name"]),
                skip_finished=arguments["skip_finished"],
                all_users=False,
            )
        )
    if operation in {"queue_v2", "cancel"}:
        return str(getattr(sky.jobs, operation)(**arguments))
    if operation == "get":
        result = sky.get(arguments["request_id"])
        # Managed launches return a list of IDs; cluster exec returns one ID.
        # Backend handles are deliberately never transported to the parent.
        if (
            isinstance(result, tuple)
            and len(result) == 2  # noqa: PLR2004
            and (result[0] is None or isinstance(result[0], list) or type(result[0]) is int)
        ):
            return result[0], None
        return result
    if operation == "api_status":
        return _api_status(**arguments)
    if operation == "api_info":
        return sky.api_info()
    if operation == "check":
        from sky.client import sdk

        return sky.get(sdk.check(infra_list=tuple(arguments["infra_list"]), verbose=arguments["verbose"]))
    if operation == "pool_apply":
        from sky.serve.serve_utils import UpdateMode

        return sky.get(
            sky.jobs.pool_apply(
                sky.Task.from_yaml(arguments["config"]), pool_name=arguments["pool_name"], mode=UpdateMode.ROLLING
            )
        )
    if operation == "pool_down":
        return sky.get(sky.jobs.pool_down(pool_names=[arguments["pool_name"]]))
    if operation == "pool_status":
        records = sky.get(sky.jobs.pool_status(pool_names=None))
        # The nightly already includes readable cloud/region/resource strings.
        # Drop legacy opaque handles, which are unnecessary for pool status.
        return [
            dict(
                record,
                replica_info=[
                    {key: value for key, value in worker.items() if key != "handle"}
                    for worker in record.get("replica_info", [])
                ],
            )
            for record in records
        ]
    msg = f"Unsupported SkyPilot broker operation: {operation}"
    raise ValueError(msg)


def _task_from_options(sky: Any, options: dict[str, Any]) -> Any:
    """Reconstruct an SDK task without transporting opaque resource objects."""
    options = dict(options)
    if "resources" in options:
        options["resources"] = [sky.Resources(**item) for item in options["resources"]]
    return sky.Task(**options)


class _Leases:
    """Track live clients independently of potentially blocking SDK requests."""

    def __init__(self, dispatch: Callable[[str, dict[str, Any]], Any]) -> None:
        self.dispatch = dispatch
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.clients = 0
        self.acquired = False
        self.last_connection: Connection | None = None

    def bootstrap(self, fd: int) -> None:
        """Detect creator death before it acquires its first socket lease."""
        os.read(fd, 1)
        with self.lock:
            if not self.acquired:
                self.stop.set()

    def accept(self, listener: Listener) -> None:
        """Accept only local authenticated connections."""
        while not self.stop.is_set():
            try:
                connection = listener.accept()
            except (OSError, EOFError):
                return
            threading.Thread(target=self._read, args=(connection,), daemon=True).start()

    def _read(self, connection: Connection) -> None:
        leased = False
        defer_close = False
        requests: queue_module.Queue[dict[str, Any] | None] = queue_module.Queue()
        worker = threading.Thread(target=self._work, args=(connection, requests), daemon=True)
        try:
            while True:
                message = json.loads(connection.recv_bytes())
                if message["op"] == "acquire":
                    with self.lock:
                        if leased or self.stop.is_set():
                            connection.send_bytes(b'{"error":"Server is shutting down"}')
                            return
                        self.clients += 1
                        leased = self.acquired = True
                    connection.send_bytes(b'{"result":"acquired"}')
                    worker.start()
                elif not leased:
                    return
                elif message["op"] == "release":
                    with self.lock:
                        self.clients -= 1
                        leased = False
                        if not self.clients:
                            defer_close = True
                            self.last_connection = connection
                            self.stop.set()
                    if not defer_close:
                        connection.send_bytes(b'{"result":null}')
                    return
                else:
                    requests.put(message)
        except (OSError, EOFError, ValueError, KeyError):
            pass
        finally:
            requests.put(None)
            with self.lock:
                if leased:
                    self.clients -= 1
                    if not self.clients:
                        self.stop.set()
            if not defer_close:
                connection.close()

    def _work(self, connection: Connection, requests: queue_module.Queue[dict[str, Any] | None]) -> None:
        while (message := requests.get()) is not None and not self.stop.is_set():
            try:
                result = self.dispatch(message["op"], message.get("args", {}))
                reply = {"result": _encode_result(result)}
            except Exception as exc:  # noqa: BLE001
                reply = {"error": f"{type(exc).__name__}: {exc}"}
            try:
                connection.send_bytes(json.dumps(reply).encode())
            except (OSError, EOFError):
                return

    def finish(self) -> None:
        """Acknowledge the last graceful release only after the tree is stopped."""
        if self.last_connection is not None:
            with contextlib.suppress(OSError, EOFError):
                self.last_connection.send_bytes(b'{"result":null}')
            self.last_connection.close()


def _wait_ready(child: subprocess.Popen[bytes], endpoint: str, stop: threading.Event) -> None:
    deadline = time.monotonic() + 100
    while not stop.is_set() and child.poll() is None and time.monotonic() < deadline:
        try:
            with urlopen(f"{endpoint}/api/health", timeout=1) as response:  # noqa: S310
                if json.load(response).get("status") == "healthy":
                    return
        except (URLError, OSError, ValueError):
            pass
        stop.wait(0.1)
    msg = "SkyPilot server failed to become healthy before timeout or client disconnect."
    raise RuntimeError(msg)


def _serve(directory: Path, log_path: Path, handshake: TextIO) -> None:
    from filelock import FileLock

    leases = _Leases(lambda operation, arguments: _dispatch(sky, operation, arguments))
    threading.Thread(target=leases.bootstrap, args=(sys.stdin.fileno(),), daemon=True).start()
    signal.signal(signal.SIGTERM, lambda *_: leases.stop.set())
    signal.signal(signal.SIGINT, lambda *_: leases.stop.set())
    with FileLock(directory / "lifetime.lock", timeout=25):
        if leases.stop.is_set():
            return
        port = _choose_port()
        endpoint = f"http://127.0.0.1:{port}"
        os.environ.update(SKYPILOT_API_SERVER_LOCAL_PORT=str(port), SKYPILOT_API_SERVER_ENDPOINT=endpoint)
        sky = _load_isolated_sdk()
        # Never let this SDK spawn an unowned replacement if our child dies.
        sky.server.common.check_server_healthy_or_start_fn = lambda *_args, **_kwargs: (
            sky.server.common.check_server_healthy()
        )
        env = dict(os.environ, IS_SKYPILOT_SERVER="true")
        child = subprocess.Popen(  # noqa: S603
            [sys.executable, "-m", "misen.executors.skypilot", "--server", "--host=127.0.0.1", f"--port={port}"],
            stdin=subprocess.DEVNULL,
            stdout=sys.stderr,
            stderr=sys.stderr,
            start_new_session=True,
            env=env,
        )
        descriptor_path = directory / "server.json"
        try:
            _wait_ready(child, endpoint, leases.stop)
            with tempfile.TemporaryDirectory(prefix="misen-sky-") as socket_dir:
                address = str(Path(socket_dir) / "rpc.sock")
                authkey = secrets.token_bytes(32)
                with Listener(address, family="AF_UNIX", authkey=authkey) as listener:
                    Path(address).chmod(0o600)
                    descriptor = {
                        "address": address,
                        "authkey": authkey.hex(),
                        "endpoint": endpoint,
                        "log_path": str(log_path),
                        "pid": os.getpid(),
                        "server_pid": child.pid,
                    }
                    descriptor_path.touch(mode=0o600, exist_ok=True)
                    descriptor_path.write_text(json.dumps(descriptor))
                    threading.Thread(target=leases.accept, args=(listener,), daemon=True).start()
                    handshake.write(json.dumps(descriptor) + "\n")
                    handshake.flush()
                    while not leases.stop.wait(0.2) and child.poll() is None:
                        pass
        finally:
            leases.stop.set()
            _stop_tree(child)
            descriptor_path.unlink(missing_ok=True)
            leases.finish()


def _broker_main() -> None:
    if sys.argv[1] == "--server":
        _load_isolated_sdk()
        sys.argv = ["sky.server.server", *sys.argv[2:]]
        runpy.run_module("sky.server.server", run_name="__main__")
        return
    # SkyPilot/loggers may write to stdout on import. Keep the one-message
    # handshake on its own fd; all other output belongs in the private log.
    with os.fdopen(os.dup(sys.stdout.fileno()), "w") as handshake:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        try:
            _serve(Path(sys.argv[1]), Path(sys.argv[2]), handshake)
        except Exception as exc:
            handshake.write(json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n")
            handshake.flush()
            raise
    # SDK background threads must not keep an otherwise cleaned broker alive.
    os._exit(0)
