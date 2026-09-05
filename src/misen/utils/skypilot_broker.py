"""Child-only SDK broker and lease-based supervisor for an isolated namespace.

Only JSON crosses the authenticated local socket. SDK imports and environment
settings stay in this process; an EOF watcher remains responsive even while an
SDK call blocks. The last client disconnect stops our server process tree.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import runpy
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from enum import Enum
from multiprocessing.connection import Listener
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.error import URLError
from urllib.request import urlopen

if TYPE_CHECKING:
    from collections.abc import Callable
    from multiprocessing.connection import Connection
    from typing import TextIO

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
    if operation == "launch":
        options = dict(arguments.pop("task"))
        options["resources"] = [sky.Resources(**item) for item in options["resources"]]
        return str(sky.jobs.launch(sky.Task(**options), **arguments))
    if operation in {"queue_v2", "cancel"}:
        return str(getattr(sky.jobs, operation)(**arguments))
    if operation == "get":
        result = sky.get(arguments["request_id"])
        # Managed launches return (job IDs, backend handle). Only IDs are used
        # by Misen; handles are deliberately never transported to the parent.
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], list):  # noqa: PLR2004
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
        requests: queue.Queue[dict[str, Any] | None] = queue.Queue()
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

    def _work(self, connection: Connection, requests: queue.Queue[dict[str, Any] | None]) -> None:
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
            [sys.executable, "-m", "misen.utils.skypilot_broker", "--server", "--host=127.0.0.1", f"--port={port}"],
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


def _main() -> None:
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


if __name__ == "__main__":
    _main()
elif __name__ == "__mp_main__":
    # Spawned SDK workers import this bootstrap afresh (notably on macOS).
    # Apply child-only compatibility paths before any request executes.
    _load_isolated_sdk()
