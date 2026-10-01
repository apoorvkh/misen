"""Remote process supervision for SSH workers, independent of the SSH client."""

# Keep remote code self-contained so bootstrap needs only the standard library.
# ruff: noqa: EM101, EM102, TRY003
from __future__ import annotations

import importlib
import inspect
import math
import os
import shlex
import shutil
import subprocess

import msgspec

from misen.exceptions import ConfigError
from misen.task_metadata import Resources


def worker_preflight() -> None:
    """Check requested accelerator memory inside the prepared task environment."""
    resources = msgspec.json.decode(os.environ.pop("MISEN_SSH_RESOURCES"), type=Resources)
    count, minimum = resources["accelerators"], resources["accelerator_memory"]
    if not count:
        return
    backend = resources["accelerator_type"]
    if backend == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        try:
            result = subprocess.run(  # noqa: S603 -- fixed executable and separate device argument
                [
                    shutil.which("nvidia-smi") or "/usr/bin/nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                    "-i",
                    visible,
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            capacities = [float(line.strip()) / 1024 for line in result.stdout.splitlines() if line.strip()]
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            raise ConfigError(f"Could not verify allocated CUDA devices: {exc}") from exc
    elif minimum is not None:
        try:
            torch = importlib.import_module("torch")
            runtime = torch.cuda if backend == "rocm" else torch.xpu
            capacities = [runtime.get_device_properties(i).total_memory / 2**30 for i in range(count)]
        except (ImportError, AttributeError, RuntimeError, AssertionError, IndexError) as exc:
            raise ConfigError(f"Could not verify allocated {backend} devices: {exc}") from exc
    else:
        return
    if len(capacities) != count or any(not math.isfinite(value) or value < (minimum or 0) for value in capacities):
        raise ConfigError(f"Assigned GPUs provide {capacities} GiB; requested {count} devices with {minimum} GiB each.")


def remote_agent() -> None:
    """Own a process group until completion, cancellation, timeout, or lease loss.

    This function is shipped as source and must use only local imports. The
    first stdin line is a launch description. Subsequent lines renew the
    controller lease or cancel execution. Task stdin is disconnected from this
    control channel. Signals and EOF trigger the same bounded group cleanup.
    """
    import contextlib
    import json
    import os
    import select
    import shutil
    import signal
    import subprocess
    import sys
    import time

    initial = bytearray()
    while not initial.endswith(b"\n"):
        byte = os.read(sys.stdin.fileno(), 1)
        if not byte:
            return
        initial.extend(byte)
    message = json.loads(initial)
    stopped = False

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True

    for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    available = sorted(os.sched_getaffinity(0))
    if message["capacity"] > len(available):
        raise ValueError("Declared SSH CPU capacity exceeds the remote process's CPU affinity.")
    cpus = [available[index] for index in message["cpus"]]
    # Affinity is inherited by the child and descendants, including bootstrap.
    os.sched_setaffinity(0, cpus)
    process = subprocess.Popen(  # noqa: S603 -- controller-supplied task, authenticated SSH channel
        [shutil.which("bash") or "/bin/bash", "-c", message["command"]],
        stdin=subprocess.DEVNULL,
        env=os.environ | message["env"],
        start_new_session=True,
    )
    started = renewed = time.monotonic()
    control = bytearray()
    status = 1
    try:
        while process.poll() is None:
            now = time.monotonic()
            if stopped or now - renewed >= message["lease"]:
                status = 130
                break
            if now - started >= message["timeout"]:
                status = 124
                break
            readable, _, _ = select.select([sys.stdin.fileno()], [], [], 0.1)
            if readable:
                data = os.read(sys.stdin.fileno(), 65536)
                if not data:
                    status = 130
                    break
                control.extend(data)
                while b"\n" in control:
                    line, _, remainder = control.partition(b"\n")
                    control = bytearray(remainder)
                    if line == b"cancel":
                        stopped = True
                    elif line == b"ping":
                        renewed = time.monotonic()
        else:
            status = process.returncode
    finally:
        # Clean surviving descendants even when their leader exited normally.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + message["grace"]
        while time.monotonic() < deadline:
            process.poll()
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    sys.exit(status if status >= 0 else 128 - status)


def agent_command() -> str:
    """Render a quoted bootstrap which does not require Misen to be installed."""
    return shlex.join(["python3", "-u", "-c", inspect.getsource(remote_agent) + "\nremote_agent()\n"])
