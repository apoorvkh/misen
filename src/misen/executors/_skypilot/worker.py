"""Reusable worker agent, guarded subprocess, and durable attempt protocol."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import select
import signal
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from misen.exceptions import (
    ExecutionError,
    StorageError,
)
from misen.utils.resource_env import narrow_accelerator_environment

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import BinaryIO

    from misen.workspace import Workspace

# Per-attempt process guard
# ----------------------------------------------------------------------------

_MAX_CONFIG_BYTES = 2 * 1024 * 1024


_POLL_S = 0.01


_FRAME_HEADER_BYTES = 4


_FAILURE_EXIT = 125


_TIMEOUT_EXIT = 124


def _read_exact(fd: int, size: int, deadline: float, stopped: list[bool]) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        remaining = deadline - time.monotonic()
        if stopped[0] or remaining <= 0:
            msg = "Worker guard stopped before receiving its configuration."
            raise ValueError(msg)
        readable, _, _ = select.select([fd], [], [], min(_POLL_S, remaining))
        if not readable:
            continue
        data = os.read(fd, size - len(chunks))
        if not data:
            msg = "Worker agent exited before publishing the full guard configuration."
            raise ValueError(msg)
        chunks.extend(data)
    return bytes(chunks)


def _configuration(fd: int, deadline: float, stopped: list[bool]) -> tuple[list[str], dict[str, str]]:
    size = struct.unpack("!I", _read_exact(fd, _FRAME_HEADER_BYTES, deadline, stopped))[0]
    if not 0 < size <= _MAX_CONFIG_BYTES:
        msg = "Invalid worker guard configuration size."
        raise ValueError(msg)
    record: Any = json.loads(_read_exact(fd, size, deadline, stopped))
    if not isinstance(record, dict):
        msg = "Worker guard configuration must be a JSON object."
        raise ValueError(msg)  # noqa: TRY004 -- all invalid wire data uses ValueError
    argv, env = record.get("argv"), record.get("env")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(arg, str) or "\x00" in arg for arg in argv)
        or not argv[0]
        or not isinstance(env, dict)
        or any(
            not key or "=" in key or "\x00" in key or not isinstance(value, str) or "\x00" in value
            for key, value in env.items()
        )
    ):
        msg = "Invalid worker guard argv or environment."
        raise ValueError(msg)
    return argv, env


def _parent_closed(fd: int) -> bool:
    readable, _, _ = select.select([fd], [], [], 0)
    if not readable:
        return False
    # No further messages are part of this protocol. EOF or unexpected data
    # both revoke the permission to keep executing this assignment.
    os.read(fd, 1)
    return True


def _stop_task_group(process: subprocess.Popen[bytes], grace_s: float) -> None:
    """Terminate and reap the leader, and kill descendants in its private group."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(min(_POLL_S, max(0.0, deadline - time.monotonic())))
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=grace_s)


def guard(parent_fd: int, *, deadline: float, grace_s: float) -> int:
    """Run one payload until completion, pipe EOF, a signal, or its hard deadline."""
    if os.name != "posix" or parent_fd < 0 or not math.isfinite(deadline) or not math.isfinite(grace_s) or grace_s <= 0:
        msg = "Worker guards require POSIX, a valid lifetime pipe, and bounded deadlines."
        raise ValueError(msg)
    stopped = [False]

    def stop(_signum: int, _frame: Any) -> None:
        stopped[0] = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    os.set_inheritable(parent_fd, False)  # noqa: FBT003 -- os exposes this flag as a positional-only argument
    process: subprocess.Popen[bytes] | None = None
    try:
        argv, env = _configuration(parent_fd, deadline, stopped)
        if stopped[0] or _parent_closed(parent_fd):
            return -signal.SIGTERM
        if time.monotonic() >= deadline:
            return _TIMEOUT_EXIT
        process = subprocess.Popen(  # noqa: S603 -- authenticated agent provides validated argv; shell=False
            argv, env=env, stdin=subprocess.DEVNULL, close_fds=True, start_new_session=True
        )
        while True:
            if stopped[0] or _parent_closed(parent_fd):
                return -signal.SIGTERM
            if time.monotonic() >= deadline:
                return _TIMEOUT_EXIT
            if (exit_code := process.poll()) is not None:
                return exit_code
            select.select([parent_fd], [], [], min(_POLL_S, max(0.0, deadline - time.monotonic())))
    finally:
        try:
            if process is not None:
                _stop_task_group(process, grace_s)
        finally:
            os.close(parent_fd)


def _guard_main() -> None:
    """Parse private launcher arguments and preserve payload exit/signal status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-fd", type=int, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--grace-s", type=float, required=True)
    args = parser.parse_args()
    try:
        status = guard(args.parent_fd, deadline=args.deadline, grace_s=args.grace_s)
    except Exception as exc:  # noqa: BLE001 -- never dump the configuration/environment into diagnostics
        sys.stderr.write(f"Worker process guard failed ({type(exc).__name__}).\n")
        sys.stderr.flush()
        status = _FAILURE_EXIT
    if status < 0:
        signum = -status
        if signum not in (signal.SIGKILL, signal.SIGSTOP):
            signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    raise SystemExit(status)


# Reusable worker agent
# ----------------------------------------------------------------------------

_WORKER_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


_MAX_JSON_BYTES = 1024 * 1024


_MAX_ARGUMENTS = 4096


_MAX_ENVIRONMENT = 4096


_PAYLOAD_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\.pkl\Z")


_VERSION = 1


_IO_ERRORS = (StorageError, OSError)


# A directly staged child may use the agent's durable accepted record as its
# execution fence.  These private environment variables form a capability
# handed across the guarded process boundary; they are always consumed before
# user code runs and are never honored piecemeal.
_DIRECT_WORKER_ID_ENV = "MISEN_SKYPILOT_DIRECT_WORKER_ID"


_DIRECT_GENERATION_ENV = "MISEN_SKYPILOT_DIRECT_GENERATION"


_DIRECT_JOB_ID_ENV = "MISEN_SKYPILOT_DIRECT_JOB_ID"


_DIRECT_CLAIM_TOKEN_ENV = "MISEN_SKYPILOT_DIRECT_CLAIM_TOKEN"  # noqa: S105 -- environment-variable name


_DIRECT_CLAIM_ENV = (
    _DIRECT_WORKER_ID_ENV,
    _DIRECT_GENERATION_ENV,
    _DIRECT_JOB_ID_ENV,
    _DIRECT_CLAIM_TOKEN_ENV,
)


def _token(value: object) -> str:
    if not isinstance(value, str) or _WORKER_TOKEN.fullmatch(value) is None:
        msg = "Worker protocol identifiers must be bounded alphanumeric tokens."
        raise ValueError(msg)
    return value


def worker_file_name(worker_id: str, kind: Literal["command", "lease", "state"]) -> str:
    """Return a flat, validated worker coordination filename."""
    if kind not in {"command", "lease", "state"}:
        msg = "Invalid worker record kind."
        raise ValueError(msg)
    return f"worker-{_token(worker_id)}.{kind}.json"


def attempt_file_name(attempt_id: str, kind: Literal["accepted", "started", "result"] | None = None) -> str:
    """Return a flat, validated attempt filename (terminal outcome by default)."""
    if kind not in {None, "accepted", "started", "result"}:
        msg = "Invalid attempt record kind."
        raise ValueError(msg)
    suffix = f".{kind}" if kind else ""
    return f"attempt-{_token(attempt_id)}{suffix}.json"


def _read_json(workspace: Workspace, run_id: str, name: str) -> dict[str, Any]:
    data = workspace.read_job_file(run_id, name)
    if len(data) > _MAX_JSON_BYTES:
        msg = "Worker protocol record exceeds the size limit."
        raise ValueError(msg)
    try:
        record = json.loads(data)
    except (ValueError, UnicodeError, RecursionError) as exc:
        msg = "Worker protocol record is not valid JSON."
        raise ValueError(msg) from exc
    if not isinstance(record, dict):
        msg = "Worker protocol record must be a JSON object."
        raise ValueError(msg)  # noqa: TRY004 -- malformed protocol data uses one validation error type
    if type(record.get("version")) is not int or record["version"] != _VERSION or record.get("run_id") != run_id:
        msg = "Worker protocol record has an invalid version or run identity."
        raise ValueError(msg)
    return record


def _write_json(workspace: Workspace, run_id: str, name: str, record: dict[str, Any]) -> None:
    data = json.dumps({"version": _VERSION, "run_id": run_id, **record}, allow_nan=False).encode()
    if len(data) > _MAX_JSON_BYTES:
        msg = "Worker protocol record exceeds the size limit."
        raise ValueError(msg)
    workspace.put_job_file(run_id, name, data)


def _positive_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = "Worker time limits must be finite positive numbers."
        raise ValueError(msg)  # noqa: TRY004 -- protocol validation consistently raises ValueError
    try:
        number = float(value)
    except OverflowError as exc:
        msg = "Worker time limits must be finite positive numbers."
        raise ValueError(msg) from exc
    if not math.isfinite(number) or number <= 0:
        msg = "Worker time limits must be finite positive numbers."
        raise ValueError(msg)
    return number


def _log_path(value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        msg = "Worker log paths must be safe relative paths."
        raise ValueError(msg)
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        msg = "Worker log paths must be safe relative paths."
        raise ValueError(msg)
    root = Path.cwd().resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        msg = "Worker log path escapes the working directory."
        raise ValueError(msg)
    return resolved


@dataclass(frozen=True)
class _Command:
    attempt_id: str
    job_id: str
    run_id: str
    argv: list[str]
    env: dict[str, str]
    log_path: Path
    execution_timeout_s: float
    setup_timeout_s: float
    direct: bool
    payload_name: str | None

    @classmethod
    def parse(cls, record: dict[str, Any]) -> _Command:
        argv = record.get("argv")
        env = record.get("env")
        direct = record.get("direct", False)
        payload_name = record.get("payload_name")
        if (
            not isinstance(argv, list)
            or not argv
            or len(argv) > _MAX_ARGUMENTS
            or any(not isinstance(arg, str) or "\x00" in arg for arg in argv)
            or not argv[0]
        ):
            msg = "Worker argv must be a bounded nonempty list of strings."
            raise ValueError(msg)
        if (
            not isinstance(env, dict)
            or len(env) > _MAX_ENVIRONMENT
            or any(
                not isinstance(key, str)
                or not key
                or "=" in key
                or "\x00" in key
                or not isinstance(value, str)
                or "\x00" in value
                for key, value in env.items()
            )
        ):
            msg = "Worker env must contain bounded string environment entries."
            raise ValueError(msg)
        if type(direct) is not bool or (
            direct and (not isinstance(payload_name, str) or _PAYLOAD_NAME.fullmatch(payload_name) is None)
        ):
            msg = "Direct worker commands require one safe payload filename."
            raise ValueError(msg)
        if not direct and payload_name is not None:
            msg = "Bootstrap worker commands cannot include a direct payload filename."
            raise ValueError(msg)
        return cls(
            attempt_id=_token(record.get("attempt_id")),
            job_id=_token(record.get("job_id")),
            run_id=_token(record.get("target_run_id", record.get("run_id"))),
            argv=argv,
            env=env,
            log_path=_log_path(record.get("log_path")),
            execution_timeout_s=_positive_number(record.get("execution_timeout_s")),
            setup_timeout_s=_positive_number(record.get("setup_timeout_s")),
            direct=direct,
            payload_name=payload_name,
        )


def _terminate_group(process: subprocess.Popen[bytes], grace_s: float) -> None:
    """Reap the child and kill its remaining process group, including descendants."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        pass
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


@dataclass
class _ActiveAttempt:
    command: _Command
    process: subprocess.Popen[bytes]
    log: BinaryIO
    launched_at: float
    lifetime_fd: int
    execution_started_at: float | None = None
    payload_path: Path | None = None


@dataclass
class _Agent:
    workspace: Workspace
    control_id: str
    worker_id: str
    lease_timeout_s: float
    shutdown_grace_s: float
    poll_interval_s: float
    max_runtime_s: float
    generation: str = field(default_factory=lambda: uuid.uuid4().hex)
    base_env: dict[str, str] = field(default_factory=lambda: dict(os.environ))
    seen_attempts: set[str] = field(default_factory=set)
    cancelled_attempts: set[str] = field(default_factory=set)
    active: _ActiveAttempt | None = None
    lease_sequence: int = -1
    lease_at: float = field(default_factory=time.monotonic)

    def _state(self, state: str, attempt_id: str | None = None) -> None:
        _write_json(
            self.workspace,
            self.control_id,
            worker_file_name(self.worker_id, "state"),
            {"worker_id": self.worker_id, "generation": self.generation, "state": state, "attempt_id": attempt_id},
        )

    def _lease(self) -> str | None:
        try:
            record = _read_json(self.workspace, self.control_id, worker_file_name(self.worker_id, "lease"))
        except FileNotFoundError:
            return "coordinator lease disappeared" if self.lease_sequence >= 0 else None
        except (*_IO_ERRORS, ValueError):
            return "coordinator lease could not be read safely"
        sequence = record.get("sequence")
        stop = record.get("stop")
        if (
            record.get("worker_id") != self.worker_id
            or type(sequence) is not int
            or sequence < 0
            or type(stop) is not bool
        ):
            return "invalid coordinator lease"
        if sequence > self.lease_sequence:
            self.lease_sequence = sequence
            self.lease_at = time.monotonic()
            if stop:
                return "coordinator requested stop"
            cancelled = record.get("cancel_attempt_id")
            if cancelled is not None:
                _token(cancelled)
                self.cancelled_attempts.add(cancelled)
                if self.active is not None and self.active.command.attempt_id == cancelled:
                    self._finish(forced=("failed", "attempt cancelled by coordinator"))
                    self._state("idle")
        return None

    def _claim(self, command: _Command) -> str | None:
        """Durably accept one generation-bound command and return its capability.

        A worker generation is represented by exactly one agent process and a
        command is addressed to that generation.  ``seen_attempts`` prevents a
        visible command from being admitted twice by that process; the durable
        accepted record prevents a replacement generation from replaying it.
        The random capability is handed only to a directly staged child, which
        validates this record instead of taking the generic execution lock.
        """
        if command.attempt_id in self.seen_attempts:
            return None
        try:
            _read_json(self.workspace, command.run_id, attempt_file_name(command.attempt_id, "accepted"))
        except FileNotFoundError:
            pass
        else:
            self.seen_attempts.add(command.attempt_id)
            return None
        # A command is bound to this worker and its random generation. One
        # agent process is therefore the sole possible writer; a replacement
        # gets a new generation and rejects the stale command. A directly
        # staged child verifies this durable record and its random capability;
        # the bootstrap fallback retains the generic execution fence.
        claim_token = uuid.uuid4().hex
        _write_json(
            self.workspace,
            command.run_id,
            attempt_file_name(command.attempt_id, "accepted"),
            {
                "worker_id": self.worker_id,
                "generation": self.generation,
                "attempt_id": command.attempt_id,
                "job_id": command.job_id,
                "claim_token": claim_token,
            },
        )
        self.seen_attempts.add(command.attempt_id)
        return claim_token

    def _outcome(self, command: _Command, state: str, reason: str | None) -> None:
        _write_json(
            self.workspace,
            command.run_id,
            attempt_file_name(command.attempt_id),
            {
                "worker_id": self.worker_id,
                "generation": self.generation,
                "attempt_id": command.attempt_id,
                "job_id": command.job_id,
                "state": state,
                "reason": reason,
            },
        )

    def _admit(self, deadline: float) -> None:
        try:
            record = _read_json(self.workspace, self.control_id, worker_file_name(self.worker_id, "command"))
        except FileNotFoundError:
            return
        if record.get("worker_id") != self.worker_id or record.get("generation") != self.generation:
            return
        command = _Command.parse(record)
        claim_token = self._claim(command)
        if claim_token is None:
            return
        if command.attempt_id in self.cancelled_attempts:
            self._outcome(command, "failed", "attempt cancelled by coordinator before process launch")
            return
        if time.monotonic() - self.lease_at >= self.lease_timeout_s or time.monotonic() >= deadline:
            self._outcome(command, "failed", "worker lease or lifetime expired before process launch")
            return
        log = None
        payload_path: Path | None = None
        lifetime_read: int | None = None
        lifetime_write: int | None = None
        try:
            command.log_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(command.log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
            os.fchmod(descriptor, 0o600)
            log = os.fdopen(descriptor, "ab", buffering=0)
            env = {
                **narrow_accelerator_environment(self.base_env, command.env),
                "MISEN_RUN_ID": command.run_id,
                "MISEN_ATTEMPT_ID": command.attempt_id,
            }
            # Reserved direct-claim variables must never arrive through a
            # dotenv file or task override.  They are capabilities supplied by
            # this generation only after its durable accepted write succeeds.
            for name in _DIRECT_CLAIM_ENV:
                env.pop(name, None)
            argv = command.argv
            if command.direct:
                payload_path = self._stage_payload(command)
                if payload_path is not None:
                    env.update(
                        {
                            _DIRECT_WORKER_ID_ENV: self.worker_id,
                            _DIRECT_GENERATION_ENV: self.generation,
                            _DIRECT_JOB_ID_ENV: command.job_id,
                            _DIRECT_CLAIM_TOKEN_ENV: claim_token,
                        }
                    )
                    argv = [
                        sys.executable,
                        "-m",
                        "misen.utils.execute",
                        "--payload",
                        str(payload_path),
                        "--job-log-path",
                        str(command.log_path),
                    ]
            configuration = json.dumps({"argv": argv, "env": env}).encode()
            if len(configuration) > 2 * _MAX_JSON_BYTES:
                msg = "Worker guard configuration exceeds the size limit."
                raise ValueError(msg)  # noqa: TRY301 -- retain the normal accepted-attempt failure path
            lifetime_read, lifetime_write = os.pipe()
            # Guard cleanup (TERM grace plus bounded reap) fits inside half
            # the agent's grace, leaving room before the fallback SIGKILL.
            guard_grace_s = min(5.0, self.shutdown_grace_s / 4)
            guard_deadline = min(deadline, time.monotonic() + command.setup_timeout_s + command.execution_timeout_s)
            process = subprocess.Popen(  # noqa: S603 -- authenticated coordinator supplies argv, never a shell string
                [
                    sys.executable,
                    "-m",
                    "misen.executors.skypilot",
                    "--worker-guard",
                    "--parent-fd",
                    str(lifetime_read),
                    "--deadline",
                    str(guard_deadline),
                    "--grace-s",
                    str(guard_grace_s),
                ],
                env=self.base_env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(lifetime_read,),
            )
            self.active = _ActiveAttempt(
                command, process, log, time.monotonic(), lifetime_write, payload_path=payload_path
            )
            payload_path = None  # Ownership moves to the active attempt only.
            lifetime_write = None  # Ownership moves to the active attempt only.
            os.close(lifetime_read)
            lifetime_read = None
            framed = struct.pack("!I", len(configuration)) + configuration
            offset = 0
            while offset < len(framed):
                offset += os.write(self.active.lifetime_fd, framed[offset:])
            self._state("running", command.attempt_id)
        except BaseException:
            if self.active is None:
                if log is not None:
                    log.close()
                self._outcome(command, "failed", "worker could not start the task process")
            raise
        finally:
            if lifetime_read is not None:
                os.close(lifetime_read)
            if lifetime_write is not None:
                os.close(lifetime_write)
            if payload_path is not None:
                with contextlib.suppress(OSError):
                    payload_path.unlink()

    def _stage_payload(self, command: _Command) -> Path | None:
        """Stage a trusted payload for direct execution, or select the bootstrap fallback."""
        if command.payload_name is None:
            return None
        try:
            payload = self.workspace.read_job_file(command.run_id, command.payload_name)
            root = Path(self.workspace.get_temp_dir()) / "agent-payloads" / self.worker_id
            root.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=f".{command.attempt_id}-", dir=root)
            target = root / f"{command.attempt_id}.pkl"
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                Path(temporary).replace(target)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
                with contextlib.suppress(OSError):
                    Path(temporary).unlink()
                raise
        except _IO_ERRORS:
            return None
        return target

    def _completion(self, active: _ActiveAttempt) -> tuple[str, str | None]:
        try:
            result = _read_json(
                self.workspace, active.command.run_id, attempt_file_name(active.command.attempt_id, "result")
            )
        except (FileNotFoundError, *_IO_ERRORS, ValueError):
            result = {}
        if active.process.returncode != 0:
            reason = f"task process exited with status {active.process.returncode}; committed outputs are preserved"
            return "failed", reason
        if (
            result.get("attempt_id") != active.command.attempt_id
            or not isinstance(result.get("state"), str)
            or result["state"] not in {"done", "failed"}
        ):
            return "unknown", "task process exited without a matching durable result"
        if result["state"] == "failed":
            return "failed", "task callable reported failure"
        return "done", None

    def _finish(self, *, forced: tuple[str, str] | None = None) -> None:
        active = self.active
        if active is None:
            return
        try:
            os.close(active.lifetime_fd)
            _terminate_group(active.process, self.shutdown_grace_s)
            state, reason = forced if forced is not None else self._completion(active)
            self._outcome(active.command, state, reason)
        finally:
            active.log.close()
            if active.payload_path is not None:
                with contextlib.suppress(OSError):
                    active.payload_path.unlink()
            self.active = None

    def _poll_active(self) -> None:
        active = self.active
        if active is None:
            return
        if active.process.poll() is not None:
            self._finish()
            self._state("idle")
            return
        if active.execution_started_at is None:
            try:
                started = _read_json(
                    self.workspace, active.command.run_id, attempt_file_name(active.command.attempt_id, "started")
                )
            except FileNotFoundError:
                started = {}
            if started.get("attempt_id") == active.command.attempt_id and started.get("state") == "running":
                active.execution_started_at = time.monotonic()
        now = time.monotonic()
        if active.execution_started_at is None:
            expired = now - active.launched_at >= active.command.setup_timeout_s
            reason = "task setup timed out"
        else:
            expired = now - active.execution_started_at >= active.command.execution_timeout_s
            reason = "task execution timed out"
        if expired:
            self._finish(forced=("failed", reason))
            self._state("idle")

    def run(self) -> None:
        deadline = time.monotonic() + self.max_runtime_s
        stop_reason = "worker stopped before task completion"
        try:
            self._state("idle")
            while True:
                lease_stop = self._lease()
                if lease_stop is not None:
                    stop_reason = lease_stop
                    break
                if time.monotonic() - self.lease_at >= self.lease_timeout_s:
                    stop_reason = "coordinator lease expired"
                    break
                if time.monotonic() >= deadline:
                    stop_reason = "worker maximum lifetime expired"
                    break
                self._poll_active()
                if self.active is None and self.lease_sequence >= 0:
                    self._admit(deadline)
                time.sleep(min(self.poll_interval_s, max(0, deadline - time.monotonic())))
        finally:
            try:
                self._finish(forced=("unknown", stop_reason))
            finally:
                self._state("stopped")


def run_worker_agent(
    workspace: Workspace,
    control_id: str,
    worker_id: str,
    *,
    lease_timeout_s: float = 60,
    shutdown_grace_s: float = 30,
    poll_interval_s: float = 0.2,
    max_runtime_s: float = 86400,
) -> None:
    """Run a leased single-slot agent, terminating only its own task processes.

    The parent coordinator must refresh a strictly increasing lease sequence.
    Repeated leases cannot extend lifetime. Setup and callable execution have
    separate deadlines; durable callable success alone does not prove process
    cleanup succeeded. Accepted attempts never replay after a process restart.
    """
    if os.name != "posix":
        msg = "Reusable worker agents require POSIX process-group support."
        raise ValueError(msg)
    if not workspace.supports_job_file_reads():
        msg = "Reusable worker agents require readable workspace coordination files."
        raise ValueError(msg)
    _Agent(
        workspace=workspace,
        control_id=_token(control_id),
        worker_id=_token(worker_id),
        lease_timeout_s=_positive_number(lease_timeout_s),
        shutdown_grace_s=_positive_number(shutdown_grace_s),
        poll_interval_s=_positive_number(poll_interval_s),
        max_runtime_s=_positive_number(max_runtime_s),
    ).run()


# Worker execution claims and durable outcomes
# ----------------------------------------------------------------------------

_RUN_ID_ENV = "MISEN_RUN_ID"


_ATTEMPT_ID_ENV = "MISEN_ATTEMPT_ID"


_ATTEMPT_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


_ERROR_TYPE_LIMIT = 80


_ATTEMPT_RECORD_LIMIT = 4096


_CLAIM_LOCK_TIMEOUT = 30


@dataclass(frozen=True)
class _DirectClaim:
    """Capability proving that one reusable agent accepted this attempt."""

    worker_id: str
    generation: str
    job_id: str
    claim_token: str


@dataclass(frozen=True)
class _AttemptIdentity:
    """Validated coordination identity consumed before user code imports."""

    run_id: str
    attempt_id: str
    direct_claim: _DirectClaim | None = None


def _consume_attempt_identity() -> _AttemptIdentity | None:
    """Remove and validate optional worker coordination identifiers."""
    run_id = os.environ.pop(_RUN_ID_ENV, None)
    attempt_id = os.environ.pop(_ATTEMPT_ID_ENV, None)
    direct_values = {name: os.environ.pop(name, None) for name in _DIRECT_CLAIM_ENV}
    if run_id is None and attempt_id is None:
        if any(value is not None for value in direct_values.values()):
            msg = "Direct worker claim variables require a complete attempt identity."
            raise ValueError(msg)
        return None
    if (
        run_id is None
        or attempt_id is None
        or _ATTEMPT_TOKEN.fullmatch(run_id) is None
        or _ATTEMPT_TOKEN.fullmatch(attempt_id) is None
    ):
        msg = "MISEN_RUN_ID and MISEN_ATTEMPT_ID must both contain valid bounded coordination identifiers."
        raise ValueError(msg)
    supplied = [value is not None for value in direct_values.values()]
    if any(supplied) and not all(supplied):
        msg = "Direct worker claim variables must be supplied together."
        raise ValueError(msg)
    direct_claim = None
    if all(supplied):
        try:
            worker_id = _token(direct_values[_DIRECT_WORKER_ID_ENV])
            generation = _token(direct_values[_DIRECT_GENERATION_ENV])
            job_id = _token(direct_values[_DIRECT_JOB_ID_ENV])
            claim_token = _token(direct_values[_DIRECT_CLAIM_TOKEN_ENV])
        except ValueError as exc:
            msg = "Direct worker claim variables contain an invalid bounded identifier."
            raise ValueError(msg) from exc
        direct_claim = _DirectClaim(
            worker_id=worker_id,
            generation=generation,
            job_id=job_id,
            claim_token=claim_token,
        )
    return _AttemptIdentity(run_id, attempt_id, direct_claim)


def _publish_attempt(
    workspace: Workspace,
    identity: _AttemptIdentity,
    state: str,
    error: BaseException | None = None,
) -> None:
    """Publish a bounded outcome without copying exception text or credentials."""
    run_id, attempt_id = identity.run_id, identity.attempt_id
    record: dict[str, int | str] = {
        "version": 1,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "state": state,
    }
    if error is not None:
        error_type = re.sub(r"[^A-Za-z0-9_]", "_", type(error).__name__)[:_ERROR_TYPE_LIMIT] or "BaseException"
        record["error_type"] = error_type
        # Exception messages can contain credentials, URLs, or whole payloads.
        # Keep them in the existing traceback only, never coordination records.
        record["error_message"] = f"Worker payload raised {error_type}; inspect its task log for details."
    suffix = {"claimed": "execution", "running": "started"}.get(state, "result")
    workspace.put_job_file(run_id, f"attempt-{attempt_id}.{suffix}.json", json.dumps(record).encode("utf-8"))


def _claim_attempt(workspace: Workspace, identity: _AttemptIdentity) -> bool:
    """Claim execution once, refusing to replay an uncertain earlier invocation."""
    run_id, attempt_id = identity.run_id, identity.attempt_id
    lock_key = "execution-" + hashlib.sha256(json.dumps((run_id, attempt_id)).encode("utf-8")).hexdigest()
    lock = workspace.lock("job", lock_key)
    with lock.context(timeout=_CLAIM_LOCK_TIMEOUT):
        try:
            result_data = workspace.read_job_file(run_id, f"attempt-{attempt_id}.result.json")
        except FileNotFoundError:
            result_data = None
        if result_data is not None:
            result = None
            if len(result_data) <= _ATTEMPT_RECORD_LIMIT:
                with contextlib.suppress(ValueError, UnicodeError):
                    result = json.loads(result_data)
            if (
                isinstance(result, dict)
                and type(result.get("version")) is int
                and result["version"] == 1
                and result.get("run_id") == run_id
                and result.get("attempt_id") == attempt_id
                and result.get("state") == "done"
            ):
                return False
            msg = "The worker attempt already has an unsuccessful or invalid outcome; refusing to replay execution."
            raise ExecutionError(msg)
        try:
            workspace.read_job_file(run_id, f"attempt-{attempt_id}.execution.json")
        except FileNotFoundError:
            pass
        else:
            msg = "An earlier worker invocation claimed this attempt without committed success; refusing to replay it."
            raise ExecutionError(msg)
        if not lock.is_locked():
            msg = "Lost the execution claim lock before publishing the attempt claim."
            raise ExecutionError(msg)
        _publish_attempt(workspace, identity, "claimed")
    return True


def _verify_direct_claim(workspace: Workspace, identity: _AttemptIdentity) -> None:
    """Verify the agent's durable acceptance capability without a second lock.

    The coordinator assigns an attempt to one worker generation, and automatic
    coordinator takeover/replay is disabled.  That generation has one agent
    process: its in-memory ``seen_attempts`` fence prevents a second admission,
    while the accepted record prevents a replacement generation from admitting
    the same attempt.  The child therefore needs one authenticated read, rather
    than another distributed lock plus execution/result probes and claim writes.

    Any absent, malformed, stale, or overwritten capability fails closed before
    the started marker or user callable.  The generic bootstrap/dedicated path
    continues to use :func:`_claim_attempt` unchanged.
    """
    claim = identity.direct_claim
    if claim is None:
        msg = "A direct attempt requires an agent acceptance capability."
        raise ExecutionError(msg)
    try:
        record = _read_json(workspace, identity.run_id, attempt_file_name(identity.attempt_id, "accepted"))
    except (FileNotFoundError, ValueError) as exc:
        msg = "The direct attempt has no valid durable agent acceptance; refusing execution."
        raise ExecutionError(msg) from exc
    expected = {
        "worker_id": claim.worker_id,
        "generation": claim.generation,
        "attempt_id": identity.attempt_id,
        "job_id": claim.job_id,
        "claim_token": claim.claim_token,
    }
    if any(record.get(name) != value for name, value in expected.items()):
        msg = "The direct attempt does not match its durable agent acceptance; refusing execution."
        raise ExecutionError(msg)


def _execute_attempt(workspace: Workspace, payload_fn: Callable[[], None], identity: _AttemptIdentity) -> None:
    """Report execution boundaries without replacing a user-code exception."""
    if identity.direct_claim is None:
        if not _claim_attempt(workspace, identity):
            return
    else:
        _verify_direct_claim(workspace, identity)
    _publish_attempt(workspace, identity, "running")
    try:
        payload_fn()
    except BaseException as exc:
        try:
            _publish_attempt(workspace, identity, "failed", exc)
        except BaseException:  # noqa: BLE001 -- retain the original payload traceback
            exc.add_note("Additionally, publishing the failed attempt result did not succeed.")
        raise
    _publish_attempt(workspace, identity, "done")
