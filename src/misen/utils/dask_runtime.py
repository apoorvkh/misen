"""Private Dask runtime roles started by executor worker programs."""

from __future__ import annotations

import asyncio
import os
import shlex
from pathlib import Path
from textwrap import dedent

DASK_ROLE_ENV = "MISEN_DASK_ROLE"
DASK_SCHEDULER_ADDRESS_ENV = "MISEN_DASK_SCHEDULER_ADDRESS"
DASK_SCHEDULER_FILE_ENV = "MISEN_DASK_SCHEDULER_FILE"
DASK_EXPECTED_WORKERS_ENV = "MISEN_DASK_EXPECTED_WORKERS"
DASK_STARTUP_TIMEOUT_ENV = "MISEN_DASK_STARTUP_TIMEOUT"
DASK_CPUS_ENV = "MISEN_DASK_CPUS"
DASK_MEMORY_GIB_ENV = "MISEN_DASK_MEMORY_GIB"
DEFAULT_DASK_STARTUP_TIMEOUT = 600
MIN_DASK_WORKERS = 2


def managed_cluster_script(
    command: list[str],
    worker_launcher: list[str],
    *,
    environment: dict[str, str],
    workers: int,
    cpus: int,
    memory_gib: int,
    startup_timeout: int,
) -> str:
    """Wrap a command in a leader-managed fixed-size Dask cluster.

    ``worker_launcher`` must run the appended command once per allocation
    member, propagate its environment, remain attached while they run, and
    fail if any member exits.
    """
    if not command or not worker_launcher:
        msg = "Dask commands and worker launchers must be nonempty."
        raise ValueError(msg)
    for name, value in (
        ("workers", workers),
        ("cpus", cpus),
        ("memory_gib", memory_gib),
        ("startup_timeout", startup_timeout),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            msg = f"{name} must be a positive integer."
            raise ValueError(msg)
    if workers < MIN_DASK_WORKERS:
        msg = f"DASK_CLIENT requires at least {MIN_DASK_WORKERS} workers."
        raise ValueError(msg)
    command_array = shlex.join(command)
    launcher_array = shlex.join(worker_launcher)
    environment_exports = "\n        ".join(
        f"export {shlex.quote(f'{name}={value}')}" for name, value in environment.items()
    )
    return dedent(
        f"""\
        set -euo pipefail

        runtime_dir="$(mktemp -d "${{TMPDIR:-/tmp}}/misen-dask.XXXXXX")"
        scheduler_file="$runtime_dir/scheduler-address"
        scheduler_pid= worker_pid= coordinator_pid=

        cleanup() {{
            trap - EXIT INT TERM
            set +e
            kill $coordinator_pid $worker_pid $scheduler_pid 2>/dev/null
            wait $coordinator_pid $worker_pid $scheduler_pid 2>/dev/null
            rm -rf -- "$runtime_dir"
        }}
        trap cleanup EXIT
        trap 'exit 130' INT
        trap 'exit 143' TERM

        command=({command_array})
        worker_launcher=({launcher_array})
        {environment_exports}
        {DASK_ROLE_ENV}=scheduler {DASK_SCHEDULER_FILE_ENV}="$scheduler_file" "${{command[@]}}" &
        scheduler_pid=$!

        deadline=$((SECONDS + {startup_timeout}))
        while [[ ! -s "$scheduler_file" ]]; do
            if ! kill -0 "$scheduler_pid" 2>/dev/null; then
                printf 'misen: Dask scheduler exited before publishing its address\n' >&2
                exit 1
            fi
            if (( SECONDS >= deadline )); then
                printf 'misen: timed out waiting for the Dask scheduler after %s seconds\n' {startup_timeout} >&2
                exit 1
            fi
            sleep 0.1
        done
        scheduler_address="$(<"$scheduler_file")"
        export {DASK_SCHEDULER_ADDRESS_ENV}="$scheduler_address"
        export {DASK_STARTUP_TIMEOUT_ENV}={startup_timeout}

        {DASK_ROLE_ENV}=worker \\
            {DASK_CPUS_ENV}={cpus} \\
            {DASK_MEMORY_GIB_ENV}={memory_gib} \\
            "${{worker_launcher[@]}}" \\
            "${{command[@]}}" &
        worker_pid=$!

        {DASK_EXPECTED_WORKERS_ENV}={workers} \\
            "${{command[@]}}" &
        coordinator_pid=$!

        while kill -0 "$coordinator_pid" 2>/dev/null; do
            if ! kill -0 "$scheduler_pid" 2>/dev/null || ! kill -0 "$worker_pid" 2>/dev/null; then
                printf 'misen: Dask runtime exited while the work unit was running\n' >&2
                exit 1
            fi
            sleep 0.2
        done
        if ! kill -0 "$scheduler_pid" 2>/dev/null || ! kill -0 "$worker_pid" 2>/dev/null; then
            printf 'misen: Dask runtime exited while the work unit was running\n' >&2
            exit 1
        fi
        wait "$coordinator_pid"
        """
    )


def run_role_from_env() -> bool:
    """Run an executor-owned Dask role, returning whether it consumed the process.

    Scheduler and worker roles block until the enclosing executor launcher
    terminates them and return ``True`` if they close normally. With no role,
    the normal worker payload continues.
    """
    # The role selects this one outer worker invocation; it must not leak into
    # user code or subprocesses launched by the coordinator/Dask worker.
    role = os.environ.pop(DASK_ROLE_ENV, None)
    if role is None:
        return False
    if role == "scheduler":
        scheduler_file = _required_env(DASK_SCHEDULER_FILE_ENV)
        asyncio.run(_run_scheduler(Path(scheduler_file)))
        return True
    if role == "worker":
        address = _required_env(DASK_SCHEDULER_ADDRESS_ENV)
        asyncio.run(
            _run_worker(
                address,
                nthreads=positive_int_env(DASK_CPUS_ENV),
                memory_gib=positive_int_env(DASK_MEMORY_GIB_ENV),
            )
        )
        return True
    msg = f"Unsupported {DASK_ROLE_ENV} value: {role!r}."
    raise RuntimeError(msg)


async def _run_scheduler(scheduler_file: Path) -> None:
    """Run a scheduler on a dynamic port and publish its address atomically."""
    from distributed import Scheduler

    scheduler_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = scheduler_file.with_name(f".{scheduler_file.name}.{os.getpid()}.tmp")
    try:
        async with Scheduler(
            port=0,
            local_directory=str(scheduler_file.parent),
            dashboard_address=None,
            allowed_failures=0,
        ) as scheduler:
            temporary.write_text(scheduler.address)
            temporary.replace(scheduler_file)
            await scheduler.finished()
    finally:
        temporary.unlink(missing_ok=True)


async def _run_worker(address: str, *, nthreads: int, memory_gib: int) -> None:
    """Run one non-restarting Dask worker until its scheduler closes."""
    from distributed import Worker

    timeout = positive_int_env(DASK_STARTUP_TIMEOUT_ENV, default=DEFAULT_DASK_STARTUP_TIMEOUT)
    async with Worker(
        address,
        nthreads=nthreads,
        memory_limit=f"{memory_gib} GiB",
        death_timeout=timeout,
        dashboard_address=None,
    ) as worker:
        await worker.finished()


def _required_env(name: str) -> str:
    """Return one required nonempty environment value."""
    value = os.environ.get(name)
    if value:
        return value
    msg = f"{name} is required for this Dask runtime role."
    raise RuntimeError(msg)


def positive_int_env(name: str, *, default: int | None = None) -> int:
    """Parse a positive integer environment value."""
    raw = os.environ.get(name)
    if raw is None and default is not None:
        return default
    if raw is None:
        raw = _required_env(name)
    try:
        value = int(raw)
    except ValueError as exc:
        msg = f"{name} must be a positive integer, got {raw!r}."
        raise RuntimeError(msg) from exc
    if value < 1:
        msg = f"{name} must be a positive integer, got {raw!r}."
        raise RuntimeError(msg)
    return value
