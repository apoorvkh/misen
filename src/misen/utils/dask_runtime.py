"""Private Dask runtime roles started by executor worker programs."""

from __future__ import annotations

import os
import shlex
import signal
from pathlib import Path
from textwrap import dedent

from misen.exceptions import ExecutionError

DASK_ROLE_ENV = "MISEN_DASK_ROLE"
DASK_SCHEDULER_ADDRESS_ENV = "MISEN_DASK_SCHEDULER_ADDRESS"
DASK_SCHEDULER_FILE_ENV = "MISEN_DASK_SCHEDULER_FILE"
DASK_SCHEDULER_HOST_ENV = "MISEN_DASK_SCHEDULER_HOST"
DASK_SCHEDULER_PORT_ENV = "MISEN_DASK_SCHEDULER_PORT"
DASK_EXPECTED_WORKERS_ENV = "MISEN_DASK_EXPECTED_WORKERS"
DASK_STARTUP_TIMEOUT_ENV = "MISEN_DASK_STARTUP_TIMEOUT"
DASK_CPUS_ENV = "MISEN_DASK_CPUS"
DASK_MEMORY_GIB_ENV = "MISEN_DASK_MEMORY_GIB"
DEFAULT_DASK_STARTUP_TIMEOUT = 600
DEFAULT_DASK_SCHEDULER_PORT = 8786
MIN_DASK_SCHEDULER_PORT = 1024
MAX_DASK_SCHEDULER_PORT = 65535
MIN_DASK_WORKERS = 2
_DASK_HTTP_ADDRESS = "127.0.0.1:0"  # fail closed if Dask bypasses the no-HTTP override
_DASK_SHUTDOWN_TIMEOUT = 30
_SHUTDOWN_SIGNALS = (signal.SIGINT, signal.SIGTERM)
_PREFLIGHT_ROLE = "preflight"


class _NoHttpServer:
    """Work around dask/distributed#8136 by disabling its HTTP service."""

    def start_http_server(self, *_args: object, **_kwargs: object) -> None:
        """Do not open an HTTP listener."""


def _environment_exports(environment: dict[str, str]) -> str:
    """Render literal Bash exports shared by each private runtime role."""
    return "\n        ".join(f"export {shlex.quote(f'{name}={value}')}" for name, value in environment.items())


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
    environment_exports = _environment_exports(environment)
    return dedent(
        f"""\
        set -euo pipefail

        runtime_dir="$(mktemp -d "${{TMPDIR:-/tmp}}/misen-dask.XXXXXX")"
        scheduler_file="$runtime_dir/scheduler-address"
        scheduler_pid= worker_pid= preflight_pid= coordinator_pid=

        cleanup() {{
            trap - EXIT INT TERM
            set +e
            kill $coordinator_pid $preflight_pid $worker_pid $scheduler_pid 2>/dev/null
            wait $coordinator_pid $preflight_pid $worker_pid $scheduler_pid 2>/dev/null
            rm -rf -- "$runtime_dir"
        }}
        trap cleanup EXIT
        trap 'exit 130' INT
        trap 'exit 143' TERM

        command=({shlex.join(command)})
        worker_launcher=({shlex.join(worker_launcher)})
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

        # Do not let a cache hit or idempotent retry finish the coordinator
        # before independently bootstrapped workers have joined.
        {DASK_ROLE_ENV}={_PREFLIGHT_ROLE} \\
            {DASK_EXPECTED_WORKERS_ENV}={workers} \\
            "${{command[@]}}" &
        preflight_pid=$!

        runtime_alive() {{
            kill -0 "$scheduler_pid" 2>/dev/null && kill -0 "$worker_pid" 2>/dev/null
        }}
        while kill -0 "$preflight_pid" 2>/dev/null && runtime_alive; do
            sleep 0.2
        done
        if ! runtime_alive; then
            printf 'misen: Dask runtime exited before every worker joined\n' >&2
            exit 1
        fi
        preflight_status=0
        wait "$preflight_pid" || preflight_status=$?
        preflight_pid=
        if (( preflight_status != 0 )); then
            printf 'misen: Dask workers did not form the requested cluster\n' >&2
            exit "$preflight_status"
        fi

        {DASK_EXPECTED_WORKERS_ENV}={workers} \\
            "${{command[@]}}" &
        coordinator_pid=$!

        while kill -0 "$coordinator_pid" 2>/dev/null && runtime_alive; do
            sleep 0.2
        done
        if ! runtime_alive; then
            printf 'misen: Dask runtime exited while the work unit was running\n' >&2
            exit 1
        fi
        coordinator_status=0
        wait "$coordinator_pid" || coordinator_status=$?
        coordinator_pid=

        # Closing the scheduler asks every worker to exit cleanly, allowing
        # the attached launcher (for example, srun) to finish normally.
        kill "$scheduler_pid" 2>/dev/null || true
        runtime_status=0
        wait "$scheduler_pid" || runtime_status=$?
        scheduler_pid=
        wait "$worker_pid" || runtime_status=$?
        worker_pid=

        if (( runtime_status != 0 )); then
            printf 'misen: Dask runtime failed during shutdown\n' >&2
            exit 1
        fi
        exit "$coordinator_status"
        """
    )


def managed_ranked_cluster_script(
    command: list[str],
    *,
    environment: dict[str, str],
    workers: int,
    cpus: int,
    memory_gib: int,
    startup_timeout: int,
    node_rank_env: str,
    node_ips_env: str,
    scheduler_port: int = DEFAULT_DASK_SCHEDULER_PORT,
) -> str:
    """Run a fixed Dask cluster through one scheduler command per node.

    Distributed launchers such as SkyPilot invoke the same command on every
    allocation member and expose a zero-based rank plus an ordered IP list.
    Rank zero reuses :func:`managed_cluster_script` for the scheduler, one
    local worker, and the Misen coordinator. Every other rank runs one worker
    in the foreground. The first node IP and fixed private port let those
    independently launched workers find the scheduler without shared disk.
    """
    if not node_rank_env or not node_ips_env:
        msg = "Dask node rank and IP environment names must be nonempty."
        raise ValueError(msg)
    if (
        isinstance(scheduler_port, bool)
        or not isinstance(scheduler_port, int)
        or not MIN_DASK_SCHEDULER_PORT <= scheduler_port <= MAX_DASK_SCHEDULER_PORT
    ):
        msg = "scheduler_port must be an integer between 1024 and 65535."
        raise ValueError(msg)

    head_script = managed_cluster_script(
        command,
        ["bash", "-c", 'exec "$@"', "_"],
        environment=environment,
        workers=workers,
        cpus=cpus,
        memory_gib=memory_gib,
        startup_timeout=startup_timeout,
    )
    environment_exports = _environment_exports(environment)
    ranked_prefix = dedent(
        f"""\
        set -euo pipefail

        command=({shlex.join(command)})
        {environment_exports}
        node_rank_name={shlex.quote(node_rank_env)}
        node_ips_name={shlex.quote(node_ips_env)}
        node_rank="${{{node_rank_env}:-}}"
        node_ip_text="${{{node_ips_env}:-}}"
        if [[ ! "$node_rank" =~ ^[0-9]+$ ]]; then
            printf 'misen: %s must be a non-negative integer, got %q\n' "$node_rank_name" "$node_rank" >&2
            exit 1
        fi
        node_rank=$((10#$node_rank))
        node_ips=()
        while IFS= read -r node_ip; do
            [[ -n "$node_ip" ]] && node_ips+=("$node_ip")
        done <<< "$node_ip_text"
        found_nodes="${{#node_ips[@]}}"
        if (( found_nodes != {workers} )); then
            printf 'misen: %s expected %s node IPs; found %s\n' "$node_ips_name" {workers} "$found_nodes" >&2
            exit 1
        fi
        if (( node_rank >= {workers} )); then
            printf 'misen: %s rank %s is outside the %s-node allocation\n' "$node_rank_name" "$node_rank" {workers} >&2
            exit 1
        fi

        head_ip="${{node_ips[0]}}"
        if [[ "$head_ip" == *:* ]]; then
            scheduler_address="tcp://[$head_ip]:{scheduler_port}"
        else
            scheduler_address="tcp://$head_ip:{scheduler_port}"
        fi
        export {DASK_SCHEDULER_HOST_ENV}="$head_ip"
        export {DASK_SCHEDULER_PORT_ENV}={scheduler_port}
        export {DASK_STARTUP_TIMEOUT_ENV}={startup_timeout}

        if (( node_rank != 0 )); then
            export {DASK_ROLE_ENV}=worker
            export {DASK_SCHEDULER_ADDRESS_ENV}="$scheduler_address"
            export {DASK_CPUS_ENV}={cpus}
            export {DASK_MEMORY_GIB_ENV}={memory_gib}
            exec "${{command[@]}}"
        fi
        """
    )
    return f"{ranked_prefix}\n\n{head_script}"


def run_role_from_env() -> bool:
    """Run an executor-owned Dask role, returning whether it consumed the process.

    Scheduler and worker roles block until the enclosing executor launcher
    terminates them; the preflight role waits for exact worker membership.
    Each returns ``True`` when it consumes the process. With no role, the
    normal worker payload continues.
    """
    # The role selects this one outer worker invocation; it must not leak into
    # user code or subprocesses launched by the coordinator/Dask worker.
    role = os.environ.pop(DASK_ROLE_ENV, None)
    if role is None:
        return False
    import asyncio

    try:
        if role == "scheduler":
            raw_port = os.environ.get(DASK_SCHEDULER_PORT_ENV)
            port = positive_int_env(DASK_SCHEDULER_PORT_ENV) if raw_port is not None else 0
            if port > MAX_DASK_SCHEDULER_PORT:
                msg = f"{DASK_SCHEDULER_PORT_ENV} must not exceed {MAX_DASK_SCHEDULER_PORT}, got {port!r}."
                raise ExecutionError(msg)
            run = _run_scheduler(
                Path(_required_env(DASK_SCHEDULER_FILE_ENV)),
                host=os.environ.get(DASK_SCHEDULER_HOST_ENV) or None,
                port=port,
            )
            asyncio.run(run)
        elif role == "worker":
            run = _run_worker(
                _required_env(DASK_SCHEDULER_ADDRESS_ENV),
                nthreads=positive_int_env(DASK_CPUS_ENV),
                memory_gib=positive_int_env(DASK_MEMORY_GIB_ENV),
            )
            asyncio.run(run)
        elif role == _PREFLIGHT_ROLE:
            _wait_for_workers(
                _required_env(DASK_SCHEDULER_ADDRESS_ENV),
                workers=positive_int_env(DASK_EXPECTED_WORKERS_ENV),
                timeout=positive_int_env(DASK_STARTUP_TIMEOUT_ENV, default=DEFAULT_DASK_STARTUP_TIMEOUT),
            )
        else:
            msg = f"Unsupported {DASK_ROLE_ENV} value: {role!r}."
            raise ExecutionError(msg)
    except OSError as exc:
        msg = f"The executor-owned Dask {role} runtime failed: {exc}"
        raise ExecutionError(msg) from exc
    return True


def _wait_for_workers(address: str, *, workers: int, timeout: int) -> None:
    """Require the complete fixed worker group before user code can start."""
    from distributed import Client

    client = Client(address, set_as_default=False, timeout=timeout)
    primary: BaseException | None = None
    try:
        client.wait_for_workers(workers, timeout=timeout)
        actual = len(client.scheduler_info()["workers"])
        if actual != workers:
            msg = f"Dask runtime expected exactly {workers} worker(s), found {actual}."
            primary = ExecutionError(msg)
    except BaseException as exc:  # noqa: BLE001 -- close before propagating interrupts
        primary = exc
    try:
        client.close()
    except BaseException as exc:
        if primary is None:
            raise
        primary.add_note(f"Additionally, closing the Dask preflight client failed: {type(exc).__name__}: {exc}")
    if primary is not None:
        raise primary


async def _run_scheduler(scheduler_file: Path, *, host: str | None = None, port: int = 0) -> None:
    """Run a scheduler on the requested interface and publish its address."""
    import asyncio

    import dask
    from distributed import Scheduler as DaskScheduler

    class Scheduler(_NoHttpServer, DaskScheduler):
        """Dask scheduler with no HTTP service."""

    scheduler_file.parent.mkdir(parents=True, exist_ok=True)
    with dask.config.set({"distributed.scheduler.http.routes": []}):
        scheduler = Scheduler(
            host=host,
            port=port,
            local_directory=str(scheduler_file.parent),
            dashboard=False,
            dashboard_address=_DASK_HTTP_ADDRESS,
            allowed_failures=0,
        )
    temporary = scheduler_file.with_name(f".{scheduler_file.name}.{os.getpid()}.tmp")
    try:
        async with scheduler:
            loop = asyncio.get_running_loop()
            shutdown = asyncio.Event()
            for sig in _SHUTDOWN_SIGNALS:
                loop.add_signal_handler(sig, shutdown.set)

            async def close_on_signal() -> None:
                await shutdown.wait()
                for address in list(scheduler.workers):
                    scheduler.close_worker(address)
                try:
                    async with asyncio.timeout(_DASK_SHUTDOWN_TIMEOUT):
                        # Scheduler exposes no awaitable worker-removal event.
                        while scheduler.workers:  # noqa: ASYNC110
                            await asyncio.sleep(0.05)
                except TimeoutError:
                    pass
                await scheduler.close(reason="executor shutdown")

            shutdown_task = asyncio.create_task(close_on_signal())
            finished_task = asyncio.create_task(scheduler.finished())
            try:
                temporary.write_text(scheduler.address)
                temporary.replace(scheduler_file)
                done, _ = await asyncio.wait(
                    (shutdown_task, finished_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                await asyncio.gather(*done)
                if shutdown.is_set():
                    await shutdown_task
            finally:
                shutdown_task.cancel()
                finished_task.cancel()
                await asyncio.gather(shutdown_task, finished_task, return_exceptions=True)
                for sig in _SHUTDOWN_SIGNALS:
                    loop.remove_signal_handler(sig)
    finally:
        temporary.unlink(missing_ok=True)


async def _run_worker(address: str, *, nthreads: int, memory_gib: int) -> None:
    """Run one non-restarting Dask worker until its scheduler closes."""
    import dask
    from distributed import Worker as DaskWorker

    class Worker(_NoHttpServer, DaskWorker):
        """Dask worker with no HTTP service."""

    timeout = positive_int_env(DASK_STARTUP_TIMEOUT_ENV, default=DEFAULT_DASK_STARTUP_TIMEOUT)
    with dask.config.set({"distributed.worker.http.routes": []}):
        async with Worker(
            address,
            nthreads=nthreads,
            memory_limit=f"{memory_gib} GiB",
            death_timeout=timeout,
            dashboard=False,
            dashboard_address=_DASK_HTTP_ADDRESS,
        ) as worker:
            await worker.finished()


def _required_env(name: str) -> str:
    """Return one required nonempty environment value."""
    if not (value := os.environ.get(name)):
        msg = f"{name} is required for this Dask runtime role."
        raise ExecutionError(msg)
    return value


def positive_int_env(name: str, *, default: int | None = None) -> int:
    """Parse a positive integer environment value."""
    raw = os.environ.get(name)
    if raw is None and default is not None:
        return default
    if raw is None:
        raw = _required_env(name)
    msg = f"{name} must be a positive integer, got {raw!r}."
    try:
        value = int(raw)
    except ValueError as exc:
        raise ExecutionError(msg) from exc
    if value < 1:
        raise ExecutionError(msg)
    return value
