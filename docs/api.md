# API Reference

The sections below document the stable user-facing surface.

Internal modules under `misen.utils.*` are implementation details and may
change without notice.

# Exceptions and failures

Misen uses built-in exceptions for ordinary Python contracts (`TypeError` for
an invalid argument type, `ValueError` for an invalid value, and `KeyError`
for a genuine mapping miss). Failures owned by a Misen subsystem use a
specific `MisenError` subclass so callers can handle them without depending on
the underlying storage, serializer, scheduler, or configuration library.

Exceptions raised by user task functions retain their original type and
traceback when execution remains in-process. Across a worker or scheduler
boundary, failed jobs expose structured failure information and raise
`JobFailedError` when the caller requests status enforcement.

Python's type system does not model checked exceptions, so Misen makes the
contract explicit through this hierarchy, public `Raises:` documentation, and
typed diagnostic attributes. In particular, `SubmissionError.submitted_jobs`
contains handles accepted before a later dispatch failed, and
`JobFailedError.failures` contains stable per-job facts suitable for a UI.

The command-line interface renders expected `MisenError` failures as concise
messages with a nonzero status. Set `MISEN_DEBUG=1` to include the complete
chained traceback. Unexpected exceptions retain Python's normal traceback.

::: misen.exceptions
    options:
      members:
        - ErrorCode
        - MisenError
        - CacheError
        - CliUsageError
        - ConfigError
        - HashError
        - LockUnavailableError
        - SerializationError
        - WorkspaceError
        - StorageError
        - SnapshotError
        - ExecutionError
        - SubmittedJob
        - SubmissionError
        - StatusQueryError
        - JobFailure
        - JobFailedError
        - ExperimentReferenceError

# Task

::: misen.tasks.Task
    options:
      members:
        - __init__
        - T
        - is_cached
        - are_deps_cached
        - done
        - is_running
        - submit
        - result
        - scratch_dir
        - with_resources

# @meta decorator

::: misen.task_metadata.meta

# Resources

::: misen.task_metadata.Resources

# Runtime sentinels

`SCRATCH_DIR` injects a per-task `pathlib.Path`. `DASK_CLIENT` injects an
allocation-scoped `distributed.Client` for supported multi-node executors.
Both are bound as top-level `Task(...)` arguments and excluded from task
identity.

# DiskWorkspace

::: misen.workspaces.disk.DiskWorkspace

# CloudWorkspace

`CloudWorkspace` stores results, locks, snapshots, payloads, and logs in S3,
GCS, or Azure Blob while keeping an expendable local cache. Remote workers
authenticate through their ambient environment or workload identity; the
generic `config` mapping cannot be embedded in worker bootstrap commands.

::: misen.workspaces.cloud.CloudWorkspace

# LocalExecutor

::: misen.executors.local.LocalExecutor

# InProcessExecutor

::: misen.executors.in_process.InProcessExecutor

# SlurmExecutor

::: misen.executors.slurm.SlurmExecutor
    options:
      members:
        - __init__

# SkyPilotExecutor

`SkyPilotExecutor` schedules ready WorkUnits on reusable cloud or cluster
allocations. `workers` is a required non-empty list of `SkyPilotWorker`
entries (`[[executor.workers]]` in TOML), with per-type and per-session
`max_workers` limits. The CLI accepts the same list as JSON through
`--executor.workers '[{"infra":"aws","cpus":4,"memory":16}]'`.
CPU, RAM, node count, accelerator type/count, and
per-device GPU memory determine eligibility. Single-node work shares declared
capacity; multi-node work reserves a complete group. Idle groups retire after
`idle_timeout_minutes`, or when another required type needs their slot.
`reuse_workers=true` shares workers and environments across graphs in the same
executor and workspace. `lookahead_seconds=90` enables bounded advance
provisioning; zero disables it. Close the executor before changing workspaces.
Preparation uses the worker's full declared CPU allocation. Prepared launch
commands reuse environments while keeping each WorkUnit in a fresh process.

```bash
uv pip install "misen[skypilot]" "skypilot-nightly[aws]>=1.0.0.dev20260905"
```

Install provider extras and configure credentials on the submitting host.
VM provisioning also requires `rsync` and SSH on the host's `PATH`.
Workers require SSH-accessible Linux nodes with Python 3 and `taskset`. The
runtime checks the declared clouds' credentials before provisioning workers.
Misen owns an isolated foreground local API server and a supervised graph
controller per executor session. No remote API server or managed controller VM is
needed. Existing SkyPilot endpoints, daemons, and runtime state remain separate.
`startup_timeout` bounds API startup and resource preflight (default 300 seconds).

Lifecycle progress uses `runtime_events` (`MISEN_RUNTIME_EVENTS=0` disables
console output). Job logs capture startup/provisioning context and complete
bootstrap/execution output from all ranks. The controller streams these logs to
the workspace and finalizes them before reporting terminal states. A shared
`<session>_skypilot.log` includes API and teardown diagnostics; original files live
under `<workspace temp>/skypilot/<session>/logs/`, including native SkyPilot logs
that would otherwise go to `~/sky_logs`.

The submitting process must remain alive. Use the executor as a context
manager or call `close()` to clean up unfinished work; normal process exit
also requests cleanup. SkyPilot autodown backs up abrupt shutdown after active
agents stop their subprocesses on connection loss or lease expiry. Job handles support individual
cancellation and local status queries; checkpoints and results remain durable in the workspace. Matching graphs reattach within
the same live executor; uncertain execution or cleanup after controller loss
blocks replay. Detached scheduling and automatic WorkUnit retries are not
supported.

Worker entries own `infra`, `instance_type`, `use_spot`, `image_id`, `disk_size`,
and `max_hourly_cost`. GPU memory comes from pinned catalog offerings;
`accelerator_memory` supplies fallback capacities in GiB/device without
inflating known values. Assigned device memory is checked before task startup.
See the [design contract](design_remote_executors.md#reusable-workers)
and README for complete configuration and scheduling details.

The adapter requires `snapshot = true`, `prewarm_envs = false`, a remotely
fetchable workspace such as `CloudWorkspace`, and a relative `cache_dir`.
Every worker needs independent object-store access. `DASK_CLIENT` creates a
private fixed-membership Dask cluster with one worker per node, a rank-zero
scheduler, and one task coordinator. The project must include `distributed`.
`dask_startup_timeout` bounds readiness; `dask_scheduler_port` (default 8786,
range 1024–65535) must be free and reachable on the allocation's trusted private
network. Without `DASK_CLIENT`, only rank zero executes the Misen payload.

::: misen.executors.skypilot.SkyPilotExecutor
    options:
      members:
        - __init__
