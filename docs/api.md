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

`SkyPilotExecutor` schedules ready work units over explicit
`capacity: dict[str, SkyPilotCapacity]` profiles. SkyPilot owns allocations;
Misen owns graph readiness, logical jobs, attempt records, and completion.
Reusable agents execute one fresh task subprocess at a time. Dedicated
profiles provide one allocation per admitted work unit.

See the [SkyPilot usage guide](skypilot.md) for installation, complete TOML
examples, provider/workspace authentication, and the current validation limits.

| Setting | Contract |
| --- | --- |
| `capacity` | Named profiles selecting exactly one `pool`, `cluster`, or `infra`; every work unit must fit one |
| `lifecycle` | `"attached"` by default; `"detached"` requires explicit remote coordination |
| `manage_api_server` | `True` by default; isolated local API lifecycle on the supported pinned nightly |
| `api_server_namespace` | Persistent isolated API identity/state; defaults to `"default"` |
| `max_run_minutes` | Finite graph/agent lifetime; defaults to 1440 |
| `setup_timeout_s` | Provisioning/bootstrap allowance; defaults to 600 |
| `shutdown_timeout_s` | Bounded cleanup grace; defaults to 30 |
| `poll_interval_s` | Workspace mailbox polling interval; defaults to 0.2 |

Attached `submit(..., blocking=False)` requires a live
`with executor.session():` context. `submit(..., blocking=True)` scopes its own
session. `Experiment.run()` waits by default for attached graph execution.
Closing an unfinished session cancels/stops owned work and attempts bounded
cleanup; it does not detach the graph or terminate borrowed pools/clusters.

Detached execution requires `manage_api_server=False`, a stable remote API
with service-account credential injection enabled, a compatible SDK in the
project's snapshotted dependencies, and a dedicated run-owned single-node
`coordinator` profile. Submission waits for durable remote acknowledgement.

`attach(run_id, workspace)` reads a trusted run manifest and reconstructs
observing/cancelling job handles. It does not resubmit, retry, or take over a
lost coordinator. Cancellation requires a live coordinator to act on its
workspace request. Uncertain execution remains uncertain until reconciled.

There is no current eager/worker mode switch and no top-level `infra` or
`pool` setting. Existing low-level managed-job records are not a substitute
for new run/attempt/allocation identities. Shared managed-job controllers may
remain billable after run cleanup.

::: misen.executors.skypilot.SkyPilotExecutor
    options:
      inherited_members: true
      members:
        - session
        - submit
        - attach

# SkyPilotCapacity

A profile reserves CPU, memory, and accelerator resources per node.
`max_workers` bounds active allocations, not individual DAG nodes. Borrowed
clusters require `max_workers=1`; multi-node profiles require
`dedicated=True`. Model names and declared accelerator memory must describe
the actual available hardware. Creation-only options are rejected for
borrowed pools/clusters.

::: misen.executors.skypilot.SkyPilotCapacity
    options:
      members:
        - fits
