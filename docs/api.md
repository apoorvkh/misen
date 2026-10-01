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

# SSHExecutor

`SSHExecutor` uses AsyncSSH to run snapshot-pinned jobs on existing Linux hosts.
Install `misen[ssh]` on the submitting machine. Configuration, alias resolution,
and core imports do not load AsyncSSH until execution is requested.

```python
from misen.executors.ssh import SSHExecutor, SSHWorker
from misen.workspaces.disk import DiskWorkspace

workspace = DiskWorkspace(directory="/shared/research/misen")
with SSHExecutor(workers=[SSHWorker(hosts=["research-node"], cpus=8, memory=32)]) as executor:
    jobs = executor.submit({task}, workspace, blocking=True)
```

The alias is `[executor] type = "ssh"`, with `[[executor.workers]]` entries.
The CLI accepts the same list through
`--executor ssh --executor.workers '[{"hosts":["research-node"],"cpus":8,"memory":32}]'`.
The [README](../README.md#remote-execution-with-ssh-optional) covers SSH keys,
resource budgets, workspace access, and lifecycle requirements.

A worker group's `hosts` must match the task's node count exactly. `cpus`,
`memory`, `accelerators`, and `accelerator_memory` are per-node capacities;
`accelerator_memory` is GiB per device. `accelerator_type` accepts `cuda`,
`rocm`, or `xpu`. `accelerator_indices` optionally selects the physical devices.
Single-node jobs share capacity; multi-node jobs occupy the complete group.
`max_concurrent_jobs` optionally caps the number of jobs reserved on a worker,
including preparation. It must be a positive integer; the default `None` adds
no limit beyond the resource budgets. Jobs waiting for this limit stay pending.
Groups must not overlap, even through different SSH aliases.
`addresses` optionally supplies a node-to-node hostname/IP for each host.

`ssh_config` selects an SSH config file; by default AsyncSSH reads the user's
SSH config. `known_hosts` selects a host-key file; omission keeps AsyncSSH's
verification defaults. Host-key verification is never disabled by the executor.
Authentication uses SSH config, local keys, or the local agent; agent forwarding
is disabled. SSH config may supply a proxy command or jump host.

Use a shared `DiskWorkspace` mounted at the same absolute path everywhere, or
`CloudWorkspace` with a relative cache directory and ambient credentials on
every host. `InMemoryWorkspace` is rejected before snapshot staging.
The SSH executor requires `snapshot=true` and `prewarm_envs=false`.
It reuses snapshot environments and prepared launch commands, with a fresh
process for each WorkUnit. The shared workspace retains results and logs.

`connect_timeout` defaults to 30 seconds, and `startup_timeout` to 600 seconds.
Task `time` covers execution after preparation. For `DASK_CLIENT`,
`dask_startup_timeout` bounds Dask readiness (default 600 seconds), and
`dask_scheduler_port` defaults to 8786 (valid range 1024–65535). The first
host runs the scheduler and sole task coordinator; each host runs one Dask
worker. The nodes need a trusted private network and a free, reachable
scheduler port. Without `DASK_CLIENT`, only rank zero executes the payload;
task code can inspect `MISEN_NODE_RANK` and `MISEN_NODE_IPS`.

The public executor and job APIs are synchronous. One background asyncio loop
handles SSH connections, concurrent streams, and scheduling. Job status is
local. Each host has a reusable connection pool, with up to eight jobs sharing
each connection. More connections open as needed for admitted jobs; SSH channel
counts do not impose an eight-job limit on the host. New connection handshakes
are serialized per host. Log writes are
batched and flushed every 200 ms or when a batch reaches roughly 256 KiB.
The TUI resolves and reads log storage outside its UI loop. Job states are
`pending`, `starting`, `running`, then `done` or `failed`. Logs are
finalized before terminal status is published. Cancelling a prerequisite fails
its dependents; cancelling one job does not cancel independent work.

Keep the submitting process alive. Context exit or `close()` cancels outstanding
jobs and awaits cleanup. Remote supervisors stop their process groups after
EOF, cancellation, time limits, or expiry of the 15-second controller lease,
with a 5-second termination grace period. These are cooperative process-group
controls; tasks must not deliberately detach descendants. Connection loss
never triggers replay. Uncertain cleanup disables the worker group for that
session. Scheduling budgets do not coordinate with other submitting processes.
Machines are not provisioned, rebooted, or terminated.

::: misen.executors.ssh.SSHExecutor

::: misen.executors.ssh.SSHWorker

::: misen.executors.ssh.SSHJob
