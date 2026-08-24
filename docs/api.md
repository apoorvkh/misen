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

`SkyPilotExecutor` is an optional remote adapter for SkyPilot 0.13 managed
jobs. Install Misen's provider-neutral extra into the environment that runs
Misen (rather than only as an isolated `uv tool`), compose it with the
upstream extras for the compute backends used by a local SkyPilot API server,
then verify provider access. A configured remote SkyPilot API server owns its
provider dependencies, so its Misen clients need only the base extra.
Misen tests this integration with SkyPilot 0.13 on Python 3.11–3.14; individual
provider extras may impose additional constraints.

```bash
uv pip install "misen[skypilot]" "skypilot[aws,gcp]>=0.13,<0.14"
# For example, instead target existing Kubernetes, SSH, and Slurm clusters:
uv pip install "misen[skypilot]" "skypilot[kubernetes,ssh,slurm]>=0.13,<0.14"
sky check
# From a Misen source checkout:
uv sync --extra skypilot
uv run --extra skypilot --with "skypilot[kubernetes,ssh,slurm]>=0.13,<0.14" sky check
```

`infra` accepts any compute infrastructure registered by the installed
SkyPilot version, including Azure, OCI, Lambda Cloud, RunPod, Kubernetes,
existing SSH machines, and Slurm. Install the corresponding named SkyPilot
extra rather than `skypilot[all]`; backend capabilities such as multi-node
support still vary. See [Installation and backend selection](design_remote_executors.md#installation-and-backend-selection)
for the provider matrix. Compute selection is independent of workspace
storage: every worker may use any supported `CloudWorkspace` object store it
can reach and authenticate to.

The implementation eagerly submits one managed job per pending work unit and
uses durable workspace markers to gate dependencies, so arbitrary DAGs are
supported and independent branches run in parallel. Descendants may provision
while waiting; failures before a worker can publish its marker propagate only
when the submitting Misen process observes SkyPilot status, or at the
dependent job's cumulative timeout. Multi-node requests use SkyPilot
`num_nodes`. A work unit that binds `DASK_CLIENT` gets one private Dask
scheduler on rank 0, one worker per node, and one rank-0 task coordinator;
without the sentinel, the Misen payload runs only on rank 0 and user code may
orchestrate the remaining nodes itself. The managed cluster has fixed
membership, uses the allocation's private TCP network, and waits up to
`dask_startup_timeout` for startup. `dask_scheduler_port` (default 8786) must
be in the range 1024–65535, free on rank 0, and reachable from every node; this
internal Dask connection is neither authenticated nor encrypted and therefore
requires a trusted network isolated from untrusted workloads. The project
environment must include `distributed`. The adapter requires worker-side snapshots
(`prewarm_envs = false`) and a remotely fetchable workspace such as
`CloudWorkspace` with a relative `cache_dir`. See the
[remote executor design](design_remote_executors.md) for the control/data-plane
contract and planned adapters.

::: misen.executors.skypilot.SkyPilotExecutor
    options:
      members:
        - __init__
