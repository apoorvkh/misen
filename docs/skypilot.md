# SkyPilot graph execution

SkyPilot manages compute allocations; Misen schedules ready work units inside
them. A reusable worker agent executes many work units without another
SkyPilot submission for each one. Cache boundaries, shared prerequisites,
task logs, and result identities remain unchanged.

This page describes the current implementation. It has local and fake-backend
coverage, but this graph architecture has **not yet been rerun on AWS**. The
previous cold/warm smoke timings measured the older per-work-unit managed-job
adapter, not the implementation described here.

## Install and authenticate

Install SkyPilot in the same environment that runs Misen, not only through an
isolated `uv tool`. The default `manage_api_server = true` starts an isolated
local API for the attached session and requires the pinned nightly:

```bash
uv pip install "misen[skypilot-managed]" "skypilot-nightly[aws]==1.0.0.dev20260905"

# For development in this repository instead:
uv sync --extra skypilot-managed
uv pip install "skypilot-nightly[aws]==1.0.0.dev20260905"
```

Install only the provider extras needed by the API server that provisions
compute. Do not install both `skypilot` and `skypilot-nightly`. With an already
configured shared/remote API, set `manage_api_server = false`; the ordinary
`misen[skypilot]` extra declares `skypilot>=0.13`. Feature support, especially
pools and remote credential injection, depends on the installed client/server
versions. A remote API server owns its provider packages and credentials.

Compute credentials and workspace credentials are separate. The submitter,
worker agents, task subprocesses, and any remote coordinator need access to the
workspace. Prefer an instance role or equivalent workload identity scoped to
the required bucket/prefix. Do not put credentials in configuration examples,
shell bootstrap text, allocation records, or logs.

For an S3 workspace, SkyPilot reading `~/.aws/credentials` does not necessarily
configure the object-store client. Export the selected profile into the
submitting process before creating the workspace; for example, with AWS CLI v2:

```bash
eval "$(aws configure export-credentials --profile default --format env)"
```

This does not provision worker-side bucket access. Workers need their own
ambient credentials. `CloudWorkspace.config` is not a way to embed credentials
in remote bootstrap commands. The submit host needs `rsync` available where
required by SkyPilot; worker images need Bash and GNU `timeout`, plus access
to the project's package sources unless their tools/caches are preinstalled.

The project's installable Misen dependency must include this implementation on
the remote worker too. Editing only the submitter checkout while keeping an
older remote Git/package pin is insufficient; update the project lockfile when
deploying a new Misen revision.

## Declare bounded capacity

The executor requires explicit capacity profiles. Each profile selects exactly
one source:

| Source | How Misen reserves it | Ownership |
| --- | --- | --- |
| `cluster` | `sky.exec()` starts an agent or dedicated task on that existing cluster | Borrowed; never resized or terminated |
| `pool` | One managed job reserves a compatible pool worker | Borrowed; never created, resized, or terminated automatically |
| `infra` | One managed job provisions the declared resource shape | Run-owned managed allocation; SkyPilot controls its infrastructure lifecycle |

An existing cluster profile requires `max_workers = 1`. For pool or `infra`
profiles, `max_workers` limits simultaneous allocations from that profile.
One reusable agent executes one task subprocess at a time. This is a fixed
bounded fleet, not a demand-driven autoscaler or a multi-slot worker scheduler.

For a small CPU workload, use an explicitly created pool:

```toml
[executor]
type = "skypilot"
lifecycle = "attached"
manage_api_server = true
api_server_namespace = "dev"
max_run_minutes = 60
setup_timeout_s = 600.0
shutdown_timeout_s = 30.0
poll_interval_s = 0.2

[executor.capacity.cpu]
pool = "misen-dev"
cpus = 2
memory = 4
max_workers = 1

[workspace]
type = "cloud"
backend = "s3"
bucket = "my-misen-workspace"
prefix = "experiments"
s3_region = "us-east-1"
cache_dir = ".cache/misen"
```

The workspace must support remotely fetchable snapshots/payloads and job-file
coordination. Use a relative `cache_dir`, `snapshot = true`, and
`prewarm_envs = false`; these are required, not optional optimizations.

Profiles specify CPU, memory, and accelerator quantities **per node**. They
must match the real borrowed capacity; setting `cpus` does not resize a VM.
SkyPilot enforces its allocation/resource-fitting rules. Misen's accounting
and environment visibility controls are not hard memory/device security
isolation for arbitrary user code.

Add a GPU profile for training branches:

```toml
[executor.capacity.gpu]
infra = "aws/us-east-1"
cpus = 8
memory = 32
accelerators = { L4 = 1 }
accelerator_type = "cuda"
accelerator_memory = 24
max_workers = 2
```

This profile authorizes up to two run-owned GPU allocations. Accelerator
models are concrete SkyPilot names; `accelerator_type` describes Misen's
programming backend, and `accelerator_memory` is the declared per-device GiB
capacity. At most one model appears in a profile. Unknown accelerator memory
does not satisfy a task's explicit minimum.

Graph execution currently supports `cuda`, `rocm`, and `xpu` device-mask
controls. Accelerator work requesting `tpu` or `mps` is rejected before
allocation. Worker masks must identify the actual reserved devices; Misen
does not guess device IDs when that scheduler information is missing.

An `infra` profile can provide an ordered list of infrastructure alternatives.
Creation options such as `instance_type`, `image_id`, `disk_size`, and
`use_spot` belong only on run-owned `infra` profiles; they are rejected for
borrowed clusters/pools. Provider-specific options must work for every chosen
alternative. Compute infrastructure and workspace backend are independent.

Tasks are routed to a compatible profile automatically, preferring smaller
compatible reservations and avoiding accelerator capacity for CPU work when
a suitable CPU profile exists. An unfit task fails preflight rather than
causing unbounded provisioning. Readiness follows actual graph edges: CPU
analysis can start as soon as its own GPU training parent commits, without
waiting for unrelated branches.

### Create a pool in an isolated namespace

The pool must exist in the same SkyPilot namespace/API identity that Misen
uses. Ordinary `sky check` and `sky jobs pool apply` do not configure the
isolated namespace below.

```yaml
# misen-pool.yaml
pool:
  workers: 1
resources:
  infra: aws/us-east-1
  cpus: 2+
  memory: 4+
```

```python
from misen.executors.skypilot import SkyPilotCapacity, SkyPilotExecutor

executor = SkyPilotExecutor(
    capacity={"cpu": SkyPilotCapacity(pool="misen-dev", cpus=2, memory=4)},
    api_server_namespace="dev",
)
with executor.session() as session:
    session.check(["aws"], verbose=True)  # enables access; does not provision
    session.pool_apply("misen-dev", "misen-pool.yaml")  # billable provisioning
    print(session.pool_status())

# Explicit cleanup after every user of this pool has finished:
with executor.session() as session:
    session.pool_down("misen-dev")
```

These session helpers are available for Misen-managed local API sessions.
With `manage_api_server = false`, manage capacity through the configured
SkyPilot service instead. Pools remain billable after a Misen session ends;
consult [SkyPilot's pool lifecycle](https://docs.skypilot.ai/en/latest/examples/pools.html).

## Attached runs and local-service cleanup

`lifecycle = "attached"` is the default. Its graph coordinator runs in the
invoking Python process, independently of job-state/UI polling. CLI runs and
`Experiment.run()` wait for completion by default. Blocking submissions open
their own session:

```python
jobs = task.submit(executor=executor, workspace=workspace, blocking=True)
# Or, for an Experiment instance:
experiment.run(executor=executor, workspace=workspace)
```

For nonblocking submissions, including `experiment.run(..., blocking=False)`,
the caller must keep the session alive:

```python
with executor.session():
    jobs = task.submit(executor=executor, workspace=workspace)
    for job in jobs.nodes():
        job.wait()
        job.raise_for_status()
```

Leaving the session stops admitting work, requests cancellation of unfinished
owned attempts, and tries to stop its agents/native jobs within the configured
shutdown grace. Unresolved outcomes/cleanup are reported, not silently treated
as successful teardown. A session exit does not detach the remaining graph.

The isolated local API has its own configuration, identity, runtime directory,
and ports under `$XDG_STATE_HOME/misen/skypilot/<namespace>` (normally
`~/.local/state/misen/skypilot/<namespace>`). Sessions in the same namespace
share the API until their last lease closes; other namespaces and ordinary
SkyPilot processes are independent. The parent environment and `HOME` are
unchanged. Keep namespace state until its cloud resources are reconciled.

The guardian stops the owned local API tree after client death. Remote agents
have finite coordinator leases and run lifetimes, but local process cleanup
does not prove cloud teardown. Cancellation targets only this run's native
jobs. Borrowed capacity is left running, and shared managed-job controllers
are not automatically deleted; they can remain billable. Verify instances,
disks, pools, and controllers separately when assessing cleanup or cost.

Each reusable attempt also has a small independent process guard. If its agent
dies, the guard detects pipe closure and terminates the task's process group,
including ordinary child processes. This is not sandbox containment for user
code that deliberately daemonizes into another session, or a guarantee against
the guard itself being killed with SIGKILL.

## Detached runs

Detached graph scheduling needs a remote Misen coordinator; it cannot be
implemented by submitting a few roots and closing the laptop. Configure all
of the following before using `lifecycle = "detached"`:

1. A stable, reachable remote SkyPilot API, already authenticated by the client.
2. Service accounts/credential injection enabled on that API server, with API
   protocol version at least 42 (distinct from the SDK package version).
3. A compatible SkyPilot SDK declared in the project's snapshotted dependencies
   and lockfile, not only installed in the submitter's environment.
4. A dedicated, run-owned, single-node coordinator profile, and workspace access
   from both the coordinator and task workers.

```toml
[executor]
type = "skypilot"
lifecycle = "detached"
manage_api_server = false
max_run_minutes = 60

[executor.coordinator]
infra = "aws/us-east-1"
dedicated = true
nodes = 1
cpus = 2
memory = 4

[executor.capacity.cpu]
pool = "existing-cpu-pool"
cpus = 2
memory = 4
max_workers = 2
```

Use the same CloudWorkspace configuration as above. The configured remote
SkyPilot identity must see `existing-cpu-pool`. Detached submission waits for
a durable native coordinator-job acknowledgement before returning; it does
not wait for the whole graph. The remote run has a finite deadline and owns
its agents and cleanup.

Only the remote coordinator requests `api_server_access = true`; ordinary
workers do not. SkyPilot can inject the endpoint and an expiring API token into
that coordinator. Injection is skipped when service accounts are disabled, so
a configured remote URL alone is insufficient. Follow the upstream
[nested-job credential requirements](https://docs.skypilot.ai/en/stable/examples/managed-jobs.html#calling-skypilot-api-from-within-managed-jobs)
and [API authentication guide](https://docs.skypilot.ai/en/latest/reference/auth.html).
Never copy the token into the Misen manifest or checked-in configuration.

### Observe or cancel an existing run

Pending logical handles expose `job.run_id`. Save that ID after submission;
fully cached submissions have no new remote run. Later:

```python
jobs = executor.attach(run_id, workspace)
for job in jobs.nodes():
    print(job.label, job.state())
# Cancel one logical job and its descendants:
next(iter(jobs.nodes())).cancel()
```

`attach()` reads a trusted manifest and serialized work-unit definitions.
Never attach to an untrusted workspace. It reconstructs handles without
resubmitting jobs, starting an API service, or taking over a coordinator.
Cancellation records are acted on by a live coordinator; if it is gone, inspect
durable allocation records and reconcile native jobs explicitly. An expired
coordinator lease is uncertainty, not evidence that remote code stopped.

## Multi-node and dedicated work

Set `dedicated = true` when a work unit needs its own allocation. Multi-node
profiles require it:

```toml
[executor.capacity.distributed]
infra = "aws/us-east-1"
dedicated = true
nodes = 2
cpus = 4
memory = 16
max_workers = 1
```

SkyPilot reserves the entire profile for each admitted dedicated work unit.
With `DASK_CLIENT`, each allocation has a temporary Dask scheduler/coordinator
on rank 0 and one worker per node. Live Dask runtimes are never shared across
work units. Without the sentinel, Misen invokes the payload only on rank 0;
user code owns any additional-node orchestration.

The current SkyPilot Dask wrapper uses a 300-second startup grace and private
port 8786, not the removed top-level Dask configuration fields. Every node
must reach that port on a trusted network: allocation-internal Dask TCP is
neither authenticated nor encrypted. The project must include `distributed`.
Worker identity/package access must work on every node.

## Transport, recovery, and current limits

- Agents and the coordinator exchange bounded JSON records through known
  workspace keys. This implementation polls mailboxes (default 0.2 seconds);
  it does not use bucket-wide listing, SSH-forwarded task RPC, or push events.
- Durable execution/result markers release descendants without waiting for
  SkyPilot allocation-health polling. Fresh subprocesses use cached project
  environments; a persistent user-code interpreter is not shared between tasks.
- Execution claims prevent deliberate replay of an uncertain attempt. A valid
  committed success can be reconciled; a lost/failed/incomplete attempt is not
  automatically retried. There is no coordinator takeover, dynamic replacement,
  speculative execution, or cross-run agent sharing.
- Logical deduplication is within a run. Separate runs retain existing
  cache-lock protection, but do not share live logical jobs; non-cacheable
  work can execute independently in each run. Arbitrary external side effects
  are not exactly-once.
- Profile limits and run deadlines bound scheduling, not cloud invoices.
  Controller, storage, idle pool, and delayed-teardown costs remain relevant.
  A 0.2-second polling interval is not a latency guarantee; object-store access,
  environment setup, Python startup, and remote provisioning still take time.

See the [architecture plan](design_skypilot_graph_execution.md) for the
implementation boundaries, remaining optimization work, and benchmark gates.

## Migrating older SkyPilot configuration

There is no separate eager/worker `mode` switch. Move old top-level `infra`,
`pool`, resource-model mappings, and creation settings into explicit
`[executor.capacity.<name>]` profiles. The old per-work-unit submission
strategy, dependency-waiting workers, and top-level `job_recovery` policy are
not the current public executor contract. Keep existing workspace/cache data;
do not treat submitting a fresh graph as recovery of an uncertain earlier run.
