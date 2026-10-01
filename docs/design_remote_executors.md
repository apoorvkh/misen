# Remote execution over SSH

The implemented remote adapter is `SSHExecutor`, using AsyncSSH to connect to
existing Linux hosts. Cloud provisioning adapters are deferred. Compute and
storage remain separate: SSH transports control and process output, while
Workspace owns snapshots, submission payloads, results, locks, and task logs.
Historical cloud benchmarks in `docs/benchmarks/` describe removed prototypes.

## Configuration and placement

A worker entry describes a host or a fixed multi-node group and its per-node
CPU, RAM, and accelerator capacity. A task must fit one complete group. Ready
single-node jobs can share its CPU, memory, and whole-device GPU budgets;
multi-node jobs reserve their group exclusively. CPU-only work prefers groups
without accelerators. Groups must be physically disjoint, and the configured
budgets must account for other activity on each host. Resource reservations
coordinate one executor instance, not independent submitting processes.
Optional `max_concurrent_jobs` limits the number of jobs reserved on a worker,
including preparation. Without it, the declared resource budgets determine
parallelism. Multi-node groups remain exclusive regardless of this setting.

SSH aliases, usernames, ports, private keys, and jump hosts use SSH config.
Server keys are verified against known_hosts. Authentication agents can be used
locally; their credentials are not forwarded to workers. AsyncSSH is an optional
client dependency and is imported only when execution is requested.

## Runtime

The public Executor/Job interface remains synchronous. A dedicated asyncio
loop in a background thread schedules ready jobs and handles SSH streams.
Submission queues work onto that loop; job state is observed locally. Blocking
workspace/log operations run outside the event loop so they cannot stall
controller heartbeats.

The session lazily opens a connection pool for each host, reusing connections
across WorkUnits. Up to eight jobs reserve each connection through preparation
and execution, with one channel per job active at a time. Further admitted jobs
open additional connections, so channel limits do not cap host parallelism at
eight. New handshakes are serialized per host to avoid bursts of unauthenticated
connections. The per-connection limit stays below OpenSSH's default
[MaxSessions of 10](https://man.openbsd.org/sshd_config#MaxSessions); servers with
lower configured limits require corresponding capacity planning. The session
closes all pooled connections after its jobs stop. A connection failure never
triggers automatic reconnection or replay of uncertain work.

Each node runs a small standard-library Python supervisor over its SSH channel.
It reads the launch description and controller heartbeats from stdin, and
starts the task bootstrap in a fresh process group with inherited CPU affinity.
Task stdin is separate from the control stream. stdout and stderr are drained
concurrently to the local job log and published through the workspace's normal
streaming/finalization hooks. The controller is the sole job-log publisher.
Output is buffered in batches of roughly 256 KiB, with a flush every 200 ms
for quieter jobs. Slow writes apply backpressure to the output streams while
SSH control traffic continues. Finalization waits for buffered and in-flight
writes, and write failures fail the job.

Environment preparation completes on every node before the job becomes
`running`. Snapshots and prepared commands reuse the existing environment store;
user work always runs in a fresh process. Preparation has a separate timeout
from task execution. No remote Misen daemon or preinstalled Misen environment
is required; the initial host needs Linux, Bash, and Python >=3.9.

## Workspaces and Dask

A shared DiskWorkspace must be mounted at identical absolute paths on the
submitter and workers. A CloudWorkspace uses its ordinary bootstrap transport;
every node needs its own object-store credentials and a relative cache_dir.
InMemoryWorkspace cannot serve remote work. The executor requires snapshots
and worker-side materialization rather than submitter-side prewarming.

With DASK_CLIENT, the first host runs the scheduler and task coordinator, and
all hosts run one Dask worker each. An optional addresses list supplies the
node-to-node network addresses. The existing fixed-membership Dask runtime
owns readiness and worker-loss detection. The group must share a trusted private
network with a free scheduler port. Without DASK_CLIENT, the payload runs once
on rank zero, and user code owns any extra distributed orchestration.

## Cancellation and failures

Queued cancellation prevents dispatch. Active cancellation sends a control
message to every rank through asyncio events, independently of the heartbeat
interval. Cancelling a job closes its channels while other jobs continue on
the same connection. Remote supervisors send TERM to their process group,
then KILL after a bounded grace period. They also clean surviving children
after normal task exit. Controller EOF, signals, execution timeouts, and a
15-second heartbeat lease all trigger cleanup. A lost network connection
therefore does not require another working SSH connection to stop user work.
These guarantees cover cooperative process groups, not children deliberately
escaping through a new session or external service.

A failed dependency prevents downstream execution. Job logs finalize before
terminal states become visible. Connection or protocol failure disables the
affected group for the current session when process cleanup is uncertain.
Neither failed nor uncertain execution is automatically retried. Cached task
results and their normal runtime leases retain Workspace's existing semantics.
Normal executor shutdown waits for owned jobs; it never powers off the hosts.

## Monitor I/O

The TUI resolves log sources and reads files or cloud-backed logs in a worker
thread. Only one log read is in flight at a time, and each source contributes
at most 65,536 characters per update. Navigation invalidates earlier reads so
late results cannot replace the newly selected task or job. UI rendering stays on
the UI loop. Leaving the monitor cancels the awaiting UI task; an already
running storage call finishes according to that backend's timeout behavior.

## Limits

Scheduling requires the submitting process to remain alive. There is no detached
controller, cross-process reattachment, dynamic host provisioning, or automatic
retry. Environment caches persist on the hosts across jobs and sessions.
Resource declarations are scheduling
budgets and cooperative controls, not OS memory limits or a multi-user scheduler.
