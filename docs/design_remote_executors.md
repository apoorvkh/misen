# Remote executors

Status: SkyPilot is the first optional remote adapter. All adapter implementation
lives in `src/misen/executors/skypilot.py`. Direct SSH, remote
Slurm, Kubernetes, Modal, and provider Batch adapters remain planned.

## Decision

SkyPilot provisions compute; persistent Misen agents execute WorkUnit subprocesses over SSH. Misen owns DAG scheduling
and WorkUnit lifetimes; Workspace owns snapshots, payloads, results, locks,
and logs. Keep these boundaries independent so users can change compute
without changing task definitions or artifact storage. The graph-dispatch
hook and workspace bootstrap transport are sufficient integration points;
there is no broad RemoteExecutor abstraction yet.

## Installation and backend selection

Install `misen[skypilot]` and the desired `skypilot-nightly` provider extras
in the submitting environment, for example:

```bash
uv pip install "misen[skypilot]" "skypilot-nightly[aws,gcp]>=1.0.0.dev20260905"
```

The minimum nightly provides configurable local API ports, isolated runtime
directories, and foreground server startup. CI covers the minimum and latest
nightly on Python 3.14; provider extras can impose additional constraints.
Every worker entry owns its `infra` string, such as `aws/us-east-1`,
or `ssh/my-pool`. Workers require SSH-accessible Linux nodes. SkyPilot retains provider validation and
capability checks. Multi-node WorkUnits require a supporting provider.

## Local control-plane ownership

Every executor session starts one supervised local graph-controller process using
the submitting interpreter. Before importing SkyPilot, its environment sets
an isolated `SKY_RUNTIME_DIR`, a private `SKYPILOT_API_SERVER_LOCAL_PORT`, and
a loopback `SKYPILOT_API_SERVER_ENDPOINT`. The controller supervises an API
child started with `sky.api_start(foreground=True, port=...)`. It never invokes
`sky.api_stop()` or shares ownership of another API server. After startup,
implicit SDK server autostart is disabled so an API failure cannot silently
create a detached replacement. Provider credentials and user configuration
remain available from the real home directory.

No cloud controller VM, remote API service, managed-jobs API, or nested
credential injection is required. The local manifest contains workspace
configuration, immutable commands, dependency IDs, and worker types; SDK
dependencies stay on the submitting host. Ordinary workers receive
`api_server_access=False`. The local API and catalog preflight must become
ready within `startup_timeout` before submission returns job handles.

Scheduling requires the submitting process to stay alive. Context exit,
`close()`, and normal process exit request cooperative controller shutdown:
tear down cloud workers first, then stop the owned API and controller tree.
Cleanup has a shared bounded wait before forced process termination.
On Unix a parent-loss watcher requests cooperative cleanup and bounds stalled
SDK waits before stopping the API tree; direct-child SIGKILL would orphan
SkyPilot helper processes. Windows uses processkit's Job Object to bind the
whole tree to its owner. SkyPilot autodown backs up interrupted cloud cleanup after native jobs finish
or hit their execution timeout. The executor does not promise detached
scheduling or successful teardown after an abrupt kill.

## Reusable workers

`workers` is a required, non-empty, unordered list of `SkyPilotWorker` types.
Each entry declares per-node CPU/RAM budgets, whole accelerators, node-group
size, provider options, and `max_workers`. The executor's `max_workers` caps
all groups across submissions in the session, including provisioning and draining groups.
Worker names are generated internal identities, not user-facing pool names.

The controller launches each group through `sky.launch` once, then opens an
SDK-authenticated SSH stream to a small stdlib agent on each node. Ready
WorkUnits use subprocesses, without `sky.exec` or native per-job status polling.
Agents report completion and kill process groups on cancellation or connection
loss. A single native guard stays active while the agent lease is renewed,
preventing SkyPilot autodown from interrupting externally dispatched work.
After lease expiry the guard exits, allowing autodown to clean up abandoned VMs.

Each worker owns its status, connections, and environment preparation state;
only durable fields enter checkpoints. The controller receives a configuration
copy without live process handles and validates resource eligibility before launch.
SDK completions and SSH events share one queue. Frontier planning, scheduling,
and retirement remain separate steps, with one capacity check for launching and
replacing workers. Shutdown waits for in-flight launches before requesting teardown.

Scheduling accounts for starting capacity and packs compatible ready work.
`lookahead_seconds` (default 90) bounds the next independent future frontier;
observed elapsed times improve estimates without treating task timeouts as
predictions. Workers materialize snapshot environments before running payloads.
Preparation uses the idle worker's full declared CPU budget. A cached launch
command then activates the prepared environment in a fresh subprocess, keeping
resource assignments and env-file values specific to each WorkUnit. Payload
transport still runs for each new job; dependency materialization does not.
All admission, provisioning, and speculation obey per-type and session limits.
The policy does not claim a global makespan optimum.

Single-node jobs pack within CPU/RAM and whole-device budgets. CPU thread
caps and memory admission are cooperative controls, not OS memory isolation.
Multi-node jobs exclusively reserve a complete group. `DASK_CLIENT` creates
one private Dask worker per node with its scheduler and coordinator on rank
zero. Otherwise only rank zero executes the Misen payload. The Dask scheduler
port must be free and reachable on the allocation's trusted private network.

GPU eligibility uses per-device GiB capacities from concrete catalog
offerings pinned to instance and region. One high-memory variant cannot
justify a lower-memory variant of the same model. `accelerator_memory`
metadata fills unknown capacities without inflating known values; unknown
capacity cannot satisfy a memory minimum. Assigned memory is checked on every
node before user code or Dask starts (CUDA via `nvidia-smi`, ROCm/XPU via the
task environment's PyTorch). Other constrained backends are rejected.

Idle groups retire after `idle_timeout_minutes` or when another type needs their slot.
With `reuse_workers=true`, submissions in the same executor and workspace share
capacity and environments; `close()` retires all groups. `reuse_workers=false`
releases the pool once all accepted work finishes. Per-WorkUnit cancellation leaves unrelated
work running; provisioning and bootstrap failures propagate to descendants.

## Status and interrupted submissions

Workspace checkpoints contain logical states, resource reservations, launch/teardown
request IDs, and owned cluster names. Ownership is saved before dispatch, and
uncertain side effects are never blindly replayed. Committed workspace
results take precedence over later native failures. Matching graphs and
snapshots reattach within the same live executor, including cache pruning
and traversal reordering. Different graphs retain separate checkpoints while sharing the session pool.
Local status files and a cancellation inbox keep scheduler traffic off S3. Infrastructure failures do not automatically retry user code.

A graph ownership record is written before starting its controller. After
controller loss, active workers with unconfirmed teardown keep affected jobs
unknown. A new submission refuses to replay an unresolved graph. Retained
local runtime state and the workspace checkpoint support diagnosis and
manual reconciliation; confirm outstanding cluster teardown before clearing
the durable graph record. Cross-process controller reattachment is not
implemented. Once every job is terminal and every worker is down, a new
submission may run normally.

## Authentication and trust

The local API uses submit-host provider packages and credentials. Each worker
independently accesses the workspace object store through an instance role,
service account, or equivalent ambient identity. Compute selection does not
select storage: an AWS worker can use any supported bucket it can reach and
authenticate to. `CloudWorkspace.config` cannot carry bootstrap credentials;
commands and snapshots are visible to the compute control plane.

The adapter requires `snapshot=true`, `prewarm_envs=false`, remotely fetchable
workspace transport, coordination-file reads, and a relative cache path.
Workers need Bash, GNU `timeout`, and snapshot dependency access. Dask work
also requires `distributed` in the project environment.

## Alternatives and planned adapters

| Adapter | Why keep a direct path | Main design work |
|---|---|---|
| SSH | Smallest path to existing machines; no provisioning layer required. | A durable remote supervisor, process identity, reconnect/status/cancel, host selection, and safe bootstrap transport. |
| Remote Slurm | Reuses sites' queues, accounting, dependencies, and allocation policy. Slurm already supports native job dependencies. | Separate the current local `sbatch`/status commands behind an SSH command transport, then remove remaining working-directory and scheduler-log shared-path assumptions. |
| Kubernetes | A native [Job](https://kubernetes.io/docs/concepts/workloads/controllers/job/) provides run-to-completion and retry semantics and integrates with cluster RBAC, quotas, and workload identity. | Kubernetes Jobs are not an arbitrary DAG engine; use native objects plus a durable controller or an established workflow primitive, without making SkyPilot-on-Kubernetes mandatory. |
| Modal | Modal [Functions](https://modal.com/docs/guide/functions) and durable [spawned calls](https://modal.com/docs/guide/function-invocation-methods) fit serverless, bursty workloads and existing Modal deployments. | Package the Misen bootstrap in Modal's image/function model and persist native call IDs for reattachment; preserve Workspace as the result authority. |
| AWS Batch | Existing queues, compute environments, IAM, and compliance may make direct submission preferable. AWS Batch has native [job dependencies](https://docs.aws.amazon.com/batch/latest/APIReference/API_SubmitJob.html), with at most 20 parents per submitted job. | Translate resources/container images, persist job IDs, map array/dependency failures, stream logs, and honor queue policy. |
| Google Cloud Batch | Direct integration fits existing GCP queues, service accounts, and regional policy. | The documented [dependent-jobs interface](https://docs.cloud.google.com/batch/docs/create-run-dependent-job) is currently alpha and region-scoped; reassess its stability before relying on it for general DAG submission. |

SkyPilot can also target Kubernetes or SSH-based clusters, but that does not
replace these adapters: the direct paths serve users whose native control
plane is already the contract.

Other broad substrates were considered but are a poorer ownership fit.
[Dask deployment](https://docs.dask.org/en/latest/how-to/deploy-dask-clusters.html)
and the [Ray Jobs API](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/api.html)
are valuable distributed runtimes, but adopting either as the universal
control plane would require a persistent cluster/runtime and duplicate
Misen's scheduling boundary. Lithops, Metaflow, and Covalent already own
function serialization, workflow DAGs, or artifacts, overlapping Misen's
core rather than supplying a thin provisioning layer. [PSI/J](https://exaworks.org/psij-python/)
is a promising helper if Misen expands to more HPC batch schedulers, but it
does not provision AWS/GCP, Kubernetes, or Modal. These remain implementation
references or possible executor-specific helpers, not Misen's remote base
class.

## Roadmap

1. Exercise representative cloud and attached-cluster targets end to end and
   document worker identity setup and diagnostic recovery.
2. Refine duration estimates and provisioning predictions from benchmark history.
   Cross-process reuse would require a separate ownership and recovery design.
3. Add direct SSH and remote Slurm command transport, then native Kubernetes,
   Modal, and provider Batch adapters using the same Workspace data plane.
4. Generalize lifecycle helpers once multiple adapters need them.
