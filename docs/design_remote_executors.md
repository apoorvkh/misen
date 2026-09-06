# Design: remote executors

Status: SkyPilot is the first optional remote adapter and can target the cloud,
Kubernetes, SSH, and Slurm compute infrastructures registered by the installed
SkyPilot SDK. Direct native SSH, remote Slurm, Kubernetes, Modal, and provider
Batch adapters remain planned.

SkyPilot execution now separates reusable compute allocations from logical
work units. See the [usage guide](skypilot.md) for the implemented contract and
[architecture plan](design_skypilot_graph_execution.md) for remaining work.

## Decision

Use SkyPilot as an adapter, not as Misen's universal remote-execution
abstraction.

SkyPilot is a strong first integration because it already translates portable
resource requests across clouds and existing clusters, and its
[managed jobs](https://docs.skypilot.ai/en/stable/examples/managed-jobs.html)
own allocation provisioning, resource teardown, and native job lifecycle.
That gives Misen a broad remote backend without immediately duplicating every
provider and cluster control plane.

It therefore remains optional.

## Installation and backend selection

The [SkyPilot usage guide](skypilot.md) is the configuration and lifecycle
reference. Install the SDK in the environment that runs Misen and only the
provider extras required by the local or remote API server doing provisioning.

The default isolated local API requires `misen[skypilot-managed]`, which pins
the tested runtime-isolation nightly. A configured shared/remote API uses
`manage_api_server=false` and can use the ordinary provider-neutral
`misen[skypilot]` extra (`skypilot>=0.13`). Client/server capabilities remain
version-dependent; do not install the stable and nightly distributions together.

A capacity profile selects exactly one source: an existing `cluster`, an
existing `pool`, or an `infra` string/list for run-owned managed allocations.
Profile `infra` values pass through to SkyPilot, for example
`aws/us-east-1`, `gcp/us-central1`, `k8s/my-context`, or
`ssh/my-node-pool`. Provider-specific creation options belong only on
`infra` profiles and must work for every selected alternative.

Install the matching provider extras and verify that the chosen backend
supports the requested topology. Pool behavior, multi-node support, controller
placement, and authentication remain SkyPilot responsibilities; Misen does
not promise identical features on every registered provider. Consult the
upstream [installation guide](https://docs.skypilot.ai/en/latest/getting-started/installation.html)
for provider packages and platform constraints.

Compute selection does not select Misen's data store. A capacity profile may
use AWS, Kubernetes, or another compute backend while `workspace.backend`
selects S3, GCS, or Azure Blob. Every participating worker/coordinator must
reach and authenticate to the chosen workspace.

Misen still needs direct adapters. Users may already have SSH hosts, Slurm
clusters, Kubernetes policy, Modal applications, or managed Batch queues that
must be addressed in their native security and operational model. Routing
all of those through SkyPilot would add an unnecessary control plane and
would hide backend features Misen may need.

Executor names therefore identify a compute control plane
(`skypilot`, future `ssh`, `remote_slurm`, `kubernetes`, `modal`,
`aws_batch`, or `gcp_batch`), not a cloud vendor. A SkyPilot target is
selected with SkyPilot's
[infrastructure strings](https://docs.skypilot.ai/en/latest/overview.html),
for example `aws/us-east-1`, `k8s/my-context`, or `ssh/my-node-pool`.

## Stable boundary

Remote execution has three owners:

1. **Misen** builds the cache-bounded work-unit DAG, publishes an immutable
   project snapshot, prepares payloads, and defines normalized job states.
2. **The executor** validates capabilities, translates resource requests,
   submits work to a durable control plane, and maps native status back to
   Misen.
3. **The workspace** is the data plane for snapshots, payloads, env files,
   results, locks, scratch synchronization, and logs.

The worker contract from `design_unified_snapshot.md` is deliberately
backend-neutral. An executor delivers a small Bash bootstrap plus opaque
workspace references; it does not copy the project or teach the remote
control plane about Misen's result format.

Do not introduce a broad `RemoteExecutor` base class yet. The existing
graph-level dispatch hook and workspace transport are the correct extension
points. Extract smaller shared pieces only after two adapters demonstrate the
same need: durable submission manifests, status normalization, cancellation,
reattachment, and command/log wrapping are likely candidates.

## SkyPilot graph contract

The public executor schedules a ready-only work-unit graph over explicit,
bounded capacity profiles. It does not eagerly submit blocked descendants or
retain a separate per-work-unit execution mode.

- Logical work units retain the existing cache boundaries. The coordinator
  releases a successor only after its own prerequisites commit success.
- Reusable agents are started once per allocation and execute one fresh task
  subprocess at a time. Existing-cluster reservations use `sky.exec()`;
  pools and run-owned infrastructure use managed jobs. Dedicated profiles
  reserve a separate allocation for each admitted work unit.
- Profiles declare per-node CPU/RAM, concrete accelerator hardware and memory,
  topology, source, and maximum worker count. There is no automatic replacement,
  multi-slot scheduling, cross-run agent sharing, or dynamic autoscaling.
- Versioned manifests distinguish runs, logical jobs, attempts, and allocations.
  Allocation records retain accepted request IDs and resolved native IDs.
  Exact-name recovery never broadens cancellation to unrelated jobs.
- The actual agent transport is a bounded JSON mailbox over known workspace
  keys, polled every 0.2 seconds by default. It is not SSH-forwarded task RPC
  or a bucket-listing queue. Allocation health checks run independently of
  task readiness/completion.
- Execution claims refuse intentional replay of uncertain attempts. Completed
  attempt records can reconcile lost observations; application errors and
  incomplete attempts are not automatically retried. Existing cache locks
  do not supply exactly-once semantics for arbitrary external side effects.
- An attached run's coordinator lives in the invoking process and advances
  independently of UI/status polling. Nonblocking submissions require a live
  session; CLI runs and `Experiment.run()` wait by default. Session exit stops
  admission, cancels unfinished owned work, and attempts bounded cleanup.
- The default local API is isolated by persistent namespace identity/state and
  leased through an authenticated local broker. Last-client disconnect cleans
  up the owned local API process tree, without modifying the parent's
  environment, `HOME`, or ordinary SkyPilot services.
- Detached execution is explicit: a dedicated single-node remote coordinator
  requires a stable remote API, enabled service-account token injection, and
  a compatible SDK in the project's locked environment. Only that coordinator
  receives SkyPilot API access. Its deadline is finite; automatic coordinator
  takeover is disabled.
- `attach(run_id, workspace)` reconstructs observing/cancelling handles from
  trusted durable state. It does not restart a lost coordinator or silently
  resubmit uncertain work.
- Pools and existing clusters are borrowed. Misen never automatically creates,
  resizes, or tears them down; cancellation targets only this run's jobs.
  Explicit session pool helpers remain available. Managed-job controllers can
  be shared and may remain billable after the run; the adapter does not guess
  their ownership or delete them.
- Multi-node profiles require `dedicated=true`. `DASK_CLIENT` adds an
  allocation-scoped runtime: one worker per node, scheduler/coordinator on
  rank 0. The current wrapper uses a 300-second Dask startup limit and port
  8786 on a trusted private network. Its TCP is not authenticated/encrypted.
  Without the sentinel, the payload runs only on rank 0 and user code owns
  any other-node orchestration.
- Snapshot/bootstrap requirements remain `snapshot=true`,
  `prewarm_envs=false`, a remotely fetchable workspace with job-file reads,
  and a relative local cache directory. Workers need Bash, GNU `timeout`,
  workspace identity, and package access; multi-node projects need
  `distributed`. Fresh subprocesses reuse worker-local environment caches.
- Durable completion is distinct from process/log finalization. Misen can
  release descendants before SkyPilot observes terminal allocation state;
  cleanup still tracks owned processes and reports unresolved work.

The implementation is locally tested, not AWS-validated as a graph executor.
The earlier pool smoke benchmark covers its predecessor. See the
[architecture plan](design_skypilot_graph_execution.md) for remaining timeout,
recovery, resource-isolation, cleanup, transport, and performance gates.

## Authentication and trust

There are several separate credential relationships:

- The Misen process authenticates to SkyPilot and to its configured workspace;
  the local or remote SkyPilot API server authenticates to each compute backend
  so it can provision and query compute. For an S3 workspace, credentials in
  `~/.aws/credentials` can be loaded into the submitter environment with AWS
  CLI v2: `eval "$(aws configure export-credentials --profile default --format env)"`.
- The worker authenticates directly to the Misen workspace bucket so it can
  fetch snapshots and payloads and publish results and logs.
- A detached Misen coordinator additionally needs SkyPilot API access through
  the remote server's service-account credential injection. Ordinary agents
  do not receive that provisioning authority. Their project environment and
  workspace must be trusted because execution payloads are serialized code.

Provision worker access with an instance role, service account, workload
identity, or an equivalent ambient mechanism. Bootstrap shell text can be
visible in control-plane commands, so `CloudWorkspace.config` is rejected
for remote bootstrap and must not carry worker credentials. Non-secret locator
fields such as an endpoint or S3 region are safe. Use least-privilege access
to the configured workspace bucket/prefix and configure the corresponding
SkyPilot worker identity outside Misen.

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

1. **Validate the graph executor on real infrastructure.** Exercise elastic-cloud and
   attached-cluster targets end to end, including AWS/GCP, Kubernetes,
   SSH/Slurm, and a single-node GPU provider; document provider identity
   setup; finish log UX and failure diagnostics.
2. **Harden graph orchestration.** Benchmark the workspace mailbox transport,
   distinguish setup/execution timeouts across every allocation type, test
   controller/orphan cleanup, and add tested result-publication fencing before
   enabling coordinator takeover or uncertain-attempt replay.
3. **Add SSH and remote Slurm.** Build the command transport and durable
   supervisor once, then reuse it for Slurm CLI submission/status. Slurm's
   [dependency options](https://slurm.schedmd.com/sbatch.html) preserve graph
   scheduling in the cluster.
4. **Add native Kubernetes, Modal, and provider Batch adapters.** Each adapter
   owns resource/status translation but reuses the same snapshot/bootstrap
   and Workspace data plane.
5. **Generalize proven lifecycle needs.** Extend the existing manifests,
   native IDs, and observing/cancelling handles only where adapters share a
   demonstrated need; add reference-safe pruning for snapshots, environments,
   and submission blobs.

Success means users can change compute control planes without changing task
definitions, result identity, cache semantics, or artifact storage.
