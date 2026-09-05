# Design: remote executors

Status: SkyPilot is the first optional remote adapter and can target the cloud,
Kubernetes, SSH, and Slurm compute infrastructures registered by the installed
SkyPilot SDK. Direct native SSH, remote Slurm, Kubernetes, Modal, and provider
Batch adapters remain planned.

## Decision

Use SkyPilot as an adapter, not as Misen's universal remote-execution
abstraction.

SkyPilot is a strong first integration because it already translates portable
resource requests across clouds and existing clusters, and its
[managed jobs](https://docs.skypilot.ai/en/stable/examples/managed-jobs.html)
own recovery, resource teardown, and job lifecycle after submission.
That gives Misen a broad remote backend without immediately duplicating every
provider and cluster control plane.

It therefore remains optional.

## Installation and backend selection

Misen declares the oldest compatible provider-neutral SDK without an upper
bound. Compatibility CI tests both that minimum and the newest stable release
on Python 3.14. Install only the upstream provider extras needed by the
SkyPilot environment that provisions compute.
For the default local API server this is the environment running Misen; a
logged-in remote SkyPilot API server owns its provider packages and
configuration, and its Misen clients need only `misen[skypilot]`:

```bash
uv pip install "misen[skypilot]" "skypilot[aws,gcp]>=0.13"
# Or select a different set:
uv pip install "misen[skypilot]" "skypilot[kubernetes,ssh,slurm]>=0.13"
uv pip install "misen[skypilot]" "skypilot[oci,lambda,runpod]>=0.13"

# Azure currently needs SkyPilot's documented uv prerequisite:
uv pip install --prerelease allow "azure-cli<2.87.0"
uv pip install "misen[skypilot]" "skypilot[azure]>=0.13"

sky check

# From a Misen source checkout:
uv sync --extra skypilot
uv run --extra skypilot --with "skypilot[runpod]>=0.13" sky check runpod
```

The SDK must be installed in the same environment that runs Misen, not only
as an isolated `uv tool`, because the executor loads `sky` in-process.
The integration requires SkyPilot 0.13 or newer. Misen supports Python
3.11–3.14; individual SkyPilot releases and provider extras may impose
additional constraints. `misen[skypilot]` includes the provider-neutral SDK,
but no provider clients.
Keeping those clients explicit avoids installing large and unrelated stacks
such as Azure CLI, Kubernetes clients, or Ray for every user. It also avoids
claiming that `skypilot[all]` has identical dependency support on every Python
version. Azure currently needs the extra uv prerequisite step in SkyPilot's
[installation guide](https://docs.skypilot.co/en/latest/getting-started/installation.html).

`SkyPilotExecutor.infra` is passed to `sky.Resources` and may be a single
infrastructure string or an ordered list of alternatives. Common mappings are:

| Target | Upstream extra | Example `infra` | Multi-node in SkyPilot 0.13 |
|---|---|---|---|
| AWS / Google Cloud | `aws` / `gcp` | `aws/us-east-1`, `gcp/us-central1` | Yes (GCP TPU caveats) |
| Azure / OCI | `azure` / `oci` | `azure/eastus`, `oci` | Yes |
| Lambda Cloud / RunPod | `lambda` / `runpod` | `lambda`, `runpod` | Yes / no |
| Kubernetes | `kubernetes` | `k8s/my-context` | Yes |
| Existing machines | `ssh` | `ssh/my-node-pool` | Yes |
| Slurm through SkyPilot | `slurm` | `slurm/my-cluster/my-partition` | Yes |

Ordered alternatives may mix backend families. Provider-specific fields such
as `instance_type` and `image_id` apply to every resource option and should
normally remain unset for a heterogeneous list.

The same pattern applies to SkyPilot's other registered compute providers:
install the matching upstream extra, configure its credentials, and use its
infrastructure name. Provider capabilities still apply; in particular, not
every provider supports multi-node resources, so managed `DASK_CLIENT` work
requires a backend for which SkyPilot reports multi-node support. A managed
job also needs an enabled infrastructure that can host SkyPilot's jobs
controller; some valid workload targets, including Lambda and Slurm, cannot
host that controller themselves. This is an upstream placement constraint,
not a Misen allowlist.

Compute selection does not select Misen's data store. `executor.infra` may be
`runpod`, `k8s/...`, or `azure/...` while `workspace.backend` is `s3`, `gcs`,
or `azure`, provided every worker has network access and ambient credentials
for that object store. This separation is intentional: SkyPilot owns compute;
`CloudWorkspace` owns snapshots, payloads, results, locks, and logs.

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

## SkyPilot MVP contract

The current `SkyPilotExecutor` uses SkyPilot Python SDK interfaces available
in version 0.13 and newer:

- Each pending Misen work unit becomes its own SkyPilot managed job. Misen
  submits its asynchronous launch request eagerly, without waiting for
  managed-job ID assignment or for parent jobs to finish. Without a pool,
  arbitrary DAG shapes are supported and independent branches can provision
  and execute in parallel when backend capacity permits. Request IDs are persisted
  first; polling resolves and durably records managed IDs. The generated launch
  name provides a recovery path if SkyPilot has already expired the request
  metadata.
- An optional `pool` sends those jobs to an existing SkyPilot worker pool.
  Reusing a worker preserves its node-local Misen environment store and avoids
  repeated provisioning after warm-up. The pool remains SkyPilot-owned: Misen
  neither creates nor tears it down, and each work unit's resource request must
  fit the pool worker shape. Pending work units must be dependency-independent
  because a descendant admitted first could occupy an exclusive worker while
  its parent waits for pool capacity; already-cached parents are fine. SkyPilot
  currently labels pools beta and its pool CLI experimental.
- An optional `manage_api_server` owns the local API process for a foreground
  session. CLI monitors and blocking submissions establish `executor.session()`
  automatically; nonblocking callers establish it explicitly. A lazy guardian
  runs the SDK in a child process with native `SKY_RUNTIME_DIR` isolation and
  private identity/configuration/ports. An authenticated local JSON socket
  leases that namespace's server; concurrent clients share it, and the last
  disconnect cleans up its process tree even after client death. Separate
  namespaces have separate lifetime locks and servers. SDK autostart is
  disabled only inside the broker. No parent SDK import, environment rewrite,
  `HOME` override, or global stop command is used. A child-only compatibility
  patch relocates SkyPilot's remaining legacy user-hash file and generated SSH
  shortcuts, including in spawned request workers.
  Normal exit drains accepted launch requests before stopping; abrupt exit can
  interrupt unresolved requests. Durable records and local SkyPilot state are
  retained for reconciliation. Managed keys include the persistent namespace
  directory; unmanaged keys remain backwards-compatible. Private namespace
  configuration rejects consolidation mode and shared databases so remote jobs
  and pool controllers can continue after the API service stops. Pools belong
  to this separate controller namespace and are managed explicitly through
  session helpers. The managed dependency extra pins a tested nightly with
  native isolation; the ordinary extra retains stable 0.13+ support.
- Managed Dask is an orthogonal, intra-work-unit layer. Each Dask-backed work
  unit owns a temporary cluster inside its one SkyPilot allocation; Dask does
  not schedule Misen's work-unit DAG, and work units never share a live Dask
  runtime. SkyPilot may reuse the underlying worker cluster serially when
  `pool` is configured.
- For non-pooled DAGs, dependencies are submission-scoped state files in the
  workspace. A worker waits for every parent to publish `done` before entering
  user code and publishes its own terminal state on exit. Once the jobs are
  accepted, these gates remain usable after the submitter exits; no long-lived
  Misen DAG controller is required for ordinary user-code success or failure.
- For non-pooled dependent DAGs, eager submission has a cost tradeoff: a
  descendant can provision while its worker is blocked on dependency markers.
  An infrastructure or bootstrap failure before the parent worker starts cannot
  publish a marker itself. Misen publishes one when the submitting process
  observes the terminal SkyPilot status; without that observation, descendants
  wait until their cumulative command timeout.
- Multi-node requests are passed to SkyPilot as `num_nodes`. A work unit that
  binds `DASK_CLIENT` runs a ranked wrapper on every node. It branches on
  `SKYPILOT_NODE_RANK`: rank 0 hosts the scheduler and sole Misen task
  coordinator, and every node (including rank 0) hosts exactly one worker.
  CPU, memory, and accelerators are per-node requests; the scheduler and
  coordinator share rank 0's resources with its worker. Without
  `DASK_CLIENT`, the Misen payload runs only on rank 0 and user code remains
  responsible for orchestrating the other nodes.
- The managed Dask runtime uses the allocation's private node network. Nodes
  must be mutually reachable over Dask's unencrypted TCP transport, and the
  scheduler must not be exposed outside the allocation. Misen does not add
  authentication or encryption, so the allocation network must be trusted and
  isolated from untrusted workloads (including other Kubernetes tenants). The
  scheduler address uses the first address in `SKYPILOT_NODE_IPS` and
  `dask_scheduler_port` (default 8786, valid range 1024–65535), which must be
  free on rank 0 and allowed by the private network. Each node independently
  fetches and materializes the immutable snapshot before joining, so worker
  identity, package access, and `dask_startup_timeout` (default 600 seconds)
  apply across the whole group. The project environment must include
  `distributed`.
- Membership is fixed rather than elastic. The coordinator waits for exactly
  one worker per node and verifies that membership remains unchanged on exit.
  Scheduler/coordinator failure tears down rank-0 roles, scheduler shutdown
  releases the remote workers, and a role failure on any rank fails the
  SkyPilot task. Non-head diagnostics may be available primarily in
  SkyPilot's managed-job logs rather than Misen's rank-0 job log.
- `snapshot=true` and `prewarm_envs=false` are required. Workers fetch and
  materialize the submitted snapshot themselves; a pooled worker can reuse its
  content-keyed environment materialization across managed jobs.
- Worker images must provide Bash and GNU `timeout`. Unless uv, Pixi (when
  needed), and dependency caches are pre-provisioned, bootstrap also requires
  outbound access to install tools and the project's locked dependencies.
- The workspace must expose a non-path bootstrap transport and a relative
  local cache path. `CloudWorkspace(cache_dir=".cache/misen")` is the normal
  pairing.
- SkyPilot receives CPU and memory as per-node minimums. Misen accelerator
  types such as `cuda` are not hardware models, so configuration explicitly
  maps each type to candidate SkyPilot names; optional capacity metadata
  filters those candidates for per-device memory requests.
- `job_recovery` accepts an infrastructure-recovery strategy name, while
  application-error restarts remain disabled. Retrying side-effecting task
  code requires a future explicit Misen policy rather than an implicit
  backend default.
- SkyPilot is the compute control plane only. Misen does not use SkyPilot file
  mounts as an alternate artifact store, and worker tasks set
  `api_server_access=False`.
- A Misen workspace job log reflects the current worker attempt. Prior-attempt,
  provisioning, early-bootstrap, and non-head diagnostics may remain only in
  SkyPilot's managed-job logs.
- One direct request-status query resolves outstanding launch requests, then
  one bulk SkyPilot queue request maps assigned managed jobs into Misen's
  `pending`, `running`, `done`, `failed`, and `unknown` lifecycle. Terminal
  observations are cached, and the workspace remains the success authority:
  a committed Misen result stays successful if SkyPilot reports the job as
  failed. Status queries refresh an autostopped managed-jobs controller and
  publish terminal markers for failures that never reached a worker.
- Misen cancellation waits for an unresolved launch request to assign its
  managed-job ID before cancelling that job. Raw SkyPilot API cancellation does
  not guarantee that the launch handler has quiesced, so externally cancelled
  requests receive a failed worker gate and a 60-second late-ID reconciliation
  window; an exceptionally late worker may still provision before the gate
  makes it exit.

Without a pool, this decomposition preserves Misen's arbitrary DAG semantics
without claiming that SkyPilot exposes a native arbitrary-DAG managed job. Its
deliberate tradeoffs are eager descendant provisioning and the need for status
observation to propagate failures that happen before Misen code starts.

## Authentication and trust

There are several separate credential relationships:

- The Misen process authenticates to SkyPilot and to its configured workspace;
  the local or remote SkyPilot API server authenticates to each compute backend
  so it can provision and query compute. For an S3 workspace, credentials in
  `~/.aws/credentials` can be loaded into the submitter environment with AWS
  CLI v2: `eval "$(aws configure export-credentials --profile default --format env)"`.
- The worker authenticates directly to the Misen workspace bucket so it can
  fetch snapshots and payloads and publish results and logs.

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

1. **Harden the SkyPilot MVP.** Exercise representative elastic-cloud and
   attached-cluster targets end to end, including AWS/GCP, Kubernetes,
   SSH/Slurm, and a single-node GPU provider; document provider identity
   setup; finish log UX and failure diagnostics.
2. **Harden dependency orchestration.** Persist attachable submission
   manifests, make pre-worker infrastructure failures propagate without a live
   status observer, and reduce the cost of descendants provisioning while
   blocked on dependency gates.
3. **Add SSH and remote Slurm.** Build the command transport and durable
   supervisor once, then reuse it for Slurm CLI submission/status. Slurm's
   [dependency options](https://slurm.schedmd.com/sbatch.html) preserve graph
   scheduling in the cluster.
4. **Add native Kubernetes, Modal, and provider Batch adapters.** Each adapter
   owns resource/status translation but reuses the same snapshot/bootstrap
   and Workspace data plane.
5. **Generalize proven lifecycle needs.** Persist backend IDs and graph
   mappings, support attach/cancel, formalize workspace capability
   validation, and add age-based pruning for snapshots, environments, and
   submission blobs.

Success means users can change compute control planes without changing task
definitions, result identity, cache semantics, or artifact storage.
