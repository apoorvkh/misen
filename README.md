# misen

A Python framework for writing **research experiments as end-to-end, reproducible workflows**; not one-off scripts. `misen` offers:

- **End-to-end experiments.** Declare your experiment as a composition of Python functions and let `misen` run the whole thing. No need to run scripts one-at-a-time and glue them together.
  - Experiments are Python classes with typed parameters; you get a CLI, hyperparameter sweeps, and named results for free.
  - `misen` tracks Experiment state (completion, failure) and logs. You can easily check which tasks are complete, failed, and need to be run or updated.

- **Caching.** `misen` caches the outputs of your experiment steps automatically. When you re-run an experiment, the results will be retrieved immediately. You don't have to save outputs to specific filenames and remember what scripts produced them. You can access these results *declaratively* (like `exp["metrics"].result()`) in Python.

- **Reproducibility.** Experiment artifacts are kept in sync with the experiment code. Edit a task and `misen` recomputes exactly everything affected. Whole project replication becomes as easy as running one command.

- **Execution.** `misen` runs your experiments' steps in parallel. You can declare necessary resources (e.g. CPUs, GPUs) per task and `misen` will provision these appropriately. You can run the code locally, on SLURM, or on SkyPilot-supported clouds and clusters through the optional SkyPilot executor. We snapshot your code, so you can freely edit while experiments are queued or running.

- **Portability.** Anyone can easily clone and replicate `misen` projects. Since they are standard Python packages, they can even be `pip install`-ed in other projects, so experiments can be modified and repurposed downstream.

## Project Setup

`misen` expects your research project to be structured as a Python [**package**](https://packaging.python.org). This makes your project `import`-able and `pip`-installable, so anyone can extend or reproduce your experiments.

The easiest way to start is with [uv](https://docs.astral.sh/uv/#installation):

```bash
uv init my-project --package --python 3.13
cd my-project
uv sync
uv add misen
```

```
my-project
├── pyproject.toml
├── src
│   └── my_project
│       └── __init__.py
└── uv.lock
```

Put your code in `src/my_project/` and run it as a module — e.g. `uv run -m my_project.experiments.training` for `src/my_project/experiments/training.py`.

If you have a `uv` project, use `uv run` instead of `python` and `uv run misen` instead of `misen` in the instructions below.

## Tasks

A **task** is a Python function annotated with `@meta`:

```python
from misen import Task, meta


@meta(cache=True)
def add(a: int, b: int) -> int:
    return a + b
```

You should run `misen fill` to tag functions with unique ids: e.g. `@meta(id="3X2CLIX6MM")`. Don't type this in yourself.

`Task(add, a=1, b=2)` is a *lazy* handle to `add(a=1, b=2)`. Compose tasks into directed, acyclic workflows by passing one task as the argument of another, like:

```python
train_task = Task(train, lr=0.001, dim=256)
eval_task = Task(evaluate, trained_model=train_task.T)
plot_task = Task(plot, metrics=eval_task.T)
```

`plot_task: Task` now represents the full workflow. `.T` is optional — it just preserves the return type for type-checkers.

Task arguments should be simple, declarative values (ints, strings, enums, `Literal[...]`). Runtime objects (tensors, models) must flow in as the output of another Task, not as direct arguments.

### Caching and versioning

When you mark a task `cache=True`, its results are persisted to the `Workspace`, keyed by a hash of `(id, arguments)`. Re-running with the same inputs returns the cached result instead of re-computing.

If you change your code in a way that invalidates old results, you must regenerate `@meta(id)` so `misen` treats it as a new task and recomputes downstream.

You can also use `@meta(versions)` to invalidate on specific argument values, `@meta(defaults)` to add new arguments, or `@meta(exclude)` to exclude arguments from the cache key.

`misen` ships with built-in serializers for standard Python and the common research stack — NumPy, pandas, Polars, PyArrow, PyTorch, TensorFlow/Keras, JAX, scikit-learn, XGBoost/LightGBM/CatBoost, Hugging Face datasets & transformers, Pydantic, attrs, msgspec, PIL, Plotly, Altair, SciPy, SymPy, xarray, GeoPandas, ONNX. For anything else, pass a custom `@meta(serializer)`.

## Experiments

An `Experiment` binds declarative parameters to a named task workflow:

```python
from misen import Experiment, Task


class TrainingExperiment(Experiment):
    lr: float = 0.001
    dim: int = 256

    def tasks(self) -> dict[str, Task]:
        train_task = Task(train, lr=self.lr, dim=self.dim)
        eval_task = Task(evaluate, trained_model=train_task.T)
        plot_task = Task(plot, metrics=eval_task.T)
        return {"metrics": eval_task, "plot": plot_task}


if __name__ == "__main__":
    TrainingExperiment.cli()
```

We suggest putting each `Experiment` in its own script, like `src/my_project/experiments/training.py`.

Run it from Python:

```python
TrainingExperiment(lr=0.1, dim=512).run()
```

or from the command line:

```bash
misen experiment my_project.experiments.training:TrainingExperiment run --lr 0.1

# or

python -m my_project.experiments.training --lr 0.1
```

Pull a named result declaratively:

```python
metrics = TrainingExperiment(lr=0.1, dim=512)["metrics"].result()
```

### Named configs

You can also pin a specific set of parameters in a named config file (e.g. `src/my_project/configs/training.py`):

```python
from my_project.experiments.training import TrainingExperiment

__config__ = TrainingExperiment(lr=0.1, dim=512)

if __name__ == "__main__":
    __config__.cli()
```

Run it with `python -m my_project.configs.training`.

Or retrieve a result like:

```python
from my_project.configs.training import __config__ as training_experiment

training_experiment["metrics"].result()
```

### Inspecting an experiment

Beyond `run`, every experiment CLI exposes inspection subcommands:

```bash
my_experiment list          # named tasks + their completion status (✓ / ○)
my_experiment tree          # ASCII DAG of the full workflow (-L N for depth)
my_experiment incomplete    # only the tasks still to compute
my_experiment count         # "Completed 7 of 12 tasks"
my_experiment logs          # browse task logs; --job for work-unit logs
my_experiment result NAME   # print a cached result to the console
```

### Sweeps

Experiments are just Python objects, so sweeping is a comprehension:

```python
def plot_sweep(metrics: dict[tuple[float, int], Metrics]) -> Plot: ...


class TrainingSweep(Experiment):
    lrs: list[float] = [0.001, 0.01]
    dims: list[int] = [256, 512]

    def tasks(self) -> dict[str, Task]:
        metrics = {(lr, dim): TrainingExperiment(lr=lr, dim=dim)["metrics"].T for lr in self.lrs for dim in self.dims}
        return {"plot": Task(plot_sweep, metrics=metrics)}
```

## Resources and Executors

Declare what a task needs:

```python
from misen import Task, meta


@meta(
    cache=True,
    resources={
        "memory": 32,
        "accelerators": 4,
        "accelerator_type": "cuda",
        "accelerator_memory": 40,
    },
)
def train(lr: float, dim: int) -> nn.Module: ...
```

Defaults: 1 node, 1 CPU, 8 GiB host RAM, 60 minutes, and 0 accelerators. Time, nodes, CPUs, and memory are positive integer quantities; CPU, memory, and accelerator counts are per node. The accelerator fields are `accelerators`, `accelerator_type`, and `accelerator_memory` in GiB per device. `AcceleratorType` is `Literal["cuda", "rocm", "xpu", "mps", "tpu"]`; its default is `"cuda"`.

The type is concrete rather than `"gpu"` or `"auto"`: task code normally supports a particular backend. A resource function can choose the type from task arguments when an implementation supports several backends.

An importing project can replace the request when it knows another execution-equivalent shape is appropriate, without changing task identity:

```python
task_from_project_a = Task(train, lr=0.001, dim=512)
task_for_project_b = task_from_project_a.with_resources(accelerators=2, accelerator_memory=80)
```

Resource functions can compute the request from task arguments when model size or tensor parallelism is argument-specific. The request describes how the task intends to run; executors translate it into site-specific allocation flags. Hardware names such as a SLURM GRES type therefore stay out of task metadata.

When several non-cacheable tasks execute sequentially in one work unit, Misen takes the maximum node, CPU, memory, and accelerator requirements; accelerator-using tasks must agree on `accelerator_type`.

At runtime, `LocalExecutor` subdivides maskable GPU-family devices and the final worker applies their visibility immediately before loading task code. A TPU task must request the complete configured pool, which Misen reserves exclusively; framework-specific TPU activation remains the task environment's responsibility. Local memory-per-device constraints are rejected because the current inventory cannot verify them. `SlurmExecutor` maps supported GPU-family counts directly and uses cluster-specific rules to recognize memory and non-default type constraints; an unrecognized constraint is rejected rather than silently under-provisioned. TPU and MPS requests are currently rejected by `SlurmExecutor`. SLURM's cgroups handle isolation.

CPU affinity and cgroup membership are inherited by children, so subprocesses (`subprocess`, `multiprocessing`) and native threading libraries automatically stay within the allotment. Three patterns to keep in mind:

- **Sizing:** `os.cpu_count()` reports the whole machine. Use `len(os.sched_getaffinity(0))` for pool sizes, `n_jobs`, DataLoader workers, etc.
- **Native threading libs (OpenMP, MKL, OpenBLAS, …):** `LocalExecutor` exports `OMP_NUM_THREADS` and friends to match the assignment. `SlurmExecutor` leaves thread counts unset — if you want OpenMP saturation matched to your CPU request, either configure your cluster's `srun` to propagate `SLURM_CPUS_PER_TASK → OMP_NUM_THREADS`, or set it yourself early in the task: `os.environ.setdefault("OMP_NUM_THREADS", str(len(os.sched_getaffinity(0))))`.
- **Libraries that reset affinity at import** (some MKL/NumPy builds, certain CUDA runtimes): re-pin after the offending import with `os.sched_setaffinity(0, os.sched_getaffinity(0))`.

For a per-task scratch directory, give the function a plain `Path` parameter and bind the `SCRATCH_DIR` sentinel to it when constructing the task: `Task(train, scratch_dir=SCRATCH_DIR)`. The signature stays misen-agnostic — you can call the function directly with any directory in a test or notebook — and `misen` resolves the sentinel to a fresh directory at execution time, excluding it from the task's identity automatically (no `@meta(exclude=...)` needed). Sentinels must be top-level `Task(...)` arguments: using `SCRATCH_DIR` as a function-signature default or nesting it inside a container raises a `TypeError` when the `Task` is constructed.

Use the directory freely as working space — including for preemption-safe checkpointing during long runs. It's cleaned up automatically on successful completion (and on failure for non-cacheable tasks); for cacheable tasks, a failed run keeps its scratch_dir so a re-run can resume from the latest checkpoint.

To flow files written into the scratch directory (model checkpoints, generated images, training logs) into downstream tasks without round-tripping their contents through memory, return a `FileMap` — a `Mapping[K, Path]` of keyed files. Build it with chainable `include_glob` / `include_tree` / `include` (and `exclude_glob` / `exclude`); the serializer moves each file into the result's cache before scratch_dir is cleaned up, preserving its relative layout. Downstream tasks see paths that resolve into the local workspace.

```python
from misen import FileMap, SCRATCH_DIR, Task, meta


@meta(cache=True, resources={"accelerators": 1})
def train(scratch_dir: Path) -> FileMap:
    # training loop writes ckpt_<step>.pt and tb_logs/ into scratch_dir
    return (
        FileMap()
        .include_glob(scratch_dir, "ckpt_*.pt", key=lambda p: int(p.stem.split("_")[1]))
        .include_tree(scratch_dir / "tb_logs")
        .exclude_glob("*.tmp")
    )


@meta(cache=True)
def analyze_at(files: FileMap, step: int) -> dict[str, float]:
    state = torch.load(files[step], weights_only=True)  # one file loaded on demand
    ...


train_task = Task(train, scratch_dir=SCRATCH_DIR)
analysis = Task(analyze_at, files=train_task.T, step=1000)
```

Keys may be `str`, `int`, `float`, `bool`, or `None`. Exclusions apply eagerly (each `exclude_*` filters what's been included so far). `FileMap.from_glob(...)` / `FileMap.from_tree(...)` are one-liner shortcuts for the single-source case. After a `FileMap` is fetched from a result, `.root` gives the single directory holding every file — hand it to a directory-consuming tool, e.g. `tensorboard --logdir <files.root>`. A `FileMap` loaded from a workspace is read-only.

**Selective access and granularity.** A `FileMap` is *one* cached result holding all its files. On a shared filesystem (`DiskWorkspace` on NFS), reading a single entry (`files[step]`) touches just that one file — loading a `FileMap` reads only its manifest, never the file contents — so accessing one checkpoint out of many is cheap. On `CloudWorkspace`, a result is fetched as a unit, so the first access materializes *all* of a `FileMap`'s files. If you have many large checkpoints and a downstream task on another machine needs only one, give each checkpoint its own cached task (so each is an independently-fetched result) rather than bundling them into one `FileMap` — this is a DAG-shaping choice, not a property of the type. (Per-entry lazy fetch on cloud would be a general `CloudWorkspace` improvement, independent of `FileMap`.)

The **Executor** decides where tasks run. `LocalExecutor` and `InProcessExecutor`
accept only single-node requests; `SlurmExecutor` and `SkyPilotExecutor` map
`nodes` to their scheduler's allocation request.

- `LocalExecutor` — parallel on your machine (default)
- `InProcessExecutor` — single-process, useful in notebooks and tests
- `SlurmExecutor` — submits each work unit as a SLURM job
- `SkyPilotExecutor` — optional execution on SkyPilot-supported clouds and clusters, with one durable managed job per work unit

For a multi-node Slurm or SkyPilot allocation, the task body still runs exactly
once on the first node. Bind `DASK_CLIENT` to use Misen's managed worker group,
with one Dask worker per allocated node, or omit it when the task intentionally
manages its own allocation-scoped runtime. In the latter case, the task also
owns remote-process bootstrap. Slurm can launch into a shared prewarmed
environment with tools such as `srun`; SkyPilot code can use its rank and node
IP environment variables to coordinate the allocation.

### Remote execution with SkyPilot (optional)

Install Misen's optional SkyPilot extra into the same environment that runs
Misen (not only as an isolated `uv tool`). The Misen extra deliberately
installs the provider-neutral SkyPilot SDK; compose it with only the upstream
provider extras needed by the SkyPilot environment that actually provisions
compute. With SkyPilot's default local API server, that is Misen's environment;
a logged-in remote API server owns its own provider packages and configuration,
so the Misen client can use the base extra alone. The integration declares
`skypilot>=0.12.1` without an upper bound; compatibility CI tests both that
minimum and the newest stable release on Python 3.14. Misen supports Python
3.11–3.14; individual SkyPilot releases and provider extras may impose
additional constraints:

```bash
# AWS and GCP
uv pip install "misen[skypilot]" "skypilot[aws,gcp]>=0.12.1"

# One or more other backends, selected independently
uv pip install "misen[skypilot]" "skypilot[kubernetes,ssh,slurm]>=0.12.1"

# Azure currently needs SkyPilot's documented uv prerequisite
uv pip install --prerelease allow "azure-cli<2.87.0"
uv pip install "misen[skypilot]" "skypilot[azure]>=0.12.1"

sky check

# Developing this repository
uv sync --extra skypilot
uv run --extra skypilot --with "skypilot[runpod]>=0.12.1" sky check runpod
```

Provider selection is passed through unchanged in `executor.infra`:

| Compute target | SkyPilot extra | Example `infra` | Multi-node in SkyPilot 0.13 |
|---|---|---|---|
| AWS | `aws` | `aws/us-east-1` | Yes |
| Google Cloud | `gcp` | `gcp/us-central1` | Yes, with TPU-node caveats |
| Microsoft Azure | `azure` | `azure/eastus` | Yes |
| Oracle Cloud | `oci` | `oci` | Yes |
| Lambda Cloud | `lambda` | `lambda` | Yes |
| RunPod | `runpod` | `runpod` | No |
| Kubernetes | `kubernetes` | `k8s/my-context` | Yes |
| Existing machines | `ssh` | `ssh/my-node-pool` | Yes |
| Slurm through SkyPilot | `slurm` | `slurm/my-cluster/my-partition` | Yes |

An ordered `infra = [...]` list may mix backend families. Values such as
`instance_type` and `image_id` apply to every alternative, so leave them unset
for heterogeneous lists unless the value is portable across every target.

Other SkyPilot providers work the same way: install their named
[upstream extra](https://docs.skypilot.co/en/latest/getting-started/installation.html),
authenticate it, and use its infrastructure string. Azure's CLI currently
needs the additional uv installation step documented upstream. Misen does not
depend on `skypilot[all]`: that would install large, unrelated provider stacks
and authentication clients for every user. Backend features remain
SkyPilot-owned: a `nodes > 1` request (including managed `DASK_CLIENT`) needs a
target with multi-node support. Some targets also cannot host SkyPilot's
managed-jobs controller themselves, so SkyPilot may require another enabled,
controller-capable infrastructure.

Then pair the executor with a `CloudWorkspace`. For example, this provisions
AWS compute while storing Misen's snapshots, payloads, results, locks, and logs
in S3:

```toml
[executor]
type = "skypilot"
infra = "aws/us-east-1"
use_spot = true
snapshot = true
prewarm_envs = false
dask_startup_timeout = 600
dask_scheduler_port = 8786

[executor.accelerators]
cuda = ["A100", "L4"]

[executor.accelerator_memory]
A100 = 40
L4 = 24

[workspace]
type = "cloud"
backend = "s3"
bucket = "my-misen-workspace"
prefix = "experiments"
cache_dir = ".cache/misen"
```

For GCP, use an infrastructure such as `infra = "gcp/us-central1"` and
`backend = "gcs"`. The configured accelerator names are concrete SkyPilot
hardware choices, not Misen accelerator types; CPU-only workflows can omit
both accelerator tables.

The compute backend and workspace object store are independent. For example,
RunPod or Kubernetes compute may use an S3 workspace, and Azure compute may
use GCS, as long as every worker can reach and authenticate to that store.
`executor.infra` controls compute provisioning; `workspace.backend` controls
Misen's snapshots, payloads, results, locks, and logs.

The machine that owns SkyPilot's API server authenticates to the compute
backend (the submitter for the default local server, or the remote server
otherwise). Workers need independent ambient access to the workspace bucket
through an instance role, service account, or equivalent workload identity.
`CloudWorkspace.config` is deliberately not copied into worker bootstrap
commands and cannot be used to carry worker secrets.

The adapter requires `snapshot = true`, `prewarm_envs = false`, a remotely
fetchable workspace transport, and a relative workspace `cache_dir`. It
supports arbitrary Misen DAGs by eagerly submitting one managed job per
pending work unit. Independent branches run in parallel; each dependent
worker waits on durable, submission-scoped workspace markers before entering
user code. This remains durable after submission, but a downstream job may
provision and incur cost while it waits.

DAG parallelism and multi-node task parallelism are separate layers. Misen
can run independent work units as separate SkyPilot managed jobs at the same
time. A work unit that binds `DASK_CLIENT` gets its own temporary Dask cluster
inside its single SkyPilot allocation; Dask does not replace Misen's DAG
scheduler or share workers between work units.

Multi-node requests use SkyPilot's `num_nodes`. Without `DASK_CLIENT`, the
Misen payload runs only on rank 0 and user code owns any additional-node
orchestration. SkyPilot invokes the same wrapper on every node; Misen branches
on `SKYPILOT_NODE_RANK` and discovers the head through `SKYPILOT_NODE_IPS`.
With `DASK_CLIENT`, rank 0 starts a private Dask scheduler and the sole task
coordinator, and every node, including rank 0, starts exactly one worker. CPU,
memory, and accelerator requests are per node; the scheduler and coordinator
share rank 0's resources with its worker. Dask traffic stays on the
allocation's private node network, which must allow every node to reach the
scheduler at `dask_scheduler_port` (default 8786). This allocation-internal
connection is not authenticated or encrypted, so the network must also be
trusted and isolated from other tenants. Do not use managed Dask where an
untrusted workload can reach the scheduler port. Choose a different free port
if the default conflicts with the worker image or network policy (valid range
1024–65535).

Worker images must provide Bash and GNU `timeout`; the normal snapshot
bootstrap also needs worker-side package-index access unless its tools and
caches are pre-provisioned. For managed Dask work, every node independently
fetches and materializes the same immutable snapshot, so its workspace
identity and dependency access must be available throughout the allocation;
the project environment must also include `distributed`. The Dask cluster has
fixed membership rather than elasticity: losing any role fails the SkyPilot
task, while normal scheduler shutdown releases workers on every rank. A Misen
job log reflects the current worker attempt; prior-attempt and non-head
diagnostics remain available in SkyPilot's managed-job logs. A user-code
failure publishes its own dependency marker, but provisioning or
early-bootstrap failures happen before that publisher exists: descendants
learn about them only when the submitting Misen process observes the terminal
SkyPilot status, or eventually fail at their cumulative command timeout.
Those early diagnostics may exist only in SkyPilot's managed-job logs. See the
[remote executor design](https://github.com/apoorvkh/misen/blob/main/docs/design_remote_executors.md) for the decision,
tradeoffs, and roadmap for SSH, remote Slurm, Kubernetes, Modal, and direct
provider Batch adapters.

### Distributed tasks with Dask

Bind `DASK_CLIENT` as a top-level task argument when a multi-node task needs the workers in its allocation. The function itself receives an ordinary [`distributed.Client`](https://distributed.dask.org/en/stable/client.html), so it has no Misen-specific runtime API and can be tested with any Dask client. For example, save this two-node smoke test as `src/my_project/experiments/multinode.py`:

```python
import socket

from distributed import Client
from misen import DASK_CLIENT, Experiment, Task, meta


def hostname() -> str:
    return socket.gethostname()


@meta(cache=False, resources={"nodes": 2, "cpus": 2, "memory": 2, "time": 10})
def check_workers(client: Client) -> dict[str, str]:
    workers = client.run(hostname)
    if len(workers) != 2 or len(set(workers.values())) != 2:
        raise RuntimeError(f"Expected two workers on distinct hosts, got {workers!r}")
    print(f"Dask workers: {workers}")
    return workers


class MultiNodeSmokeTest(Experiment):
    def tasks(self) -> dict[str, Task]:
        return {"workers": Task(check_workers, DASK_CLIENT)}


if __name__ == "__main__":
    MultiNodeSmokeTest.cli()
```

Fill the task id, then submit with either supported multi-node executor:

```bash
uv run misen fill src/my_project/experiments/multinode.py
uv run -m my_project.experiments.multinode --executor slurm run --no-tui
# Or, with a configured CloudWorkspace:
uv run -m my_project.experiments.multinode --executor skypilot run --no-tui
```

Add backend-specific options such as `--executor.partition <name>` for Slurm
or `--executor.infra aws/us-east-1` for SkyPilot when needed. A successful run
prints two worker addresses mapped to two distinct hostnames.

`DASK_CLIENT` requires `nodes > 1`. Misen provisions a private Dask runtime only for pending work units that bind this sentinel. Its client connection opens when the first uncached function resolves the sentinel, is shared by every requesting task in the work unit, and closes when that work unit finishes; task code must not close the client or shut down the cluster. The client represents the work unit's complete fixed allocation, with one Dask worker per node. Every task using it in the same work unit must therefore request exactly the work unit's node and accelerator topology. `cpus`, `memory`, and `accelerators` are per-node quantities.

This is intra-work-unit parallelism, independent of Misen's DAG scheduling.
Two ready Dask-backed work units can run concurrently as two separate backend
allocations; workers are never pooled or shared between them.

Gather futures into ordinary serializable values before returning—Dask clients and futures are tied to the live allocation and are not task results. `SlurmExecutor` and `SkyPilotExecutor` realize `DASK_CLIENT`; `LocalExecutor` and `InProcessExecutor` remain single-node executors and reject it. Unit tests can call the task function directly with a local Dask client.

The coordinator runs once on the first node and should stay lightweight while work executes through the client. Each node has one multi-threaded Dask worker; pure-Python CPU work therefore gains process parallelism across nodes, while within-node execution follows Dask's normal threading semantics. The cluster has fixed membership: the coordinator waits for exactly one worker per node and fails if membership changes. All allocated nodes must share a trusted, mutually reachable compute network because the managed runtime uses Dask's TCP transport without authentication or encryption inside the allocation.

Runtime resources do not contribute to task identity. If node count or accelerator topology affects the logical result, make that choice an ordinary task argument (and therefore part of the task hash) and derive `resources` from it with a resource callback.

Select a compatible backend from the CLI or a config file:

```bash
python -m my_project.experiments.training --executor slurm
```

Both multi-node executors expose `dask_startup_timeout`, the positive number
of seconds allowed while the managed scheduler and complete fixed worker set
start (default 600). SkyPilot additionally exposes `dask_scheduler_port`
(default 8786, valid range 1024–65535), which must be free on rank 0 and
reachable from every allocated node. For Slurm, set cluster-specific fields in
`.misen.toml` (`partition`, `account`, `qos`, `constraint`, plus any
`default_flags`).
Executor `rules` match the resource fields directly, then set local flags such
as `gpu-type`, `partition`, or `constraint`. Allocation-shaping flags such as
`nodes`, `cpus-per-task`, `mem`, `time`, and `gpus-per-node` are owned by Misen
and cannot be overridden this way. For example, this site declares how to
satisfy Project B's request above:

```toml
[[executor.rules]]
[executor.rules.when]
accelerator_memory = 80
accelerator_type = "cuda"
[executor.rules.set]
gpu-type = "a100-80gb"
partition = "gpu"
```

Accelerator count with the default CUDA type needs no rule; specified memory or a non-default type must be covered by matching rules. Configure `LocalExecutor` with `accelerators`, `accelerator_type`, and optional `accelerator_indices`. TPU jobs must request the complete configured count. MPS has no visibility-mask environment variable, so its configured capacity controls scheduling but cannot provide process-level device isolation.

Before dispatching, `misen` takes a **snapshot** of your project: your code (each local package built to a wheel, or an sdist for packages with native extensions) plus dependency metadata (`pyproject.toml`, `uv.lock`, `.python-version`, and pixi manifests). The snapshot is content-hashed and published into the **workspace**, so resubmitting unchanged code stores nothing new, and remote jobs fetch code through the same storage that already carries results and logs — no shared filesystem needed beyond what the workspace itself uses. Jobs stay pinned to the code you submitted while you keep editing. Copies of `.env` files and per-job payloads travel separately as submission-scoped workspace files; they can hold secrets and are retained until workspace pruning or manual removal, while the content-addressed snapshot never contains secrets.

Environments are **materialized from** snapshots into a content-keyed **env store** (`env_store_dir`, default `/tmp/misen-env-store-<user>`): one immutable entry per distinct frozen uv dependency state plus a small overlay venv per distinct code state, and one conda env per `pixi.lock` state — coordinated with NFS-safe locking, so concurrent builders of the same entry build it once. Remote dependencies are installed with `uv sync --frozen --no-install-local`, while the submitted local artifacts go into the overlay. Where and when environments build is executor policy:

- `prewarm_envs = true` (the `LocalExecutor` default): environments build once at submission, on the submitting host; jobs dispatch with direct activation and environment failures surface before anything is queued.
- `prewarm_envs = false` (the `SlurmExecutor` default): each job materializes (or reuses) environments at startup in the env store on its *own* host's local disk — the first job per node pays the build (including dependency downloads, so workers need index access or a warm shared `UV_CACHE_DIR`), and later jobs on that node share it. Native local packages compile on the worker inside the pixi activation, against the locked toolchain and the worker's actual platform. A Bash bootstrap locates `uv` and optional Pixi, runs the workspace's Bash transport for non-path snapshot/job refs, then enters `uv run --with <locked-misen-requirement>` to materialize the environment and execute its Python. Misen may be locked from a package index or Git commit; a local checkout works when its staged artifact is visible through a path workspace.
- SLURM with a *shared* `env_store_dir` plus `prewarm_envs = true` builds everything at submission on the shared filesystem — no worker-side builds and no network needed on compute nodes (the right mode for air-gapped clusters).

Notes:

- Every workspace implements `Workspace.bootstrap_transport()`: path-backed workspaces return `None`, while remote workspaces return Bash source. Misen embeds that source in each submitted worker bootstrap and invokes it with `MISEN_TRANSPORT_OPERATION`, `MISEN_TRANSPORT_REF`, and `MISEN_TRANSPORT_DEST`, plus resolved `MISEN_UV_BIN` and optional project Pixi as `MISEN_PIXI_BIN`. A transport resolves any additional tools it needs itself. Python-backed workspaces can write their fetcher as a normal self-contained function and use `misen.utils.bootstrap_transport.render_python_transport()` to extract it, embed non-secret JSON context, and declare `uv run --with` dependencies automatically. After all data is local, Misen verifies the snapshot and builds or reuses its environment.
- A managed multi-node Dask job runs that same idempotent bootstrap for the scheduler and coordinator on the first node, plus one worker on every node. Cold per-node materialization counts against `dask_startup_timeout`; custom transports must therefore allow the same immutable refs to be fetched concurrently and more than once.
- Slurm still requires its job working directory and job-log path to be visible on the batch node. A remote snapshot/job-file transport removes those data-plane path requirements, but a fully path-free Slurm logging/bootstrap flow remains future work.
- Bash is the root prerequisite for a non-prewarmed snapshot worker. The bootstrap prefers `MISEN_UV_BIN`, then a compatible `uv` on `PATH`; if neither exists, it installs Misen's pinned uv version once under the worker's env store and reuses it. Submitter-side snapshot operations follow the same policy and cache the fallback under `$XDG_DATA_HOME/misen/tools` (or `~/.local/share/misen/tools`). Automatic installation needs outbound HTTPS plus `curl` or `wget`; set `MISEN_UV_AUTO_INSTALL=0` to require a preinstalled uv. After materialization, jobs execute the environment's Python directly, so prewarmed workers do not need uv. Projects with a `pixi.lock` must still provision `pixi` through the executor image, cluster module, container, node setup, or scheduler prologue.
- Bootstrap shell text may be visible in executor or scheduler command lines, so transports must not embed credentials. `CloudWorkspace` reads worker credentials from the ambient environment or workload identity; its generic `config` mapping is intentionally rejected for worker bootstrap dispatch. Dedicated non-secret locator fields such as `endpoint` and `s3_region` remain supported.
- When the default uv/pixi caches sit on a different filesystem than the env store (typical on clusters: home vs. data disk), `misen` points builds at a cache co-located with the store so environments hardlink instead of copying gigabytes. An explicitly set `UV_CACHE_DIR` / `PIXI_CACHE_DIR` is always respected.
- Worker dependency builds use the staged uv project directly, preserving named and explicit indexes, `[tool.uv.sources]`, and the rest of uv's frozen lock semantics.
- Snapshot, environment, and submission job-file stores are not pruned automatically (a reused environment entry's `.complete` marker mtime is touched, so age-based pruning can be added later). It is safe to delete an env store — or the workspace's `snapshots/` — whenever no jobs are queued or running; the next submission republishes and rebuilds. Running `uv cache prune` is also safe: envs hold hardlinks, so their files survive cache pruning.
- Multi-user stores on a shared filesystem need group-writable directories *without* the sticky bit (stale-lock recovery must be able to remove another user's lock files).

The **Workspace** (default: `DiskWorkspace` under `.misen/`) stores cached results, task/job logs, and runtime locks. Cacheable tasks with the same identity are mutually exclusive per Workspace — a concurrent duplicate submission fails fast rather than running twice, and any later submission returns the cached result. A few `Task` methods are useful for scripting around the Workspace: `task.is_cached(...)`, `task.done(...)`, `task.is_running(...)`, and `task.scratch_dir(...)`.

## Configuration

Put defaults in `.misen.toml` (project root) or `$XDG_CONFIG_HOME/misen.toml` (user-wide):

```toml
[executor]
type = "local"
num_cpus = "all"

[workspace]
type = "disk"
directory = ".misen"
```

`./.misen.toml` values override user-level `$XDG_CONFIG_HOME/misen.toml`, and `--config PATH` or `$MISEN_CONFIG` overrides both (`--config` wins over `$MISEN_CONFIG`). An explicit config *replaces* the merge chain entirely — it is not merged on top.

### Environment variables

Project-wide variables go in `.env` (commit it); machine-local overrides and secrets go in `.env.local` (don't commit it — `misen` tightens its permissions to `0600` and reads it after `.env`, so local values win). Both files are auto-loaded when tasks run and copied into owner-only, submission-scoped workspace job files, so SLURM jobs and other remote runs see the same environment as your local shell without placing secrets in content-addressed snapshots.

## System dependencies via Pixi

If your project needs native libraries (CUDA toolkit, compilers, MKL), drop a `pixi.lock` in the project root and `misen` will materialize a matching conda environment alongside your uv venv when taking execution snapshots. PyPI packages stay in `pyproject.toml`; only native/system dependencies belong in `pixi.toml`.

## Static files

Put non-Python files (configs, templates, data) *inside* the package — e.g. `src/my_project/assets/config.yaml` — not at the project root. Files under the package directory are bundled into the wheel, so they ship with `pip install` and are visible to editable, wheel, and zipped installs alike.

Access them at runtime with [`importlib.resources`](https://docs.python.org/3/library/importlib.resources.html), not relative paths from `__file__`:

```python
from importlib.resources import files

config = (files("my_project.assets") / "config.yaml").read_text()
```

## Sharing your work

Because your project is a Python package, anyone can install and reproduce it:

```bash
pip install "git+https://github.com/ORG/REPO.git"
```

```python
from my_project.experiments.training_sweep import TrainingSweep

plot = TrainingSweep()["plot"].result()
```

That's the payoff: artifacts, code, and configuration stay in sync — across iterations, collaborators, and machines.
