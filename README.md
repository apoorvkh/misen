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

You should run `misen fill` to tag functions with unique ids: e.g. `@meta(id="4fG7Kp2mQ9xR")`. Don't type this in yourself.

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

At runtime, `LocalExecutor` uses declared memory as an aggregate admission budget: concurrent requests stay within `max_memory`, but `LocalExecutor` does not impose a per-job hard memory limit. It subdivides maskable GPU-family devices and injects each assignment's visibility into the job launch environment. These visibility masks and native-thread variables are cooperative controls, not a security boundary; project activation or task code can replace them. A TPU task must request the complete configured pool, which Misen reserves exclusively; framework-specific TPU activation remains the task environment's responsibility. Local memory-per-device constraints are rejected because the current inventory cannot verify them. `SlurmExecutor` maps supported GPU-family counts directly and uses cluster-specific rules to recognize memory and non-default type constraints; an unrecognized constraint is rejected rather than silently under-provisioned. TPU and MPS requests are currently rejected by `SlurmExecutor`. SLURM's cgroups handle isolation.

On Linux and Windows, local CPU affinity is inherited by children; scheduler-provided SLURM cgroup membership is inherited too. Subprocesses (`subprocess`, `multiprocessing`) and native threading libraries therefore normally stay within the allotment. Three patterns to keep in mind:

- **Sizing:** `os.cpu_count()` can report the whole machine. On Linux, use `len(os.sched_getaffinity(0))` for pool sizes, `n_jobs`, DataLoader workers, etc.
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
- `SkyPilotExecutor` — ready-only graph scheduling over explicit reusable or dedicated SkyPilot capacity

Slurm persists native handles for reattachment on later submission. SkyPilot
persists run manifests, logical jobs, attempts, and allocation identities;
`executor.attach(run_id, workspace)` reconstructs observing/cancelling handles
without resubmitting work. Both support `job.cancel()` through their own
scheduler lifecycle.

For a multi-node Slurm or SkyPilot allocation, the task body still runs exactly
once on the first node. Bind `DASK_CLIENT` to use Misen's managed worker group,
with one Dask worker per allocated node, or omit it when the task intentionally
manages its own allocation-scoped runtime. In the latter case, the task also
owns remote-process bootstrap. Slurm can launch into a shared prewarmed
environment with tools such as `srun`; SkyPilot code can use its rank and node
IP environment variables to coordinate the allocation.

### Remote execution with SkyPilot (optional)

SkyPilot manages compute; Misen schedules the ready work in your graph.
One reusable worker agent can execute many work units without another
SkyPilot submission. Shared prerequisites, per-task caching, heterogeneous
resources, and fan-out/fan-in remain Misen responsibilities.

Configure explicit capacity profiles in `.misen.toml`. Each profile selects
exactly one existing `pool`, existing `cluster`, or provisioning `infra`,
and declares a per-worker resource reservation and worker-count limit:

```toml
[executor]
type = "skypilot"
lifecycle = "attached"
manage_api_server = true
api_server_namespace = "dev"
max_run_minutes = 60

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

This example borrows a pool that you create explicitly in the same SkyPilot
namespace. Misen neither creates, resizes, nor terminates borrowed pools or
clusters. Add CPU/GPU profiles for incompatible hardware, or a
`dedicated = true` profile for work requiring its own allocation, including
multi-node tasks. Profile limits bound active allocations; there is no
automatic worker replacement, application retry, or cross-run agent sharing.

The default scoped local API requires the isolated-runtime nightly extra.
Install it in the environment running Misen, together with the provider
packages needed there:

```bash
uv pip install "misen[skypilot-managed]" "skypilot-nightly[aws]==1.0.0.dev20260905"
# From this source checkout instead:
uv sync --extra skypilot-managed
uv pip install "skypilot-nightly[aws]==1.0.0.dev20260905"
```

Do not install both `skypilot` and `skypilot-nightly`. A configured shared or
remote API can instead use `manage_api_server = false` with
`misen[skypilot]`; it is outside Misen's local-service lifecycle.

For a fresh isolated namespace, enable its provider credentials before
launching work or creating a pool:

```python
from misen.executors.skypilot import SkyPilotCapacity, SkyPilotExecutor

executor = SkyPilotExecutor(
    capacity={"cpu": SkyPilotCapacity(pool="misen-dev", cpus=2, memory=4)},
    api_server_namespace="dev",
)
with executor.session() as session:
    session.check(["aws"], verbose=True)
    # Explicit pool provisioning is billable:
    session.pool_apply("misen-dev", "misen-pool.yaml")
```

The [SkyPilot usage guide](docs/skypilot.md) supplies the matching pool YAML,
credential setup, heterogeneous examples, detached requirements, and cleanup
checks. Ordinary `sky check` and `sky jobs pool apply` use a different
namespace from this isolated session.

Attached CLI runs and `Experiment.run()` wait for the graph by default.
Nonblocking Python submissions require a live session:

```python
with executor.session():
    jobs = task.submit(executor=executor, workspace=workspace)
    for job in jobs.nodes():
        job.wait()
        job.raise_for_status()
```

Leaving that session stops admission, cancels unfinished owned work, and
attempts bounded cleanup; it does not silently detach the remaining graph.
The isolated local API server stops after its last session disconnects.
Borrowed workers and shared managed-job controllers can remain billable.

Detached execution is explicit: `lifecycle = "detached"` requires a stable
remote SkyPilot API, service-account credential injection enabled on that
server, a compatible SkyPilot SDK in the project's snapshotted dependencies,
and a dedicated run-owned single-node coordinator profile. Attaching to a
saved run observes/cancels it; it does not take over a lost coordinator.

The implemented worker channel uses known workspace mailbox keys, polled
every 0.2 seconds by default—not an SSH task-dispatch channel. Completion
records release descendants independently of slower SkyPilot allocation
health queries. One agent executes one fresh subprocess at a time and reuses
worker-local environment/artifact caches. Snapshot and cache identity are
unchanged.

This graph executor has local/fake-backend test coverage; the earlier AWS
cold/warm smoke results measured the previous per-work-unit implementation.
They are not a cloud validation or performance guarantee for this architecture.

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

Add `--executor.partition <name>` for Slurm when needed. For SkyPilot, configure
a dedicated two-node profile in `.misen.toml`, for example:

```toml
[executor.capacity.multinode]
infra = "aws/us-east-1"
dedicated = true
nodes = 2
cpus = 2
memory = 2
max_workers = 1
```

A successful run prints two worker addresses mapped to two distinct hostnames.

`DASK_CLIENT` requires `nodes > 1`. Misen provisions a private Dask runtime only for pending work units that bind this sentinel. Its client connection opens when the first uncached function resolves the sentinel, is shared by every requesting task in the work unit, and closes when that work unit finishes; task code must not close the client or shut down the cluster. The client represents the work unit's complete fixed allocation, with one Dask worker per node. Every task using it in the same work unit must therefore request exactly the work unit's node and accelerator topology. `cpus`, `memory`, and `accelerators` are per-node quantities.

This is intra-work-unit parallelism, independent of Misen's DAG scheduling.
Two ready Dask-backed work units use separate dedicated allocations and never
share a live Dask runtime. A compatible pool profile may reuse its underlying
worker cluster after an allocation finishes; `max_workers` bounds concurrency.

Gather futures into ordinary serializable values before returning—Dask clients and futures are tied to the live allocation and are not task results. `SlurmExecutor` and `SkyPilotExecutor` realize `DASK_CLIENT`; `LocalExecutor` and `InProcessExecutor` remain single-node executors and reject it. Unit tests can call the task function directly with a local Dask client.

The coordinator runs once on the first node and should stay lightweight while work executes through the client. Each node has one multi-threaded Dask worker; pure-Python CPU work therefore gains process parallelism across nodes, while within-node execution follows Dask's normal threading semantics. The cluster has fixed membership: the coordinator waits for exactly one worker per node and fails if membership changes. All allocated nodes must share a trusted, mutually reachable compute network because the managed runtime uses Dask's TCP transport without authentication or encryption inside the allocation.

Runtime resources do not contribute to task identity. If node count or accelerator topology affects the logical result, make that choice an ordinary task argument (and therefore part of the task hash) and derive `resources` from it with a resource callback.

Select a compatible backend from the CLI or a config file:

```bash
python -m my_project.experiments.training --executor slurm
```

Slurm exposes `dask_startup_timeout`, the positive number of seconds allowed
while the managed scheduler and fixed worker set start (default 600). The
SkyPilot wrapper currently uses a 300-second Dask startup limit and private
scheduler port 8786; that port must be free on rank 0 and reachable from every
allocated node. For Slurm, set cluster-specific fields in
`.misen.toml` (`partition`, `account`, `qos`, `constraint`, plus any
`default_flags`).
Executor `rules` match the resource fields directly, then set local flags such
as `gpu-type`, `partition`, or `constraint`. Allocation-shaping flags such as
`nodes`, `cpus-per-task`, `mem`, `time`, and `gpus-per-node` are owned by Misen
and cannot be overridden this way. Slurm jobs are requeue-eligible and append
scheduler output across attempts, so their requeue and open-mode flags are also
executor-owned. For example, this site declares how to
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
- A managed multi-node Dask job runs that same idempotent bootstrap for the scheduler and coordinator on the first node, plus one worker on every node. Cold per-node materialization counts against the runtime startup allowance; custom transports must therefore allow the same immutable refs to be fetched concurrently and more than once.
- Slurm still requires its job working directory and job-log path to be visible on the batch node. A remote snapshot/job-file transport removes those data-plane path requirements, but a fully path-free Slurm logging/bootstrap flow remains future work.
- Bash is the root prerequisite for a non-prewarmed snapshot worker. The bootstrap prefers `MISEN_UV_BIN`, then a compatible `uv` on `PATH`; if neither exists, it installs Misen's pinned uv version once under the worker's env store and reuses it. Submitter-side snapshot operations follow the same policy and cache the fallback under `$XDG_DATA_HOME/misen/tools` (or `~/.local/share/misen/tools`). Automatic installation needs outbound HTTPS plus `curl` or `wget`; set `MISEN_UV_AUTO_INSTALL=0` to require a preinstalled uv. After materialization, jobs execute the environment's Python directly, so prewarmed workers do not need uv. Projects with a `pixi.lock` must still provision `pixi` through the executor image, cluster module, container, node setup, or scheduler prologue.
- Bootstrap shell text may be visible in executor or scheduler command lines, so transports must not embed credentials. `CloudWorkspace` reads worker credentials from the ambient environment or workload identity; its generic `config` mapping is intentionally rejected for worker bootstrap dispatch. Dedicated non-secret locator fields such as `endpoint` and `s3_region` remain supported.
- When the default uv/pixi caches sit on a different filesystem than the env store (typical on clusters: home vs. data disk), `misen` points builds at a cache co-located with the store so environments hardlink instead of copying gigabytes. An explicitly set `UV_CACHE_DIR` / `PIXI_CACHE_DIR` is always respected.
- Worker dependency builds use the staged uv project directly, preserving named and explicit indexes, `[tool.uv.sources]`, and the rest of uv's frozen lock semantics.
- Snapshot, environment, and submission job-file stores are not pruned automatically (a reused environment entry's `.complete` marker mtime is touched, so age-based pruning can be added later). It is safe to delete an env store — or the workspace's `snapshots/` — whenever no jobs are queued or running; the next submission republishes and rebuilds. Running `uv cache prune` is also safe: envs hold hardlinks, so their files survive cache pruning.
- Multi-user stores on a shared filesystem need group-writable directories *without* the sticky bit (stale-lock recovery must be able to remove another user's lock files).

The **Workspace** (default: `DiskWorkspace` under `.misen/`) stores cached results, task/job logs, and runtime locks. Cacheable tasks with the same identity are mutually exclusive per Workspace — a duplicate waits for the renewable lease (or its stale expiry), then returns the cached result if the first attempt committed. A few `Task` methods are useful for scripting around the Workspace: `task.is_cached(...)`, `task.done(...)`, `task.is_running(...)`, and `task.scratch_dir(...)`.

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
