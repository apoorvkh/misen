# misen

A Python framework for writing **research experiments as end-to-end, reproducible workflows**; not one-off scripts. `misen` offers:

- **End-to-end experiments.** Declare your experiment as a composition of Python functions and let `misen` run the whole thing. No need to run scripts one-at-a-time and glue them together.
  - Experiments are Python classes with typed parameters; you get a CLI, hyperparameter sweeps, and named results for free.
  - `misen` tracks Experiment state (completion, failure) and logs. You can easily check which tasks are complete, failed, and need to be run or updated.

- **Caching.** `misen` caches the outputs of your experiment steps automatically. When you re-run an experiment, the results will be retrieved immediately. You don't have to save outputs to specific filenames and remember what scripts produced them. You can access these results *declaratively* (like `exp["metrics"].result()`) in Python.

- **Reproducibility.** Experiment artifacts are kept in sync with the experiment code. Edit a task and `misen` recomputes exactly everything affected. Whole project replication becomes as easy as running one command.

- **Execution.** `misen` runs your experiments' steps in parallel. You can declare necessary resources (e.g. CPUs, GPUs) per task and `misen` will provision these appropriately. You can run the code on any system; we integrate with different backends, like SLURM. We snapshot your code, so you can freely edit while experiments are queued or running.

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
eval_task  = Task(evaluate, trained_model=train_task.T)
plot_task  = Task(plot, metrics=eval_task.T)
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
        eval_task  = Task(evaluate, trained_model=train_task.T)
        plot_task  = Task(plot, metrics=eval_task.T)
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
    lrs:  list[float] = [0.001, 0.01]
    dims: list[int]   = [256, 512]

    def tasks(self) -> dict[str, Task]:
        metrics = {
            (lr, dim): TrainingExperiment(lr=lr, dim=dim)["metrics"].T
            for lr in self.lrs for dim in self.dims
        }
        return {"plot": Task(plot_sweep, metrics=metrics)}
```

## Resources and Executors

Declare what a task needs:

```python
@meta(id="...", cache=True, resources={"gpus": 1, "memory": 32})
def train(lr: float, dim: int) -> nn.Module: ...
```

Defaults: 1 CPU, 8 GiB RAM, 0 GPUs. Fields: `time`, `memory`, `cpus`, `gpus`, `gpu_memory`, `gpu_runtime` (`"cuda" | "rocm" | "xpu"`).

At runtime, `misen` allocates at least the resources you request and binds them to the task process. `LocalExecutor` masks GPUs via `CUDA_VISIBLE_DEVICES` and pins CPU affinity; `SlurmExecutor` lets SLURM's cgroups handle isolation. Either way, your task code reads the same runtime view — `os.sched_getaffinity(0)` for CPU cores, `range(torch.cuda.device_count())` for GPUs.

CPU affinity and cgroup membership are inherited by children, so subprocesses (`subprocess`, `multiprocessing`) and native threading libraries automatically stay within the allotment. Three patterns to keep in mind:

- **Sizing:** `os.cpu_count()` reports the whole machine. Use `len(os.sched_getaffinity(0))` for pool sizes, `n_jobs`, DataLoader workers, etc.
- **Native threading libs (OpenMP, MKL, OpenBLAS, …):** `LocalExecutor` exports `OMP_NUM_THREADS` and friends to match the assignment. `SlurmExecutor` touches nothing — if you want OpenMP saturation matched to your CPU request, either configure your cluster's `srun` to propagate `SLURM_CPUS_PER_TASK → OMP_NUM_THREADS`, or set it yourself early in the task: `os.environ.setdefault("OMP_NUM_THREADS", str(len(os.sched_getaffinity(0))))`.
- **Libraries that reset affinity at import** (some MKL/NumPy builds, certain CUDA runtimes): re-pin after the offending import with `os.sched_setaffinity(0, os.sched_getaffinity(0))`.

Pass `SCRATCH_DIR: Path` as an argument for a per-task scratch directory. It persists across runs for `cache=True` tasks (useful for checkpointing against preemption) and is ephemeral otherwise.

The **Executor** decides where tasks run:

- `LocalExecutor` — parallel on your machine (default)
- `InProcessExecutor` — single-process, useful in notebooks and tests
- `SlurmExecutor` — submits each work unit as a SLURM job

Switch backends from the CLI or a config file — no code changes:

```bash
python -m my_project.experiments.training --executor-type slurm
```

For SLURM, set cluster-specific fields in `.misen.toml` (`partition`, `account`, `qos`, `constraint`, plus any `default_flags`). For GPUs on a local machine, declare what's available to the executor via `num_cuda_gpus` / `cuda_gpu_indices` (same for `rocm` and `xpu`).

Before dispatching, `misen` takes a **snapshot** of your project — a frozen copy of your source tree, `uv.lock`, `pixi.lock`, and env files. Remote jobs run against the snapshot, so you can keep editing code locally while queued SLURM jobs stay pinned to the version you submitted.

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

Project-wide variables go in `.env` (commit it); machine-local overrides and secrets go in `.env.local` (don't commit it — `misen` tightens its permissions to `0600` and reads it after `.env`, so local values win). Both files are auto-loaded when tasks run *and* copied into execution snapshots, so SLURM jobs and other remote runs see the same environment as your local shell.

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
