---
icon: lucide/network
---

# Architecture

This page documents the core design decisions in `misen` and the boundaries
between the main abstractions.

## Public Surface

The intended public API is:

- `@meta`, `TaskMetadata`, `Resources`, `AcceleratorType`
- `Task`
- `SCRATCH_DIR`, `DASK_CLIENT`
- `Workspace` (`DiskWorkspace` by default; `CloudWorkspace` for remote data)
- `Executor` (`LocalExecutor`, `InProcessExecutor`, `SlurmExecutor`,
  and optional `SkyPilotExecutor`)
- `Experiment`

Most user code should only import from `misen.__init__`.

## Identity Model

Every task instance has three related identities:

- `task_hash`: structure-level identity before dependency resolution.
- `resolved_hash`: identity after dependency outputs are resolved.
- `result_hash`: identity of the computed output when an explicit stable-hash
  handler exists, otherwise resolved task identity.

This separation enables:

- stable deduping of graph structure,
- lock scoping on runtime-resolved inputs,
- cache invalidation without renaming tasks.

## Cache and Scheduling Boundaries

Executors do not schedule individual tasks directly. They schedule `WorkUnit`s:

- A `WorkUnit` is a connected subgraph of non-cacheable tasks.
- Cacheable tasks form boundaries and become `WorkUnit` roots.

This keeps backend scheduling aligned with cache semantics.

## Locking Contract

`Workspace` is the source of truth for concurrency control:

- `namespace="task"` locks enforce one active runtime for a cacheable task
  with a given resolved identity.
- `namespace="result"` locks serialize result materialization.

Backends remain simple because they do not implement custom cache-lock logic.

## Backend/Storage Separation

- `Executor`: graph submission, job lifecycle, backend dispatch.
- `Workspace`: hash/result persistence, locking, task/job logs.

This split allows changing the compute control plane (local, in-process,
SLURM, or a SkyPilot-supported cloud or cluster) without changing cache format
or lock semantics. Remote executors use a remotely fetchable workspace
transport; SkyPilot does not replace the workspace as Misen's data plane.

## Runtime Argument Injection

Sentinels are bound as top-level `Task(...)` arguments and resolved at
execution time:

- `SCRATCH_DIR` -> per-task scratch directory
- `DASK_CLIENT` -> the ordinary `distributed.Client` for a managed multi-node
  allocation

Function signatures stay misen-agnostic: parameters are ordinary `Path` or
`distributed.Client` objects, and the sentinel is bound at task construction,
for example `Task(train, scratch_dir=SCRATCH_DIR)`. Sentinel-valued arguments
are excluded from task identity automatically because the injected value
varies per allocation. Misuse fails at graph-build time: a sentinel left as an
unbound function-signature default, or nested inside a container argument,
raises `TypeError` when the `Task` is constructed — signature defaults are
applied by Python at call time and would bypass the argument resolver, leaking
the raw sentinel object into the function body.

`DASK_CLIENT` requires a task request with `nodes > 1` and is realized by
`SlurmExecutor` and `SkyPilotExecutor`. Misen starts one worker per allocated
node and executes the task body once on the coordinator. With SkyPilot, this
runtime is contained within one `num_nodes` managed job: rank 0 hosts the
scheduler and coordinator, and every rank hosts one worker. `LocalExecutor`
and `InProcessExecutor` intentionally remain single-node executors.
This is intra-work-unit parallelism: the executor still schedules the Misen
DAG as separate work units, and each Dask-backed work unit owns an isolated
temporary cluster rather than sharing a global worker pool.

To discover the resources allotted to a task at runtime, read what the
runtime sees: `os.sched_getaffinity(0)` for CPU cores and the accelerator's
visibility view (e.g. `range(torch.cuda.device_count())`), or the PJRT/XLA
device inventory for a TPU task. For resource requests supported by both
backends, the same task definition runs locally or on SLURM: `LocalExecutor`
applies the configured accelerator type's visibility mask when one exists and
pins CPU affinity on Linux and Windows, while `SlurmExecutor` lets SLURM's
cgroups handle isolation.
Local memory is an aggregate scheduling budget rather than a hard process
limit, and accelerator visibility masks are cooperative runtime controls rather
than a device-access security boundary.

## Serialization

`misen.utils.serde` persists task arguments and results into the workspace.
A type gets a built-in serializer only if it satisfies both:

1. **Faithful round-trip** — the loaded object behaves identically to the
   original at its public API (same Python type, same data, same observable
   methods and attributes). Internal storage detail that no public API
   exposes (e.g. a dask task graph, zarr's on-disk codec) may differ.
2. **Version-stable persistence** — preferred via library-provided save/load
   (`torch.save`, `df.to_parquet`, `model.save_pretrained`); fallback to
   stable formats we drive directly (JSON, GraphML, NPY) where no library
   save exists. We do not call `pickle.dumps` ourselves on arbitrary types,
   and we do not use library save/load paths the library itself documents
   as not portable across versions.

Types that fail either test (`matplotlib.figure.Figure`, `statsmodels.Results`,
`sklearn` estimators, `memoryview`, ...) are intentionally excluded — users
reshape their Task to return something serializable (e.g. `state_dict()`
instead of an `nn.Module`, refit-inputs instead of a fitted sklearn
estimator). The full policy and current exclusion list live in the
`misen.utils.serde.libs` module docstring.

## Why This Design

The model intentionally optimizes for:

- deterministic reproducibility,
- explicit cache behavior,
- backend portability,
- minimal user-facing API complexity.

Experiment parameters should stay declarative. Prefer strings, enums, and
`Literal[...]` values for config choices, and resolve runtime objects inside
task code.
