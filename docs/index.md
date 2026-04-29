---
icon: lucide/network
---

# Architecture

This page documents the core design decisions in `misen` and the boundaries
between the main abstractions.

## Public Surface

The intended public API is:

- `@meta`, `TaskMetadata`, `Resources`
- `Task`
- `Workspace` (`DiskWorkspace` by default)
- `Executor` (`LocalExecutor`, `InProcessExecutor`, `SlurmExecutor`)
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

This split allows changing execution backend (local, in-process, SLURM) without
changing cache format or lock semantics.

## Runtime Argument Injection

Sentinel arguments are resolved at execution time:

- `SCRATCH_DIR`
- `ASSIGNED_RESOURCES`
- `ASSIGNED_RESOURCES_PER_NODE`

The same task definition can run locally or on SLURM with backend-specific
resource assignment.

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
