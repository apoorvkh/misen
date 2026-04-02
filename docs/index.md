---
icon: lucide/network
---

# Architecture

This page documents the core design decisions in `misen` and the boundaries
between the main abstractions.

## Public Surface

The intended public API is:

- `@task`, `TaskProperties`, `Resources`
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

- `WORK_DIR`
- `ASSIGNED_RESOURCES`
- `ASSIGNED_RESOURCES_PER_NODE`

The same task definition can run locally or on SLURM with backend-specific
resource assignment.

## Why This Design

The model intentionally optimizes for:

- deterministic reproducibility,
- explicit cache behavior,
- backend portability,
- minimal user-facing API complexity.

Experiment parameters should stay declarative. Prefer strings, enums, and
`Literal[...]` values for config choices, and resolve runtime objects inside
task code.
