# API Reference

The sections below document the stable user-facing surface.

Internal modules under `misen.utils.*` are implementation details and may
change without notice.

# Task

::: misen.tasks.Task
    options:
      members:
        - __init__
        - T
        - is_cached
        - are_deps_cached
        - done
        - is_running
        - submit
        - result
        - scratch_dir
        - with_resources

# @meta decorator

::: misen.task_metadata.meta

# Resources

::: misen.task_metadata.Resources

# Runtime sentinels

`SCRATCH_DIR` injects a per-task `pathlib.Path`. `DASK_CLIENT` injects an
allocation-scoped `distributed.Client` for supported multi-node executors.
Both are bound as top-level `Task(...)` arguments and excluded from task
identity.

# DiskWorkspace

::: misen.workspaces.disk.DiskWorkspace

# LocalExecutor

::: misen.executors.local.LocalExecutor

# InProcessExecutor

::: misen.executors.in_process.InProcessExecutor

# SlurmExecutor

::: misen.executors.slurm.SlurmExecutor
    options:
      members:
        - __init__
