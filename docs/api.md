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
        - work_dir

# @task decorator

::: misen.task_properties.task

# Resources

::: misen.task_properties.Resources

# DiskWorkspace

::: misen.workspaces.disk.DiskWorkspace

# LocalExecutor

::: misen.executors.local.LocalExecutor

# SlurmExecutor

::: misen.executors.slurm.SlurmExecutor
    options:
      members:
        - __init__
