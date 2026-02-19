"""Public API surface for the ``misen`` task-execution framework.

This package is intentionally split into a small set of composable concepts:

- ``Task``: Lazy computation node with deterministic identity.
- ``Workspace``: Artifact store and lock manager for caching/runtime state.
- ``Executor``: Backend that schedules and runs cache-bounded units of work.
- ``Experiment``: Declarative container of named tasks.

Most user code only needs the symbols re-exported here.
"""

from misen.executor import Executor
from misen.experiment import Experiment
from misen.sentinels import ASSIGNED_RESOURCES, ASSIGNED_RESOURCES_PER_NODE, WORK_DIR
from misen.task_properties import Resources, TaskProperties, task
from misen.tasks import Task
from misen.utils.settings import Settings
from misen.workspace import Workspace

__all__ = [
    "ASSIGNED_RESOURCES",
    "ASSIGNED_RESOURCES_PER_NODE",
    "WORK_DIR",
    "Executor",
    "Experiment",
    "Resources",
    "Settings",
    "Task",
    "TaskProperties",
    "Workspace",
    "task",
]
