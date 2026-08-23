"""Public API surface for the ``misen`` task-execution framework.

This package is intentionally split into a small set of composable concepts:

- ``Task``: Lazy computation node with deterministic identity.
- ``Workspace``: Artifact store and lock manager for caching/runtime state.
- ``Executor``: Backend that schedules and runs cache-bounded units of work.
- ``Experiment``: Declarative container of named tasks.

Most user code only needs the symbols re-exported here.
"""

import logging

from misen.exceptions import (
    CacheError,
    CliUsageError,
    ConfigError,
    ErrorCode,
    ExecutionError,
    ExperimentReferenceError,
    HashError,
    JobFailedError,
    JobFailure,
    LockUnavailableError,
    MisenError,
    SerializationError,
    SnapshotError,
    StatusQueryError,
    StorageError,
    SubmissionError,
    SubmittedJob,
    WorkspaceError,
)
from misen.executor import Executor
from misen.experiment import Experiment
from misen.sentinels import DASK_CLIENT, SCRATCH_DIR
from misen.task_metadata import AcceleratorType, Resources, meta
from misen.tasks import Task
from misen.utils.file_map import FileMap
from misen.utils.settings import Settings
from misen.workspace import Workspace

TRACE_LEVEL = logging.DEBUG - 5

if logging.getLevelName(TRACE_LEVEL) != "TRACE":
    logging.addLevelName(TRACE_LEVEL, "TRACE")

logging.getLogger("misen").addHandler(logging.NullHandler())

__all__ = [
    "DASK_CLIENT",
    "SCRATCH_DIR",
    "TRACE_LEVEL",
    "AcceleratorType",
    "CacheError",
    "CliUsageError",
    "ConfigError",
    "ErrorCode",
    "ExecutionError",
    "Executor",
    "Experiment",
    "ExperimentReferenceError",
    "FileMap",
    "HashError",
    "JobFailedError",
    "JobFailure",
    "LockUnavailableError",
    "MisenError",
    "Resources",
    "SerializationError",
    "Settings",
    "SnapshotError",
    "StatusQueryError",
    "StorageError",
    "SubmissionError",
    "SubmittedJob",
    "Task",
    "Workspace",
    "WorkspaceError",
    "meta",
]
