"""In-process executor resource validation."""

from typing import TYPE_CHECKING, cast

import pytest

from misen import DASK_CLIENT, SubmissionError, Task, meta
from misen.executors.in_process import InProcessExecutor

if TYPE_CHECKING:
    from misen.workspace import Workspace


@meta(id="in_process_multinode_test_task", resources={"nodes": 2})
def _multinode_task() -> None:
    return None


@meta(id="in_process_dask_test_task")
def _dask_task(client: object) -> None:
    del client


def test_in_process_executor_rejects_multiple_nodes() -> None:
    """An in-process executor cannot realize a multi-node allocation."""
    with pytest.raises(SubmissionError, match="only single-node"):
        InProcessExecutor().submit({Task(_multinode_task)}, cast("Workspace", None))


def test_in_process_executor_rejects_dask_client() -> None:
    """An in-process executor cannot isolate an allocation-scoped Dask runtime."""
    with pytest.raises(SubmissionError, match="cannot provide DASK_CLIENT"):
        InProcessExecutor().submit({Task(_dask_task, DASK_CLIENT)}, cast("Workspace", None))
