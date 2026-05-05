import sys

import pytest

from misen import Task, meta
from misen.exceptions import CacheError
from misen.utils.hashing import TaskHash
from misen.utils.work_unit import WorkUnit
from misen.workspaces.disk import DiskWorkspace, LMDBMapping


@meta(id="log_task_a", cache=True)
def log_task_a() -> int:
    return 1


@meta(id="log_task_b", cache=True)
def log_task_b() -> int:
    return 2


@meta(id="log_task_source", cache=False)
def log_task_source() -> int:
    return 1


@meta(id="log_task_sink", cache=False)
def log_task_sink(value: int) -> int:
    sys.stdout.write(f"sink {value}\n")
    return value


def test_job_log_iter_filters_by_work_unit(tmp_path) -> None:
    workspace = DiskWorkspace(directory=str(tmp_path / ".misen"))
    work_unit_a = WorkUnit(root=Task(log_task_a), dependencies=set())
    work_unit_b = WorkUnit(root=Task(log_task_b), dependencies=set())

    log_a_1 = workspace.get_job_log(job_id="job-a1", work_unit=work_unit_a)
    log_a_2 = workspace.get_job_log(job_id="job-a2", work_unit=work_unit_a)
    log_b_1 = workspace.get_job_log(job_id="job-b1", work_unit=work_unit_b)

    log_a_1.write_text("a1")
    log_a_2.write_text("a2")
    log_b_1.write_text("b1")

    assert set(workspace.job_log_iter(work_unit=work_unit_a)) == {log_a_1, log_a_2}
    assert set(workspace.job_log_iter()) == {log_a_1, log_a_2, log_b_1}


def test_task_logs_require_resolvable_dependencies(tmp_path) -> None:
    """Task logs are keyed by ``resolved_hash``; lookup before deps are
    computed surfaces ``CacheError`` rather than silently using a path that
    a later run would collide with.
    """
    workspace = DiskWorkspace(directory=str(tmp_path / ".misen-task-logs"))
    source_task = Task(log_task_source)
    sink_task = Task(log_task_sink, value=source_task.T)

    with pytest.raises(CacheError):
        sink_task.resolved_hash(workspace=workspace)
    with pytest.raises(CacheError):
        workspace.get_task_log(task=sink_task, job_id="job-live")
    with pytest.raises(CacheError):
        workspace.read_task_log(task=sink_task, job_id="job-live")


def test_work_unit_downstream_task_log_uses_resolved_hash(tmp_path) -> None:
    workspace = DiskWorkspace(directory=str(tmp_path / ".misen-work-unit-task-logs"))
    source_task = Task(log_task_source)
    sink_task = Task(log_task_sink, value=source_task.T)
    work_unit = WorkUnit(root=sink_task, dependencies=set())

    WorkUnit.execute(work_unit.graph, workspace=workspace, job_id="job-live")

    key = sink_task.resolved_hash(workspace=workspace).b32()
    log_path = tmp_path / ".misen-work-unit-task-logs" / "task_logs" / key[:2] / f"{key}_job-live.log"
    assert log_path.read_text(encoding="utf-8") == "sink 1\n"


def test_disk_workspace_close_releases_lmdb_and_blocks_further_use(tmp_path) -> None:
    workspace = DiskWorkspace(directory=str(tmp_path / ".misen-close"))
    cache = workspace._resolved_hash_cache  # noqa: SLF001
    assert isinstance(cache, LMDBMapping)

    workspace.close()
    # Idempotent.
    workspace.close()

    sample_key = TaskHash.from_object(("close-test",))
    with pytest.raises(RuntimeError, match="closed"):
        _ = sample_key in cache
    with pytest.raises(RuntimeError, match="closed"):
        _ = len(cache)
