import pytest

from misen import Task, meta
from misen.utils.hashing import TaskHash
from misen.utils.work_unit import WorkUnit
from misen.workspaces.disk import DiskWorkspace, LMDBMapping


@meta(id="log_task_a", cache=True)
def log_task_a() -> int:
    return 1


@meta(id="log_task_b", cache=True)
def log_task_b() -> int:
    return 2


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
