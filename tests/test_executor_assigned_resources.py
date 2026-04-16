import socket

from misen.executors.local.scheduler import LocalScheduler
from misen.executors.slurm.parsing import (
    get_assigned_resources_slurm,
    get_assigned_resources_slurm_per_node,
)
from misen.utils.assigned_resources import AssignedResources, select_local_assigned_resources


def _resources(
    *,
    cpu_indices: list[int],
    gpu_indices: list[int],
    memory: int | None,
    gpu_memory: int | None,
) -> AssignedResources:
    return AssignedResources(
        cpu_indices=cpu_indices,
        gpu_indices=gpu_indices,
        memory=memory,
        gpu_memory=gpu_memory,
    )


def test_get_assigned_resources_slurm_parses_cpu_gpu_and_memory() -> None:
    env = {
        "SLURM_CPUS_PER_TASK": "2",
        "SLURM_STEP_GPUS": "0,1",
        "SLURM_MEM_PER_NODE": "8192",
        "SLURM_MEM_PER_GPU": "2048",
    }
    assert get_assigned_resources_slurm(env=env) == _resources(
        cpu_indices=[0, 1],
        gpu_indices=[0, 1],
        memory=8,
        gpu_memory=2,
    )


def test_get_assigned_resources_slurm_uses_uuid_tokens_without_synthesizing_indices() -> None:
    env = {
        "SLURM_GPUS_ON_NODE": "2",
        "SLURM_STEP_GPUS": "GPU-abc,GPU-def",
    }
    assert get_assigned_resources_slurm(env=env) == _resources(
        cpu_indices=[],
        gpu_indices=[],
        memory=None,
        gpu_memory=None,
    )


def test_get_assigned_resources_slurm_per_node_expands_nodelist() -> None:
    env = {
        "SLURM_JOB_NODELIST": "node[01-02],node05",
        "SLURM_CPUS_PER_TASK": "3",
        "SLURM_STEP_GPUS": "1",
        "SLURM_MEM_PER_NODE": "4096",
    }
    expected = _resources(
        cpu_indices=[0, 1, 2],
        gpu_indices=[1],
        memory=4,
        gpu_memory=None,
    )
    assert get_assigned_resources_slurm_per_node(env=env) == {
        "node01": expected,
        "node02": expected,
        "node05": expected,
    }


def test_select_local_assigned_resources_returns_single_node_payload() -> None:
    assigned_resources = _resources(
        cpu_indices=[0, 1],
        gpu_indices=[0],
        memory=8,
        gpu_memory=None,
    )
    assert select_local_assigned_resources(assigned_resources) == assigned_resources


def test_select_local_assigned_resources_prefers_hostname_from_env(monkeypatch) -> None:
    monkeypatch.setenv("HOSTNAME", "trainer-2")
    assigned_resources = {
        "trainer-1": _resources(cpu_indices=[0], gpu_indices=[], memory=4, gpu_memory=None),
        "trainer-2": _resources(cpu_indices=[1], gpu_indices=[0], memory=8, gpu_memory=16),
    }
    assert select_local_assigned_resources(assigned_resources) == _resources(
        cpu_indices=[1],
        gpu_indices=[0],
        memory=8,
        gpu_memory=16,
    )


def test_select_local_assigned_resources_falls_back_to_short_hostname(monkeypatch) -> None:
    monkeypatch.setenv("HOSTNAME", "trainer-2.cluster.local")
    assigned_resources = {
        "trainer-2": _resources(cpu_indices=[1], gpu_indices=[0], memory=8, gpu_memory=16),
    }
    assert select_local_assigned_resources(assigned_resources) == _resources(
        cpu_indices=[1],
        gpu_indices=[0],
        memory=8,
        gpu_memory=16,
    )


def test_select_local_assigned_resources_single_entry_fallback_when_host_unknown(monkeypatch) -> None:
    monkeypatch.delenv("HOSTNAME", raising=False)
    monkeypatch.setattr(socket, "gethostname", lambda: "unknown-host")
    monkeypatch.setattr(socket, "getfqdn", lambda: "unknown-host.domain")

    only_resources = _resources(cpu_indices=[3], gpu_indices=[], memory=2, gpu_memory=None)
    assert select_local_assigned_resources({"node-does-not-match": only_resources}) == only_resources


def test_select_local_assigned_resources_no_host_match_multinode_returns_none(monkeypatch) -> None:
    monkeypatch.delenv("HOSTNAME", raising=False)
    monkeypatch.setattr(socket, "gethostname", lambda: "unknown-host")
    monkeypatch.setattr(socket, "getfqdn", lambda: "unknown-host.domain")

    resources_per_node = {
        "node-a": _resources(cpu_indices=[0], gpu_indices=[], memory=4, gpu_memory=None),
        "node-b": _resources(cpu_indices=[1], gpu_indices=[0], memory=8, gpu_memory=16),
    }
    assert select_local_assigned_resources(resources_per_node) is None


def test_local_scheduler_allocate_indices_consumes_pool() -> None:
    pool = [0, 1, 2]
    allocated = LocalScheduler._allocate_indices(pool, 2)
    assert allocated == [0, 1]
    assert pool == [2]


def test_local_scheduler_allocate_indices_returns_none_when_insufficient() -> None:
    pool = [0]
    assert LocalScheduler._allocate_indices(pool, 2) is None
    assert pool == [0]
