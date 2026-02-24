import json
from collections.abc import Mapping

import pytest

from misen.executor import Executor
from misen.executors.local.scheduler import LocalScheduler
from misen.executors.slurm import SlurmExecutor
from misen.utils.assigned_resources import (
    ASSIGNED_RESOURCES_ENV,
    ASSIGNED_RESOURCES_SOURCE_ENV,
    AssignedResources,
    AssignedResourcesResolver,
    AssignedResourcesPerNode,
    SourceT,
    select_local_assigned_resources,
)


class CustomEnvExecutor(Executor):
    @classmethod
    def assigned_resources_source_resolvers(cls) -> Mapping[SourceT, AssignedResourcesResolver]:
        _ = cls
        return {"inline": _custom_assigned_resources}


def _custom_assigned_resources(_: Mapping[str, str]) -> AssignedResources | AssignedResourcesPerNode | None:
    return {
        "cpu_indices": [7],
        "gpu_indices": [],
        "memory": None,
        "gpu_memory": None,
        "gpu_runtime": "cuda",
    }


def test_executor_assigned_resources_source_resolvers_default_inline_only() -> None:
    assert set(Executor.assigned_resources_source_resolvers().keys()) == {"inline"}


def test_slurm_executor_assigned_resources_source_resolvers_register_slurm_sources() -> None:
    assert set(SlurmExecutor.assigned_resources_source_resolvers().keys()) == {"inline", "slurm", "slurm_per_node"}


def test_custom_executor_can_override_assigned_resources_resolvers() -> None:
    assert set(CustomEnvExecutor.assigned_resources_source_resolvers().keys()) == {"inline"}
    resolved = CustomEnvExecutor.assigned_resources_source_resolvers()["inline"]({})
    assert resolved == {
        "cpu_indices": [7],
        "gpu_indices": [],
        "memory": None,
        "gpu_memory": None,
        "gpu_runtime": "cuda",
    }


def test_custom_executor_override_is_used_when_resolving_from_runtime_type() -> None:
    env = {
        Executor.EXECUTOR_ENV_VAR: f"{CustomEnvExecutor.__module__}:{CustomEnvExecutor.__qualname__}",
        ASSIGNED_RESOURCES_SOURCE_ENV: "inline",
        ASSIGNED_RESOURCES_ENV: json.dumps(
            {
                "cpu_indices": [0],
                "gpu_indices": [],
                "memory": None,
                "gpu_memory": None,
                "gpu_runtime": "cuda",
            }
        ),
    }
    assert Executor.resolve_assigned_resources_from_env(env=env) == {
        "cpu_indices": [7],
        "gpu_indices": [],
        "memory": None,
        "gpu_memory": None,
        "gpu_runtime": "cuda",
    }


def test_executor_assigned_resources_from_env_inline() -> None:
    expected = {
        "cpu_indices": [0, 1],
        "gpu_indices": [0],
        "memory": 8,
        "gpu_memory": 16,
        "gpu_runtime": "cuda",
    }
    env = {
        ASSIGNED_RESOURCES_SOURCE_ENV: "inline",
        ASSIGNED_RESOURCES_ENV: json.dumps(expected),
    }
    assert Executor.resolve_assigned_resources_from_env(env=env) == expected


def test_slurm_executor_assigned_resources_parsing() -> None:
    env = {
        ASSIGNED_RESOURCES_SOURCE_ENV: "slurm",
        "SLURM_CPUS_PER_TASK": "2",
        "SLURM_STEP_GPUS": "0,1",
        "SLURM_MEM_PER_NODE": "8192",
        "SLURM_MEM_PER_GPU": "2048",
    }
    assert SlurmExecutor.resolve_assigned_resources_from_env(env=env) == {
        "cpu_indices": [0, 1],
        "gpu_indices": [0, 1],
        "memory": 8,
        "gpu_memory": 2,
        "gpu_runtime": "cuda",
    }


def test_executor_delegates_slurm_source_to_slurm_executor() -> None:
    env = {
        Executor.EXECUTOR_ENV_VAR: f"{SlurmExecutor.__module__}:{SlurmExecutor.__qualname__}",
        ASSIGNED_RESOURCES_SOURCE_ENV: "slurm",
        "SLURM_CPUS_PER_TASK": "3",
        "SLURM_STEP_GPUS": "1",
        "SLURM_MEM_PER_NODE": "4096",
    }
    assert Executor.resolve_assigned_resources_from_env(env=env) == {
        "cpu_indices": [0, 1, 2],
        "gpu_indices": [1],
        "memory": 4,
        "gpu_memory": None,
        "gpu_runtime": "cuda",
    }


def test_executor_assigned_resources_from_env_rejects_malformed_inline_payload() -> None:
    env = {
        ASSIGNED_RESOURCES_SOURCE_ENV: "inline",
        ASSIGNED_RESOURCES_ENV: json.dumps({"cpu_indices": [0]}),
    }
    with pytest.raises(ValueError, match="must contain exactly keys"):
        Executor.resolve_assigned_resources_from_env(env=env)


def test_executor_assigned_resources_from_env_rejects_unknown_source() -> None:
    env = {ASSIGNED_RESOURCES_SOURCE_ENV: "unknown_source"}
    with pytest.raises(ValueError, match="Unknown"):
        Executor.resolve_assigned_resources_from_env(env=env)


def test_select_local_assigned_resources_no_host_match_multinode_returns_none() -> None:
    resources_per_node = {
        "this-host-should-never-match-1": {
            "cpu_indices": [0, 1],
            "gpu_indices": [0],
            "memory": 8,
            "gpu_memory": None,
            "gpu_runtime": "cuda",
        },
        "this-host-should-never-match-2": {
            "cpu_indices": [2, 3],
            "gpu_indices": [1],
            "memory": 8,
            "gpu_memory": None,
            "gpu_runtime": "cuda",
        },
    }
    assert select_local_assigned_resources(resources_per_node) is None


def test_local_scheduler_allocate_indices_consumes_pool() -> None:
    pool = [0, 1, 2]
    allocated = LocalScheduler._allocate_indices(pool, 2)
    assert allocated == [0, 1]
    assert pool == [2]


def test_slurm_executor_parses_gpu_runtime_hint_from_assigned_resources_env() -> None:
    env = {
        ASSIGNED_RESOURCES_SOURCE_ENV: "slurm",
        ASSIGNED_RESOURCES_ENV: json.dumps(
            {
                "cpu_indices": [],
                "gpu_indices": [],
                "memory": None,
                "gpu_memory": None,
                "gpu_runtime": "rocm",
            }
        ),
        "SLURM_CPUS_PER_TASK": "1",
    }
    assert SlurmExecutor.resolve_assigned_resources_from_env(env=env)["gpu_runtime"] == "rocm"
