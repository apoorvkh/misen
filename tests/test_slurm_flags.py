from misen.executors.slurm.executor import _resolve_dynamic_sbatch_flags
from misen.executors.slurm.rules import SlurmRule
from misen.task_metadata import Resources


def _resources(gpus: int = 0) -> Resources:
    return {
        "time": None,
        "nodes": 1,
        "memory": 8,
        "cpus": 1,
        "gpus": gpus,
        "gpu_memory": None,
        "gpu_runtime": "cuda",
    }


def test_gpus_requested_without_gpu_type() -> None:
    flags = _resolve_dynamic_sbatch_flags(
        resources=_resources(gpus=2), default_flags={}, rules=[]
    )
    assert "--gpus-per-node=2" in flags


def test_gpus_requested_with_gpu_type_prefix() -> None:
    flags = _resolve_dynamic_sbatch_flags(
        resources=_resources(gpus=4),
        default_flags={"gpu-type": "a100"},
        rules=[],
    )
    assert "--gpus-per-node=a100:4" in flags
    assert not any(f.startswith("--gpu-type") for f in flags)


def test_no_gpu_flag_when_gpus_zero() -> None:
    flags = _resolve_dynamic_sbatch_flags(
        resources=_resources(gpus=0),
        default_flags={"gpu-type": "a100"},
        rules=[],
    )
    assert not any(f.startswith("--gpus-per-node") for f in flags)
    assert not any(f.startswith("--gpu-type") for f in flags)


def test_gpu_type_set_via_rule() -> None:
    flags = _resolve_dynamic_sbatch_flags(
        resources=_resources(gpus=1),
        default_flags={},
        rules=[SlurmRule(when={"gpus": 1}, set={"gpu-type": "h100"})],
    )
    assert "--gpus-per-node=h100:1" in flags
