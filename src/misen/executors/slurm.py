"""SLURM-backed executor implementation."""

from __future__ import annotations

import logging
import operator
import os
import re
import shlex
import shutil
import subprocess
from functools import cache
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

import msgspec

from misen.executor import Executor, Job
from misen.utils.assigned_resources import AssignedResources, AssignedResourcesPerNode
from misen.utils.runtime_events import work_unit_label
from misen.utils.snapshot import LocalSnapshot

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from misen.utils.work_unit import WorkUnit
    from misen.workspace import Workspace

__all__ = ("SlurmExecutor", "SlurmJob")

logger = logging.getLogger(__name__)


class SlurmJob(Job):
    """Job handle backed by a SLURM job id."""

    __slots__ = ("slurm_job_id",)

    def __init__(self, work_unit: WorkUnit, job_id: str, slurm_job_id: str, log_path: Path) -> None:
        """Initialize SLURM job wrapper."""
        super().__init__(work_unit=work_unit, job_id=job_id, log_path=log_path)
        self.slurm_job_id = slurm_job_id

    def state(self) -> _State:
        """Return the current SLURM state, normalized to a misen job state."""
        for command, args in (
            ("squeue", ["-h", "-j", self.slurm_job_id, "-o", "%T"]),
            ("sacct", ["-n", "-j", self.slurm_job_id, "--format=State"]),
        ):
            result = subprocess.run(  # noqa: S603
                [_resolve_slurm_cmd(command), *args],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0 or not (output := result.stdout.strip()):
                continue

            state = output.splitlines()[0].strip()
            if command == "sacct":
                state = state.split()[0]
            state = state.upper().split("+", maxsplit=1)[0].split(":", maxsplit=1)[0]
            return _SLURM_STATE_MAP.get(state, "unknown")

        return "unknown"


class SlurmExecutor(Executor[SlurmJob, LocalSnapshot]):
    """Executor that submits work units to SLURM via ``sbatch``."""

    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    constraint: str | None = None
    default_flags: dict[str, _SetValue] = msgspec.field(default_factory=dict)
    rules: list[_SlurmRule] = msgspec.field(default_factory=list)

    def __post_init__(self) -> None:
        """Normalize untyped config into msgspec structs."""
        self.default_flags = msgspec.convert(self.default_flags, type=dict[str, _SetValue])
        self.rules = msgspec.convert(self.rules, type=list[_SlurmRule])

    def _make_snapshot(self, workspace: Workspace) -> LocalSnapshot:
        return self._make_local_snapshot(workspace=workspace)

    def _dispatch(
        self,
        work_unit: WorkUnit,
        dependencies: set[SlurmJob],
        workspace: Workspace,
        snapshot: LocalSnapshot,
    ) -> SlurmJob:
        """Submit one work unit to SLURM."""
        resources = work_unit.resources
        label = work_unit_label(work_unit)
        logger.info("Submitting SLURM work unit %s with %d dependency job(s).", label, len(dependencies))

        sbatch_cmd = [
            _resolve_slurm_cmd("sbatch"),
            "--parsable",
            "--job-name",
            f"misen-{work_unit.root.task_hash().short_b32()}",
            "--ntasks-per-node",
            "1",
            "--nodes",
            str(resources["nodes"]),
            "--cpus-per-task",
            str(resources["cpus"]),
            "--mem",
            f"{resources['memory']}G",
        ]
        if resources["time"] is not None:
            sbatch_cmd.extend(["--time", str(resources["time"])])

        flags = dict(self.default_flags)
        flags.update(
            {
                key: value
                for key, value in (
                    ("partition", self.partition),
                    ("account", self.account),
                    ("qos", self.qos),
                    ("constraint", self.constraint),
                )
                if value is not None
            }
        )
        for rule in self.rules:
            if all(_condition_matches(resources[key], condition) for key, condition in rule.when.items()):
                flags.update(rule.set)

        gpu_type = flags.pop("gpu-type", None)
        if resources["gpus"] > 0:
            flags["gpus-per-node"] = f"{gpu_type}:{resources['gpus']}" if gpu_type else resources["gpus"]

        for flag in sorted(flags):
            value = flags[flag]
            if value is None or value is False:
                continue
            if value is True:
                sbatch_cmd.append(f"--{flag}")
            elif isinstance(value, list):
                sbatch_cmd.extend(f"--{flag}={item}" for item in value)
            else:
                sbatch_cmd.append(f"--{flag}={value}")

        if dependencies:
            sbatch_cmd.extend(["--dependency", f"afterok:{':'.join(job.slurm_job_id for job in dependencies)}"])

        job_id, argv, env_overrides = snapshot.prepare_job(
            work_unit=work_unit,
            workspace=workspace,
            assigned_resources_getter=_assigned_resources_slurm_per_node
            if resources["nodes"] > 1
            else _assigned_resources_slurm,
            gpu_runtime=resources["gpu_runtime"],
        )
        log_path = workspace.get_job_log(job_id=job_id, work_unit=work_unit)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        wrapped = [*argv]
        if env_overrides:
            wrapped = ["env", *(f"{key}={value}" for key, value in env_overrides.items()), *wrapped]
        sbatch_cmd.extend(["--output", str(log_path), "--export", "ALL", "--wrap", shlex.join(wrapped)])
        logger.debug("sbatch command for %s: %s", label, shlex.join(sbatch_cmd))

        try:
            result = subprocess.run(sbatch_cmd, check=True, capture_output=True, text=True)  # noqa: S603
        except subprocess.CalledProcessError as e:
            logger.exception("sbatch failed while submitting %s.", label)
            msg = f"sbatch failed: {(e.stderr or e.stdout or '').strip()}"
            raise RuntimeError(msg) from e

        output = result.stdout.strip()
        slurm_job_id = output.split(";", 1)[0].split(None, 1)[0] if output else ""
        if not slurm_job_id.isdigit():
            msg = f"Unexpected sbatch output: {output!r}"
            logger.error("%s", msg)
            raise RuntimeError(msg)

        logger.info("Submitted SLURM work unit %s (job_id=%s, slurm_job_id=%s).", label, job_id, slurm_job_id)
        return SlurmJob(work_unit=work_unit, job_id=job_id, slurm_job_id=slurm_job_id, log_path=log_path)


_State = Literal["pending", "running", "done", "failed", "unknown"]
_ResourceKey: TypeAlias = Literal["time", "nodes", "memory", "cpus", "gpus", "gpu_memory", "gpu_runtime"]
_OperatorName: TypeAlias = Literal["eq", "ne", "lt", "le", "gt", "ge", "contains", "is_", "is_not"]
_SetValue: TypeAlias = str | int | float | bool | None | list[str]


class _ResourcePredicate(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    """One predicate against a resource value."""

    op: _OperatorName
    value: int | str | list[int | str] | None = None


_ResourceCondition: TypeAlias = int | str | None | _ResourcePredicate | list[_ResourcePredicate]


class _SlurmRule(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    """One conditional sbatch-flag override rule."""

    when: dict[_ResourceKey, _ResourceCondition] = msgspec.field(default_factory=dict)
    set: dict[str, _SetValue] = msgspec.field(default_factory=dict)


_SLURM_STATE_MAP: dict[str, _State] = {
    **dict.fromkeys(("PENDING", "CONFIGURING", "SUSPENDED", "REQUEUED", "REQUEUED_HOLD", "STAGE_OUT"), "pending"),
    **dict.fromkeys(("RUNNING", "COMPLETING"), "running"),
    **dict.fromkeys(
        (
            "BOOT_FAIL",
            "CANCELLED",
            "DEADLINE",
            "FAILED",
            "NODE_FAIL",
            "OUT_OF_MEMORY",
            "PREEMPTED",
            "TIMEOUT",
            "TIMEOUT_SIGNAL",
            "SPECIAL_EXIT",
        ),
        "failed",
    ),
    "COMPLETED": "done",
}


def _condition_matches(value: int | str | None, condition: _ResourceCondition) -> bool:
    if isinstance(condition, list):
        condition = cast("list[_ResourcePredicate]", condition)
        return all(_predicate_matches(value, predicate) for predicate in condition)
    if isinstance(condition, _ResourcePredicate):
        return _predicate_matches(value, condition)
    return value is None if condition is None else value == condition


def _predicate_matches(value: int | str | None, predicate: _ResourcePredicate) -> bool:
    op = getattr(operator, predicate.op)
    rhs = predicate.value

    if predicate.op == "contains":
        if not isinstance(rhs, list):
            msg = "Predicate op='contains' expects `value` to be a list."
            raise TypeError(msg)
        return value is not None and bool(op(rhs, value))

    if predicate.op in {"eq", "ne"}:
        if not isinstance(rhs, (int, str)):
            msg = f"Predicate op={predicate.op!r} expects `value` to be an integer or string."
            raise TypeError(msg)
        return value is not None and bool(op(value, rhs))

    if predicate.op in {"lt", "le", "gt", "ge"}:
        if not isinstance(rhs, int):
            msg = f"Predicate op={predicate.op!r} expects `value` to be an integer."
            raise TypeError(msg)
        return isinstance(value, int) and bool(op(value, rhs))

    if isinstance(rhs, list):
        msg = f"Predicate op={predicate.op!r} does not accept list `value`."
        raise TypeError(msg)
    return bool(op(value, rhs))


@cache
def _resolve_slurm_cmd(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        msg = f"Required command {name!r} not found on PATH. Is SLURM installed on this system?"
        raise FileNotFoundError(msg)
    return path


def _assigned_resources_slurm(env: Mapping[str, str] | None = None) -> AssignedResources:
    """Parse node-local resources from SLURM environment variables."""
    env = os.environ if env is None else env

    cpu_indices = _parse_numeric_indices(env.get("SLURM_CPU_BIND_LIST"))
    cpu_count = None
    for key in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE"):
        if match := re.search(r"\d+", env.get(key, "")):
            cpu_count = int(match.group(0))
            break
    if cpu_count is None and cpu_indices:
        cpu_count = len(cpu_indices)
    if not cpu_indices and cpu_count is not None:
        cpu_indices = list(range(cpu_count))

    gpu_visible = next((value for key in ("SLURM_STEP_GPUS", "SLURM_JOB_GPUS") if (value := env.get(key))), None)
    gpu_indices = _parse_numeric_indices(gpu_visible)
    gpu_count = None
    for key in ("SLURM_GPUS_PER_TASK", "SLURM_GPUS_ON_NODE"):
        if match := re.search(r"\d+", env.get(key, "")):
            gpu_count = int(match.group(0))
            break

    uuid_count = sum(
        1 for token in re.split(r"[,\s]+", (gpu_visible or "").strip()) if token.upper().startswith(("GPU-", "MIG-"))
    )
    if gpu_count is None:
        gpu_count = len(gpu_indices) or uuid_count or None
    if not gpu_indices and gpu_count is not None and uuid_count == 0:
        gpu_indices = list(range(gpu_count))

    return AssignedResources(
        cpu_indices=cpu_indices,
        gpu_indices=gpu_indices,
        memory=_parse_slurm_memory_to_gib(
            next((value for key in ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU") if (value := env.get(key))), None)
        ),
        gpu_memory=_parse_slurm_memory_to_gib(env.get("SLURM_MEM_PER_GPU")),
    )


def _assigned_resources_slurm_per_node(env: Mapping[str, str] | None = None) -> AssignedResourcesPerNode:
    """Parse host -> resources from SLURM environment variables."""
    env = os.environ if env is None else env
    hostnames = _expand_slurm_nodelist(
        next(
            (
                value
                for key in ("SLURM_STEP_NODELIST", "SLURM_JOB_NODELIST", "SLURM_NODELIST")
                if (value := env.get(key))
            ),
            None,
        )
    )
    if not hostnames and (hostname := env.get("SLURMD_NODENAME")):
        hostnames = (hostname,)

    resources = _assigned_resources_slurm(env)
    return {
        hostname: AssignedResources(
            cpu_indices=[*resources["cpu_indices"]],
            gpu_indices=[*resources["gpu_indices"]],
            memory=resources["memory"],
            gpu_memory=resources["gpu_memory"],
        )
        for hostname in hostnames
    }


def _parse_slurm_memory_to_gib(value: str | None) -> int | None:
    """Parse a SLURM memory token as GiB, rounding up."""
    if not value or not (value := value.strip().upper()):
        return None

    match = re.fullmatch(r"(\d+)\s*([KMGT]?)(?:I?B)?", value)
    if match is None:
        match = re.search(r"\d+", value)
        return None if match is None else (int(match.group(0)) + 1023) // 1024

    amount = int(match.group(1))
    match match.group(2):
        case "":
            return (amount + 1023) // 1024
        case "K":
            return max(1, (amount + 1024 * 1024 - 1) // (1024 * 1024))
        case "M":
            return max(1, (amount + 1023) // 1024)
        case "G":
            return amount
        case "T":
            return amount * 1024
    return None


def _parse_numeric_indices(value: str | None) -> list[int]:
    """Parse comma/space-delimited index tokens and ranges like ``0,2-4``."""
    if not value:
        return []

    indices: list[int] = []
    seen: set[int] = set()
    for token in (token for token in re.split(r"[,\s]+", value.strip()) if token):
        if token.upper().startswith(("GPU-", "MIG-")):
            continue
        for start_s, end_s in re.findall(r"(\d+)(?:-(\d+))?", token):
            start = int(start_s)
            end = int(end_s) if end_s else start
            step = 1 if end >= start else -1
            for index in range(start, end + step, step):
                if index not in seen:
                    seen.add(index)
                    indices.append(index)
    return indices


def _expand_slurm_nodelist(nodelist: str | None) -> tuple[str, ...]:
    """Expand a SLURM node-list expression into hostnames."""
    if not nodelist:
        return ()

    hosts: list[str] = []
    for token in _split_top_level_csv(nodelist):
        match = re.fullmatch(r"([^\[\],]*)(?:\[([^\]]+)\])?([^\[\],]*)", token)
        if match is None or match.group(2) is None:
            hosts.append(token)
            continue

        prefix, body, suffix = match.groups()
        for part in _split_top_level_csv(body):
            if range_match := re.fullmatch(r"(\d+)-(\d+)", part):
                start_s, end_s = range_match.groups()
                width = max(len(start_s), len(end_s))
                start, end = int(start_s), int(end_s)
                step = 1 if end >= start else -1
                hosts.extend(f"{prefix}{index:0{width}d}{suffix}" for index in range(start, end + step, step))
            else:
                hosts.append(f"{prefix}{part}{suffix}")
    return tuple(hosts)


def _split_top_level_csv(value: str) -> list[str]:
    """Split CSV while respecting bracket nesting."""
    parts: list[str] = []
    depth = start = 0
    for index, char in enumerate(value):
        if char == "[":
            depth += 1
        elif char == "]":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            if part := value[start:index].strip():
                parts.append(part)
            start = index + 1
    if tail := value[start:].strip():
        parts.append(tail)
    return parts
