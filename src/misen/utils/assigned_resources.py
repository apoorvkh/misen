"""Runtime representation of resources assigned by an executor or scheduler."""

from __future__ import annotations

import os
import re
import socket
from typing import TYPE_CHECKING, Literal, TypedDict

import msgspec

if TYPE_CHECKING:
    from collections.abc import Mapping

ASSIGNED_RESOURCES_ENV_VAR = "MISEN_ASSIGNED_RESOURCES"
ASSIGNED_RESOURCES_EXECUTOR_ENV_VAR = "MISEN_EXECUTOR_TYPE"
_SLURM_NODELIST_KEYS = ("SLURM_STEP_NODELIST", "SLURM_JOB_NODELIST", "SLURM_NODELIST")
_SLURM_CPU_COUNT_KEYS = ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE")
_SLURM_GPU_COUNT_KEYS = ("SLURM_GPUS_PER_TASK", "SLURM_GPUS_ON_NODE", "SLURM_JOB_GPUS")
_SLURM_GPU_INDEX_KEYS = ("CUDA_VISIBLE_DEVICES", "SLURM_STEP_GPUS", "SLURM_JOB_GPUS")

__all__ = [
    "ASSIGNED_RESOURCES_ENV_VAR",
    "ASSIGNED_RESOURCES_EXECUTOR_ENV_VAR",
    "AssignedResources",
    "assigned_resources_env",
    "assigned_resources_from_environ",
    "executor_type_from_environ",
    "local_assigned_resources_from_environ",
    "slurm_assigned_resources_from_environ",
]


class AssignedResources(TypedDict):
    """Scheduler-assigned runtime resources for the current job.

    Note:
        `cpu_indices` / `gpu_indices` represent scheduler-provided physical indices when available.
        Callers can derive logical visibility (e.g., from `CUDA_VISIBLE_DEVICES`) as needed.
    """

    executor: Literal["local", "slurm", "unknown"]
    hostnames: tuple[str, ...]
    cpu_indices: tuple[int, ...]
    gpu_indices: tuple[int, ...]
    cpu_count: int | None
    gpu_count: int | None


def assigned_resources_env(resources: AssignedResources) -> dict[str, str]:
    """Serialize assigned resources into environment variables."""
    return {ASSIGNED_RESOURCES_ENV_VAR: msgspec.json.encode(resources).decode("utf-8")}


def assigned_resources_from_environ(env: Mapping[str, str] | None = None) -> AssignedResources:
    """Resolve assigned resources from process environment."""
    env = os.environ if env is None else env
    if executor_type_from_environ(env) == "slurm":
        return slurm_assigned_resources_from_environ(env)
    return local_assigned_resources_from_environ(env)


def executor_type_from_environ(env: Mapping[str, str] | None = None) -> Literal["local", "slurm"]:
    """Infer the active executor type from explicit env marker or scheduler vars."""
    env = os.environ if env is None else env
    if (executor_type := env.get(ASSIGNED_RESOURCES_EXECUTOR_ENV_VAR)) in ("local", "slurm"):
        return executor_type
    return "slurm" if "SLURM_JOB_ID" in env else "local"


def local_assigned_resources_from_environ(env: Mapping[str, str] | None = None) -> AssignedResources:
    """Resolve assigned resources for LocalExecutor jobs."""
    env = os.environ if env is None else env
    if (payload := _decode_assigned_resources(env)) is not None:
        return payload
    hostname = socket.gethostname()
    return _assigned_resources(executor="local", hostnames=(hostname,) if hostname else ())


def slurm_assigned_resources_from_environ(env: Mapping[str, str] | None = None) -> AssignedResources:
    """Resolve assigned resources for SLURM jobs."""
    env = os.environ if env is None else env
    if (payload := _decode_assigned_resources(env)) is not None:
        return payload
    hostnames = _expand_slurm_nodelist(_first_nonempty(env, _SLURM_NODELIST_KEYS))
    if not hostnames and (hostname := env.get("SLURMD_NODENAME")):
        hostnames = (hostname,)

    cpu_indices = _parse_numeric_indices(env.get("SLURM_CPU_BIND_LIST"))
    cpu_count = _first_int(_first_nonempty(env, _SLURM_CPU_COUNT_KEYS)) or (len(cpu_indices) if cpu_indices else None)
    if not cpu_indices and cpu_count is not None:
        cpu_indices = tuple(range(cpu_count))

    gpu_visible = _first_nonempty(env, _SLURM_GPU_INDEX_KEYS)
    gpu_indices = _parse_numeric_indices(gpu_visible)
    gpu_count = _first_int(_first_nonempty(env, _SLURM_GPU_COUNT_KEYS))
    if gpu_count is None:
        if gpu_indices:
            gpu_count = len(gpu_indices)
        elif gpu_visible is not None:
            gpu_count = _count_uuid_tokens(gpu_visible)
    if not gpu_indices and gpu_count is not None and _count_uuid_tokens(gpu_visible or "") == 0:
        # Fallback only when we don't have non-numeric UUID device IDs.
        gpu_indices = tuple(range(gpu_count))

    return _assigned_resources(
        executor="slurm",
        hostnames=hostnames,
        cpu_indices=cpu_indices,
        gpu_indices=gpu_indices,
        cpu_count=cpu_count,
        gpu_count=gpu_count,
    )


def _decode_assigned_resources(env: Mapping[str, str]) -> AssignedResources | None:
    if not (payload := env.get(ASSIGNED_RESOURCES_ENV_VAR)):
        return None
    try:
        return msgspec.json.decode(payload.encode("utf-8"), type=AssignedResources)
    except (msgspec.DecodeError, msgspec.ValidationError):
        return None


def _assigned_resources(
    *,
    executor: Literal["local", "slurm", "unknown"],
    hostnames: tuple[str, ...] = (),
    cpu_indices: tuple[int, ...] = (),
    gpu_indices: tuple[int, ...] = (),
    cpu_count: int | None = None,
    gpu_count: int | None = None,
) -> AssignedResources:
    return {
        "executor": executor,
        "hostnames": hostnames,
        "cpu_indices": cpu_indices,
        "gpu_indices": gpu_indices,
        "cpu_count": cpu_count,
        "gpu_count": gpu_count,
    }


def _first_nonempty(env: Mapping[str, str], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if value := env.get(key):
            return value
    return None


def _first_int(value: str | None) -> int | None:
    match = re.search(r"\d+", value or "")
    return None if match is None else int(match.group(0))


def _parse_numeric_indices(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()

    tokens = [token for token in re.split(r"[,\s]+", value.strip()) if token]
    indices: list[int] = []
    seen: set[int] = set()

    for token in tokens:
        if token.upper().startswith(("GPU-", "MIG-")):
            continue

        for start, end in re.findall(r"(\d+)(?:-(\d+))?", token):
            _start = int(start)
            _end = int(end) if end else _start
            step = 1 if _end >= _start else -1
            for index in range(_start, _end + step, step):
                if index not in seen:
                    indices.append(index)
                    seen.add(index)

    return tuple(indices)


def _count_uuid_tokens(value: str) -> int:
    return sum(1 for token in re.split(r"[,\s]+", value.strip()) if token.upper().startswith(("GPU-", "MIG-")))


def _expand_slurm_nodelist(nodelist: str | None) -> tuple[str, ...]:
    """Expand a SLURM node-list expression into hostnames."""
    if not nodelist:
        return ()

    hosts: list[str] = []

    for token in _split_top_level_csv(nodelist):
        match = re.fullmatch(r"([^\[\],]*)(?:\[([^\]]+)\])?([^\[\],]*)", token)
        if match is None:
            hosts.append(token)
            continue

        prefix, body, suffix = match.groups()
        if body is None:
            hosts.append(token)
            continue

        for part in _split_top_level_csv(body):
            if (range_match := re.fullmatch(r"(\d+)-(\d+)", part)) is None:
                hosts.append(f"{prefix}{part}{suffix}")
                continue

            start_s, end_s = range_match.groups()
            width = max(len(start_s), len(end_s))
            start = int(start_s)
            end = int(end_s)
            step = 1 if end >= start else -1
            hosts.extend(f"{prefix}{index:0{width}d}{suffix}" for index in range(start, end + step, step))

    return tuple(hosts)


def _split_top_level_csv(value: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0

    for i, char in enumerate(value):
        if char == "[":
            depth += 1
        elif char == "]":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            part = value[start:i].strip()
            if part:
                parts.append(part)
            start = i + 1

    tail = value[start:].strip()
    if tail:
        parts.append(tail)

    return parts
