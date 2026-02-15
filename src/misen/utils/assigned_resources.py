"""Runtime representation of scheduler-assigned execution resources."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["AssignedResources", "get_assigned_resources_slurm"]


class AssignedResources(TypedDict):
    """Scheduler-assigned runtime resources for the current job.

    Note:
        `cpu_indices` / `gpu_indices` represent scheduler-provided physical indices when available.
        Callers can derive logical visibility (e.g., from `CUDA_VISIBLE_DEVICES`) as needed.
    """

    hostnames: tuple[str, ...]
    cpu_count: int | None
    cpu_indices: tuple[int, ...]
    gpu_count: int | None
    gpu_indices: tuple[int, ...]


def get_assigned_resources_slurm() -> AssignedResources:
    """Resolve assigned resources from SLURM environment variables.

    Returns:
        Assigned resources dictionary with host, CPU, and GPU metadata.
    """
    nodelist_keys = ("SLURM_STEP_NODELIST", "SLURM_JOB_NODELIST", "SLURM_NODELIST")
    cpu_count_keys = ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE")
    gpu_count_keys = ("SLURM_GPUS_PER_TASK", "SLURM_GPUS_ON_NODE", "SLURM_JOB_GPUS")
    gpu_index_keys = ("SLURM_STEP_GPUS", "SLURM_JOB_GPUS")

    env = os.environ
    hostnames = _expand_slurm_nodelist(_first_nonempty(env, nodelist_keys))
    if not hostnames and (hostname := env.get("SLURMD_NODENAME")):
        hostnames = (hostname,)

    cpu_indices = _parse_numeric_indices(env.get("SLURM_CPU_BIND_LIST"))
    cpu_count = _first_int(_first_nonempty(env, cpu_count_keys)) or (len(cpu_indices) if cpu_indices else None)
    if not cpu_indices and cpu_count is not None:
        cpu_indices = tuple(range(cpu_count))

    gpu_visible = _first_nonempty(env, gpu_index_keys)
    gpu_indices = _parse_numeric_indices(gpu_visible)
    gpu_count = _first_int(_first_nonempty(env, gpu_count_keys))
    if gpu_count is None:
        if gpu_indices:
            gpu_count = len(gpu_indices)
        elif gpu_visible is not None:
            gpu_count = _count_uuid_tokens(gpu_visible)
    if not gpu_indices and gpu_count is not None and _count_uuid_tokens(gpu_visible or "") == 0:
        # Fallback only when we don't have non-numeric UUID device IDs.
        gpu_indices = tuple(range(gpu_count))

    return AssignedResources(
        hostnames=hostnames,
        cpu_indices=cpu_indices,
        gpu_indices=gpu_indices,
        cpu_count=cpu_count,
        gpu_count=gpu_count,
    )


def _first_nonempty(env: Mapping[str, str], keys: tuple[str, ...]) -> str | None:
    """Return first non-empty environment value among keys."""
    for key in keys:
        if value := env.get(key):
            return value
    return None


def _first_int(value: str | None) -> int | None:
    """Extract first integer token from string."""
    match = re.search(r"\d+", value or "")
    return None if match is None else int(match.group(0))


def _parse_numeric_indices(value: str | None) -> tuple[int, ...]:
    """Parse comma/space-delimited index tokens and ranges.

    Args:
        value: String like ``"0,2-4"``.

    Returns:
        Deduplicated ordered index tuple.
    """
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
    """Count GPU UUID-like tokens in a device-list string."""
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
    """Split CSV string while respecting bracket nesting."""
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
