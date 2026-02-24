"""Parsing helpers for SLURM-assigned resource metadata."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

    from misen.utils.assigned_resources import AssignedResources, AssignedResourcesPerNode


def get_assigned_resources_slurm(env: Mapping[str, str] | None = None) -> AssignedResources:
    """Parse node-local assigned resources from SLURM environment variables.

    This is a best-effort parser:
      - may return partial information
      - may synthesize cpu/gpu indices from counts when explicit indices are unavailable
      - memory values are normalized to GiB
    """
    resolved_env = os.environ if env is None else env

    # ----------------------------
    # CPU assignment
    # ----------------------------
    cpu_bind_list = resolved_env.get("SLURM_CPU_BIND_LIST")
    cpu_indices = _parse_numeric_indices(cpu_bind_list)

    cpu_count = _first_int(
        _first_nonempty(
            resolved_env,
            ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE"),
        )
    )

    if cpu_count is None and cpu_indices:
        cpu_count = len(cpu_indices)

    if not cpu_indices and cpu_count is not None:
        cpu_indices = list(range(cpu_count))

    # ----------------------------
    # GPU assignment
    # ----------------------------
    gpu_visible = _first_nonempty(resolved_env, ("SLURM_STEP_GPUS", "SLURM_JOB_GPUS"))
    gpu_indices = _parse_numeric_indices(gpu_visible)

    gpu_count = _first_int(
        _first_nonempty(
            resolved_env,
            ("SLURM_GPUS_PER_TASK", "SLURM_GPUS_ON_NODE", "SLURM_JOB_GPUS"),
        )
    )

    if gpu_count is None:
        if gpu_indices:
            gpu_count = len(gpu_indices)
        elif gpu_visible:
            gpu_count = _count_uuid_tokens(gpu_visible)

    if not gpu_indices and gpu_count is not None and _count_uuid_tokens(gpu_visible or "") == 0:
        gpu_indices = list(range(gpu_count))

    # ----------------------------
    # Memory (normalized to GiB)
    # ----------------------------
    memory = _parse_slurm_memory_to_gib(_first_nonempty(resolved_env, ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU")))
    gpu_memory = _parse_slurm_memory_to_gib(resolved_env.get("SLURM_MEM_PER_GPU"))

    return {
        "cpu_indices": cpu_indices,
        "gpu_indices": gpu_indices,
        "memory": memory,
        "gpu_memory": gpu_memory,
    }


def get_assigned_resources_slurm_per_node(env: Mapping[str, str] | None = None) -> AssignedResourcesPerNode:
    """Parse host->resources mapping from SLURM environment variables."""
    resolved_env = os.environ if env is None else env

    hostnames = _expand_slurm_nodelist(
        _first_nonempty(
            resolved_env,
            ("SLURM_STEP_NODELIST", "SLURM_JOB_NODELIST", "SLURM_NODELIST"),
        )
    )
    if not hostnames and (hostname := resolved_env.get("SLURMD_NODENAME")):
        hostnames = (hostname,)

    if not hostnames:
        return {}

    resources = get_assigned_resources_slurm(env=resolved_env)
    return {
        hostname: cast(
            "AssignedResources",
            {
                "cpu_indices": [*resources["cpu_indices"]],
                "gpu_indices": [*resources["gpu_indices"]],
                "memory": resources["memory"],
                "gpu_memory": resources["gpu_memory"],
            },
        )
        for hostname in hostnames
    }


def _parse_slurm_memory_to_gib(value: str | None) -> int | None:
    """Parse a SLURM memory token and return GiB (rounded up)."""
    amount_gib: int | None = None
    if value:
        normalized_value = value.strip().upper()
        if normalized_value:
            memory_match = re.fullmatch(r"(\d+)\s*([KMGT]?)(?:I?B)?", normalized_value)
            if memory_match is None:
                raw = _first_int(normalized_value)
                if raw is not None:
                    amount_gib = _ceil_div(raw, 1024)
            else:
                amount = int(memory_match.group(1))
                unit = memory_match.group(2)

                if unit == "":
                    amount_gib = _ceil_div(amount, 1024)
                elif unit == "K":
                    amount_gib = max(1, _ceil_div(amount, 1024 * 1024))
                elif unit == "M":
                    amount_gib = max(1, _ceil_div(amount, 1024))
                elif unit == "G":
                    amount_gib = amount
                elif unit == "T":
                    amount_gib = amount * 1024

    return amount_gib


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _first_nonempty(env: Mapping[str, str], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if value := env.get(key):
            return value
    return None


def _first_int(value: str | None) -> int | None:
    match = re.search(r"\d+", value or "")
    return None if match is None else int(match.group(0))


def _parse_numeric_indices(value: str | None) -> list[int]:
    """Parse comma/space-delimited index tokens and ranges like ``0,2-4``."""
    if not value:
        return []

    indices: list[int] = []
    seen: set[int] = set()

    for token in (t for t in re.split(r"[,\s]+", value.strip()) if t):
        if token.upper().startswith(("GPU-", "MIG-")):
            continue

        for start_s, end_s in re.findall(r"(\d+)(?:-(\d+))?", token):
            start = int(start_s)
            end = int(end_s) if end_s else start
            step = 1 if end >= start else -1
            for idx in range(start, end + step, step):
                if idx not in seen:
                    seen.add(idx)
                    indices.append(idx)

    return indices


def _count_uuid_tokens(value: str) -> int:
    return sum(1 for t in re.split(r"[,\s]+", value.strip()) if t.upper().startswith(("GPU-", "MIG-")))


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
            range_match = re.fullmatch(r"(\d+)-(\d+)", part)
            if range_match is None:
                hosts.append(f"{prefix}{part}{suffix}")
                continue

            start_s, end_s = range_match.groups()
            width = max(len(start_s), len(end_s))
            start, end = int(start_s), int(end_s)
            step = 1 if end >= start else -1
            hosts.extend(f"{prefix}{i:0{width}d}{suffix}" for i in range(start, end + step, step))

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
