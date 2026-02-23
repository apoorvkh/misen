"""Runtime representation of scheduler-assigned execution resources."""

from __future__ import annotations

import json
import os
import re
import socket
from collections.abc import Mapping
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

from typing_extensions import TypeIs, assert_never

if TYPE_CHECKING:
    from misen.task_properties import GpuRuntime

__all__ = [
    "AssignedResources",
    "AssignedResourcesPerNode",
    "build_assigned_resources_env",
    "get_assigned_resources_from_env",
    "get_assigned_resources_slurm",
    "get_assigned_resources_slurm_per_node",
    "get_gpu_runtime_from_env",
    "select_local_assigned_resources",
]


class AssignedResources(TypedDict):
    """Scheduler-assigned runtime resources for a single node."""

    cpu_indices: list[int]
    gpu_indices: list[int]
    memory: int | None  # GiB
    gpu_memory: int | None  # GiB


AssignedResourcesPerNode = dict[str, AssignedResources]

_ASSIGNED_RESOURCES_KEYS = frozenset(AssignedResources.__annotations__)

SourceT = Literal["inline", "slurm", "slurm_per_node"]

_ASSIGNED_RESOURCES_ENV = "MISEN_ASSIGNED_RESOURCES"
_ASSIGNED_RESOURCES_SOURCE_ENV = "MISEN_ASSIGNED_RESOURCES_SOURCE"
_GPU_RUNTIME_ENV = "MISEN_GPU_RUNTIME"


def _is_assigned_resources(x: Any) -> TypeIs[AssignedResources]:
    return isinstance(x, Mapping) and set(x.keys()) == _ASSIGNED_RESOURCES_KEYS


def build_assigned_resources_env(
    assigned_resources: AssignedResources | AssignedResourcesPerNode | None,
    gpu_runtime: GpuRuntime,
    source: SourceT,
) -> dict[str, str]:
    """Serialize runtime resource metadata for launched child processes."""
    env: dict[str, str] = {_GPU_RUNTIME_ENV: gpu_runtime}

    match source:
        case "inline":
            if assigned_resources is not None:
                env[_ASSIGNED_RESOURCES_SOURCE_ENV] = "inline"
                env[_ASSIGNED_RESOURCES_ENV] = json.dumps(assigned_resources)
        case _:
            # For scheduler-derived modes, child resolves directly from env at runtime.
            env[_ASSIGNED_RESOURCES_SOURCE_ENV] = source

    return env


def get_assigned_resources_from_env() -> AssignedResources | AssignedResourcesPerNode | None:
    """Resolve assigned resources configured by executors for this process."""
    if _ASSIGNED_RESOURCES_SOURCE_ENV in os.environ:
        source = cast("SourceT", os.environ[_ASSIGNED_RESOURCES_SOURCE_ENV])

        match source:
            case "inline":
                if _ASSIGNED_RESOURCES_ENV in os.environ:
                    return json.loads(os.environ[_ASSIGNED_RESOURCES_ENV])
            case "slurm":
                return get_assigned_resources_slurm()
            case "slurm_per_node":
                return get_assigned_resources_slurm_per_node()
            case _:
                assert_never(source)

    return None


def get_gpu_runtime_from_env(default: GpuRuntime = "cuda") -> GpuRuntime:
    """Resolve requested GPU runtime from process environment."""
    return cast("GpuRuntime", os.environ.get(_GPU_RUNTIME_ENV, default))


def select_local_assigned_resources(
    assigned_resources: AssignedResources | AssignedResourcesPerNode | None,
) -> AssignedResources | None:
    """Resolve node-local assignment from a single-node or per-node payload."""
    if assigned_resources is None or _is_assigned_resources(assigned_resources):
        return assigned_resources

    if len(assigned_resources) == 0:
        return None

    for hostname in _candidate_hostnames():
        if hostname in assigned_resources:
            return assigned_resources[hostname]
        short = hostname.split(".", 1)[0]
        if short and short in assigned_resources:
            return assigned_resources[short]

    # Fallback: any entry (stable insertion order)
    return next(iter(assigned_resources.values()))


def get_assigned_resources_slurm() -> AssignedResources:
    """Resolve assigned resources from SLURM environment variables.

    This is a best-effort parser:
      - may return partial information
      - may synthesize cpu/gpu indices from counts when explicit indices are unavailable
      - memory values are normalized to GiB
    """
    env = os.environ

    # ----------------------------
    # CPU assignment
    # ----------------------------
    # SLURM_CPU_BIND_LIST can vary by bind mode; parse only clearly numeric/range-like forms.
    cpu_bind_list = env.get("SLURM_CPU_BIND_LIST")
    cpu_indices = _parse_numeric_indices(cpu_bind_list)

    cpu_count = _first_int(
        _first_nonempty(
            env,
            ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE"),
        )
    )

    if cpu_count is None and cpu_indices:
        cpu_count = len(cpu_indices)

    if not cpu_indices and cpu_count is not None:
        # Synthesized local indices; not guaranteed to reflect true physical pinning.
        cpu_indices = list(range(cpu_count))

    # ----------------------------
    # GPU assignment
    # ----------------------------
    gpu_visible = _first_nonempty(env, ("SLURM_STEP_GPUS", "SLURM_JOB_GPUS"))
    gpu_indices = _parse_numeric_indices(gpu_visible)

    gpu_count = _first_int(_first_nonempty(env, ("SLURM_GPUS_PER_TASK", "SLURM_GPUS_ON_NODE", "SLURM_JOB_GPUS")))

    if gpu_count is None:
        if gpu_indices:
            gpu_count = len(gpu_indices)
        elif gpu_visible:
            gpu_count = _count_uuid_tokens(gpu_visible)

    # Only synthesize 0..N-1 when device IDs are numeric-like (not UUID/MIG IDs).
    if not gpu_indices and gpu_count is not None and _count_uuid_tokens(gpu_visible or "") == 0:
        gpu_indices = list(range(gpu_count))

    # ----------------------------
    # Memory (normalized to GiB)
    # ----------------------------
    # NOTE: Bare integers in SLURM memory env vars are typically MB/MiB-ish in practice.
    # We convert to GiB with ceiling to avoid under-reporting.
    memory = _parse_slurm_memory_to_gib(_first_nonempty(env, ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU")))
    gpu_memory = _parse_slurm_memory_to_gib(env.get("SLURM_MEM_PER_GPU"))

    return AssignedResources(
        cpu_indices=cpu_indices,
        gpu_indices=gpu_indices,
        memory=memory,
        gpu_memory=gpu_memory,
    )


def get_assigned_resources_slurm_per_node() -> AssignedResourcesPerNode:
    """Resolve host->resources mapping from SLURM environment variables.

    This is typically a shared per-node template (derived from local env), not a true
    node-specific allocation parser.
    """
    env = os.environ
    hostnames = _expand_slurm_nodelist(
        _first_nonempty(env, ("SLURM_STEP_NODELIST", "SLURM_JOB_NODELIST", "SLURM_NODELIST"))
    )
    if not hostnames and (hostname := env.get("SLURMD_NODENAME")):
        hostnames = (hostname,)

    if not hostnames:
        return {}

    resources = get_assigned_resources_slurm()

    return {hostname: cast("AssignedResources", dict(resources)) for hostname in hostnames}


def _candidate_hostnames() -> tuple[str, ...]:
    hostnames: list[str] = [v for k in ("SLURMD_NODENAME", "HOSTNAME") if (v := os.environ.get(k))]
    with suppress(OSError):
        hostnames.append(socket.gethostname())
    with suppress(OSError):
        hostnames.append(socket.getfqdn())
    return tuple(dict.fromkeys(hostnames))  # preserve order, dedupe


def _parse_slurm_memory_to_gib(value: str | None) -> int | None:
    """Parse a SLURM memory token and return GiB (rounded up).

    Accepts examples like:
      - "8192"   (treated as MiB/MB-ish; converted to 8 GiB)
      - "8192M"  / "8192MB"
      - "8G"     / "8GB"
      - "1T"     / "1TB"
      - "500K"   / "500KB" (rounds up to 1 GiB)

    Returns None when parsing fails.
    """
    if not value:
        return None

    s = value.strip().upper()
    if not s:
        return None

    # SLURM may sometimes include non-digit suffix text; parse leading numeric+unit.
    match = re.fullmatch(r"(\d+)\s*([KMGT]?)(?:I?B)?", s)
    if match is None:
        # Fallback: find first integer; treat as MiB-ish to avoid drastic overestimates.
        raw = _first_int(s)
        return None if raw is None else _ceil_div(raw, 1024)

    amount = int(match.group(1))
    unit = match.group(2)

    # Convert to GiB with ceiling.
    match unit:
        case "":
            # Bare integers in SLURM memory env vars are commonly MB/MiB-ish.
            return _ceil_div(amount, 1024)
        case "K":
            # KiB -> GiB
            return max(1, _ceil_div(amount, 1024 * 1024))
        case "M":
            # MiB -> GiB
            return max(1, _ceil_div(amount, 1024))
        case "G":
            return amount
        case "T":
            return amount * 1024
        case _:
            return None


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
    """Parse comma/space-delimited index tokens and ranges like '0,2-4'.

    Non-numeric GPU UUID/MIG tokens (e.g. 'GPU-...', 'MIG-...') are ignored.
    This is best-effort and intentionally does not parse hex masks.
    """
    if not value:
        return []

    indices: list[int] = []
    seen: set[int] = set()

    for token in (t for t in re.split(r"[,\s]+", value.strip()) if t):
        # Skip UUID-style device IDs (e.g. GPU-..., MIG-...)
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
