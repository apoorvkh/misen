"""Runtime representation of scheduler-assigned execution resources."""

from __future__ import annotations

import os
import socket
from collections.abc import Mapping
from contextlib import suppress
from typing import Any, TypedDict

from typing_extensions import TypeIs

__all__ = ["AssignedResources", "AssignedResourcesPerNode", "select_local_assigned_resources"]


class AssignedResources(TypedDict):
    """Scheduler-assigned runtime resources for a single node."""

    cpu_indices: list[int]
    gpu_indices: list[int]
    memory: int | None  # GiB
    gpu_memory: int | None  # GiB


AssignedResourcesPerNode = dict[str, AssignedResources]


def _is_assigned_resources(x: Any) -> TypeIs[AssignedResources]:
    """Return whether ``x`` has the exact shape of ``AssignedResources``."""
    return isinstance(x, Mapping) and set(x.keys()) == set(AssignedResources.__annotations__)


def select_local_assigned_resources(
    assigned_resources: AssignedResources | AssignedResourcesPerNode | None,
) -> AssignedResources | None:
    """Resolve node-local assignment from a single-node or per-node payload.

    For per-node payloads, hostname matching is conservative. If no hostname
    match is found and multiple node entries exist, returns ``None`` rather than
    binding potentially incorrect resources.
    """
    if assigned_resources is None or _is_assigned_resources(assigned_resources):
        return assigned_resources

    for hostname in _candidate_hostnames():
        if hostname in assigned_resources:
            return assigned_resources[hostname]
        short = hostname.split(".", 1)[0]
        if short and short in assigned_resources:
            return assigned_resources[short]

    # Fallback only when payload is single-node and host metadata is unavailable.
    if len(assigned_resources) == 1:
        return next(iter(assigned_resources.values()))

    return None


def _candidate_hostnames() -> tuple[str, ...]:
    """Return hostname candidates in priority order for per-node lookup."""
    hostnames: list[str] = []
    if "HOSTNAME" in os.environ:
        hostnames.append(os.environ["HOSTNAME"])
    with suppress(OSError):
        hostnames.append(socket.gethostname())
    with suppress(OSError):
        hostnames.append(socket.getfqdn())
    return tuple(dict.fromkeys(hostnames))  # preserve order, dedupe
