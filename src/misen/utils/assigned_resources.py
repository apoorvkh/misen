"""Runtime representation of scheduler-assigned execution resources.

``AssignedResources`` is intentionally a :class:`typing.TypedDict` rather than a
bespoke class so task functions annotated with it remain callable outside
``misen`` — downstream users can construct a plain ``dict`` matching the shape
and invoke the function directly, without importing a framework-specific type.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Mapping
from contextlib import suppress
from typing import Any, TypedDict

from typing_extensions import TypeIs

__all__ = ["AssignedResources", "AssignedResourcesPerNode", "select_local_assigned_resources"]


class AssignedResources(TypedDict):
    """Scheduler-assigned runtime resources for a single node.

    Values:
        cpu_indices: Reserved CPU logical-core indices for the work unit.
        gpu_indices: Reserved GPU device indices, interpreted per the task's
            ``gpu_runtime``.
        memory: Memory cap in GiB, or ``None`` when the scheduler does not
            report one.
        gpu_memory: GPU memory cap in GiB, or ``None`` when unknown.
    """

    cpu_indices: list[int]
    gpu_indices: list[int]
    memory: int | None  # GiB
    gpu_memory: int | None  # GiB


# A multi-node assignment maps hostname -> single-node resources.
AssignedResourcesPerNode = dict[str, AssignedResources]


def select_local_assigned_resources(
    assigned_resources: AssignedResources | AssignedResourcesPerNode | None,
) -> AssignedResources | None:
    """Resolve the node-local assignment from a single-node or per-node payload.

    For per-node payloads, hostname matching is conservative. If no hostname
    match is found and multiple node entries exist, returns ``None`` rather
    than binding potentially incorrect resources.
    """
    if assigned_resources is None or _is_assigned_resources(assigned_resources):
        return assigned_resources
    return _select_from_per_node(assigned_resources)


def _is_assigned_resources(x: Any) -> TypeIs[AssignedResources]:
    """Return whether ``x`` has the shape of a single-node ``AssignedResources``.

    A single-node payload is recognized by the presence of ``cpu_indices`` and
    ``gpu_indices`` at the top level. A per-node payload has arbitrary hostname
    keys; a collision would require a hostname literally named ``cpu_indices``
    or ``gpu_indices``, which is not a real deployment concern. This relaxed
    check is also forward-compatible with future optional fields.
    """
    return isinstance(x, Mapping) and "cpu_indices" in x and "gpu_indices" in x


def _select_from_per_node(per_node: AssignedResourcesPerNode) -> AssignedResources | None:
    """Return the current host's slice from a per-node assignment."""
    for hostname in _candidate_hostnames():
        if hostname in per_node:
            return per_node[hostname]
        short = hostname.split(".", 1)[0]
        if short and short in per_node:
            return per_node[short]

    # Fallback only when payload is single-entry and host metadata is unavailable.
    if len(per_node) == 1:
        return next(iter(per_node.values()))

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
