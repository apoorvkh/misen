"""Data models for SLURM rule-based flag resolution."""

from __future__ import annotations

from typing import Literal, TypeAlias

import msgspec

ResourceKey: TypeAlias = Literal["time", "nodes", "memory", "cpus", "gpus", "gpu_memory", "gpu_runtime"]
OperatorName: TypeAlias = Literal["eq", "ne", "lt", "le", "gt", "ge", "contains", "is_", "is_not"]


class ResourcePredicate(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    """One predicate against a resource value."""

    op: OperatorName
    value: int | str | list[int | str] | None = None


ResourceCondition: TypeAlias = int | str | None | ResourcePredicate | list[ResourcePredicate]
SetValue: TypeAlias = str | int | float | bool | None | list[str]


class SlurmRule(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    """One conditional sbatch-flag override rule."""

    when: dict[ResourceKey, ResourceCondition] = {}
    set: dict[str, SetValue] = {}
