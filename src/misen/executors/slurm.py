"""SLURM-backed executor implementation.

This backend delegates scheduling and dependency coordination to SLURM itself.
The misen side is responsible for:

- packaging work-unit payloads via snapshots
- translating resource requests into ``sbatch`` arguments
- mapping SLURM state strings to generic job states
"""

from __future__ import annotations

import operator
import shlex
import shutil
import subprocess
from functools import cache
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

import msgspec

from misen.executor import Executor, Job
from misen.utils.assigned_resources import get_assigned_resources_slurm, get_assigned_resources_slurm_per_node
from misen.utils.runtime_events import runtime_event, work_unit_label
from misen.utils.snapshot import LocalSnapshot

if TYPE_CHECKING:
    from pathlib import Path

    from misen.task_properties import Resources
    from misen.utils.work_unit import WorkUnit
    from misen.workspace import Workspace


class SlurmJob(Job):
    """Job handle that maps SLURM command output to misen job states."""

    __slots__ = ("slurm_job_id",)

    def __init__(self, work_unit: WorkUnit, job_id: str, slurm_job_id: str, log_path: Path) -> None:
        """Initialize SLURM job wrapper.

        Args:
            work_unit: Work unit associated with this job.
            job_id: Misen-internal job identifier.
            slurm_job_id: Job id returned by ``sbatch``.
            log_path: Path where SLURM writes stdout/stderr.
        """
        super().__init__(work_unit=work_unit, job_id=job_id, log_path=log_path)
        self.slurm_job_id = slurm_job_id

    def state(self) -> Literal["pending", "running", "done", "failed", "unknown"]:
        """Return current state by querying SLURM CLI.

        Returns:
            Generic job state derived from SLURM job state strings.
        """
        state = _query_slurm_state(self.slurm_job_id)

        if state is None:
            return "unknown"

        state = state.strip().upper()
        state = state.split("+", maxsplit=1)[0]
        state = state.split(":", maxsplit=1)[0]

        match state:
            case "PENDING" | "CONFIGURING" | "SUSPENDED" | "REQUEUED" | "REQUEUED_HOLD" | "STAGE_OUT":
                return "pending"
            case "RUNNING" | "COMPLETING":
                return "running"
            case (
                "BOOT_FAIL"
                | "CANCELLED"
                | "DEADLINE"
                | "FAILED"
                | "NODE_FAIL"
                | "OUT_OF_MEMORY"
                | "PREEMPTED"
                | "TIMEOUT"
                | "TIMEOUT_SIGNAL"
                | "SPECIAL_EXIT"
            ):
                return "failed"
            case "COMPLETED":
                return "done"
        return "unknown"


class SlurmExecutor(Executor[SlurmJob, LocalSnapshot]):
    """Executor that submits work units to SLURM via ``sbatch``."""

    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    constraint: str | None = None
    default_flags: dict[str, SetValue] = {}  # noqa: RUF012
    rules: list[SlurmRule] = []  # noqa: RUF012

    def __post_init__(self) -> None:
        """Load rules as Struct from untyped kwargs."""
        self.default_flags = msgspec.convert(self.default_flags, type=dict[str, SetValue])
        self.rules = msgspec.convert(self.rules, type=list[SlurmRule])

    @classmethod
    @cache
    def _cached_snapshot(cls, snapshots_dir: Path) -> LocalSnapshot:
        """Return cached snapshot instance for a snapshot directory."""
        return LocalSnapshot(snapshots_dir=snapshots_dir)

    def _make_snapshot(self, workspace: Workspace) -> LocalSnapshot:
        """Create or reuse snapshot for SLURM submission.

        Args:
            workspace: Workspace used to locate snapshots directory.

        Returns:
            Snapshot instance.
        """
        snapshots_dir = (workspace.get_temp_dir() / "snapshots").resolve()
        return SlurmExecutor._cached_snapshot(snapshots_dir=snapshots_dir)

    def _dispatch(
        self, work_unit: WorkUnit, dependencies: set[SlurmJob], workspace: Workspace, snapshot: LocalSnapshot
    ) -> SlurmJob:
        """Dispatch one work unit to SLURM.

        Args:
            work_unit: Work unit to submit.
            dependencies: Upstream SLURM jobs that must finish successfully.
            workspace: Workspace for logs/artifacts.
            snapshot: Snapshot used to build payload command.

        Returns:
            Submitted SLURM job handle.

        Raises:
            RuntimeError: If ``sbatch`` fails or returns unexpected output.
        """
        work_unit_repr = work_unit.root.task_hash().short_b32()
        resources = work_unit.resources

        sbatch_cmd: list[str] = [_resolve_slurm_cmd("sbatch"), "--parsable"]
        sbatch_cmd.extend(["--job-name", f"misen-{work_unit_repr}"])
        sbatch_cmd.extend(["--ntasks-per-node", "1"])
        sbatch_cmd.extend(["--nodes", str(resources.nodes)])
        sbatch_cmd.extend(["--cpus-per-task", str(resources.cpus)])
        sbatch_cmd.extend(["--mem", f"{resources.memory}G"])
        sbatch_cmd.extend(["--time", str(resources.time or 1)])

        resolved_default_flags = dict(self.default_flags)
        if self.partition is not None:
            resolved_default_flags["partition"] = self.partition
        if self.account is not None:
            resolved_default_flags["account"] = self.account
        if self.qos is not None:
            resolved_default_flags["qos"] = self.qos
        if self.constraint is not None:
            resolved_default_flags["constraint"] = self.constraint

        sbatch_cmd.extend(
            _resolve_dynamic_sbatch_flags(
                resources=resources,
                default_flags=resolved_default_flags,
                rules=self.rules,
            )
        )

        if dependencies:
            dep_ids = ":".join(job.slurm_job_id for job in dependencies)
            sbatch_cmd.extend(["--dependency", f"afterok:{dep_ids}"])

        job_id, argv, env_overrides = snapshot.prepare_job(
            work_unit=work_unit,
            workspace=workspace,
            assigned_resources_getter=get_assigned_resources_slurm_per_node
            if resources.nodes > 1
            else get_assigned_resources_slurm,
        )

        job_log_path = workspace.get_job_log(job_id=job_id, work_unit=work_unit)
        sbatch_cmd.extend(["--output", str(job_log_path)])

        env_prefix = ["env", *[f"{k}={v}" for k, v in env_overrides.items()]]
        sbatch_cmd.extend(["--export", "ALL"])
        sbatch_cmd.extend(["--wrap", shlex.join([*env_prefix, *argv])])

        try:
            result = subprocess.run(sbatch_cmd, check=True, capture_output=True, text=True)  # noqa: S603
        except subprocess.CalledProcessError as e:
            msg = f"sbatch failed: {(e.stderr or e.stdout or '').strip()}"
            raise RuntimeError(msg) from e

        out = result.stdout.strip()
        slurm_job_id = out.split(";", 1)[0].split(None, 1)[0]
        if not slurm_job_id.isdigit():
            msg = f"Unexpected sbatch output: {out!r}"
            raise RuntimeError(msg)
        runtime_event(
            (f"SLURM job submitted: {work_unit_label(work_unit)} (job_id={job_id}, slurm_job_id={slurm_job_id})"),
            style="green",
        )
        return SlurmJob(work_unit=work_unit, job_id=job_id, slurm_job_id=slurm_job_id, log_path=job_log_path)


ResourceKey: TypeAlias = Literal["time", "nodes", "memory", "cpus", "gpus", "gpu_memory", "gpu_vendor"]
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


def _resolve_dynamic_sbatch_flags(
    resources: Resources,
    default_flags: dict[str, SetValue],
    rules: list[SlurmRule],
) -> list[str]:
    """Resolve sbatch flags by applying matching rules over defaults.

    Semantics:
      - Start from `default_flags` (copied into a mutable dict).
      - For each rule in order:
          * if rule.when matches `resources`, apply `rule.set` onto the flag dict
            (simple overwrite per key).
      - Render the resulting dict into a list of `--flag` / `--flag=value` args.

    Condition semantics:
      - `when: {"gpus": 1}` means gpus == 1
      - `when: {"gpu_memory": None}` means gpu_memory is null (None)
      - `when: {"gpu_vendor": "nvidia"}` means gpu_vendor == "nvidia"
      - Predicate form supports operator names from Python's `operator` module:
        eq, ne, lt, le, gt, ge, contains, is_, is_not.

    Value semantics:
      - None => omit flag
      - True => emit `--flag`
      - False => omit `--flag`
      - list[str] => emit repeated flags: `--flag=item` for each element
    """
    resolved: dict[str, SetValue] = dict(default_flags)
    for rule in rules:
        if _rule_matches(resources, rule.when):
            resolved |= rule.set

    if "gpu-type" in resolved and resources.gpus > 0:
        resolved["gpus-per-node"] = f"{resolved['gpu-type']}:{resources.gpus}"

    return _render_flags(resolved)


def _rule_matches(resources: Resources, when: dict[ResourceKey, ResourceCondition]) -> bool:
    for key, cond in when.items():
        v = getattr(resources, key)
        if not _match_condition(v, cond):
            return False
    return True


def _match_condition(v: int | str | None, cond: ResourceCondition) -> bool:
    if isinstance(cond, list):
        cond = cast("list[ResourcePredicate]", cond)
        return all(_match_predicate(v, pred) for pred in cond)
    if isinstance(cond, ResourcePredicate):
        return _match_predicate(v, cond)
    if cond is None:
        return operator.is_(v, None)
    return operator.eq(v, cond)


def _match_predicate(v: int | str | None, pred: ResourcePredicate) -> bool:
    op = getattr(operator, pred.op)
    rhs = pred.value

    if pred.op == "contains":
        if not isinstance(rhs, list):
            msg = "Predicate op='contains' expects `value` to be a list."
            raise TypeError(msg)
        return False if v is None else bool(op(rhs, v))

    if pred.op in {"eq", "ne"}:
        if not isinstance(rhs, (int, str)):
            msg = f"Predicate op={pred.op!r} expects `value` to be an integer or string."
            raise TypeError(msg)
        return False if v is None else bool(op(v, rhs))

    if pred.op in {"lt", "le", "gt", "ge"}:
        if not isinstance(rhs, int):
            msg = f"Predicate op={pred.op!r} expects `value` to be an integer."
            raise TypeError(msg)
        return False if not isinstance(v, int) else bool(op(v, rhs))

    if isinstance(rhs, list):
        msg = f"Predicate op={pred.op!r} does not accept list `value`."
        raise TypeError(msg)
    return bool(op(v, rhs))


def _render_flags(flags: dict[str, SetValue]) -> list[str]:
    """Turn a {flag: value} map into sbatch CLI args."""
    argv: list[str] = []
    local_flags = dict(flags)

    # Stable output for reproducibility.
    for flag in sorted(local_flags.keys()):
        val = local_flags[flag]

        # None => omit entirely
        if val is None:
            continue

        # bool => include only if True
        if isinstance(val, bool):
            if val:
                argv.append(f"--{flag}")
            continue

        # list[str] => repeat flag
        if isinstance(val, list):
            argv.extend(f"--{flag}={item}" for item in val)
            continue

        argv.append(f"--{flag}={val}")

    return argv


def _query_slurm_state(job_id: str) -> str | None:
    """Fetch SLURM state using ``squeue`` with ``sacct`` fallback.

    Args:
        job_id: SLURM job id.

    Returns:
        Raw SLURM state string, or ``None`` if unavailable.
    """
    squeue = subprocess.run(  # noqa: S603
        [_resolve_slurm_cmd("squeue"), "-h", "-j", job_id, "-o", "%T"], check=False, capture_output=True, text=True
    )
    if squeue.returncode == 0:
        output = squeue.stdout.strip()
        if output:
            return output.splitlines()[0].strip()

    sacct = subprocess.run(  # noqa: S603
        [_resolve_slurm_cmd("sacct"), "-n", "-j", job_id, "--format=State"], check=False, capture_output=True, text=True
    )
    if sacct.returncode == 0:
        output = sacct.stdout.strip()
        if output:
            return output.splitlines()[0].strip().split()[0]
    return None


@cache
def _resolve_slurm_cmd(name: str) -> str:
    """Resolve required SLURM CLI command from ``PATH``.

    Args:
        name: Command name, for example ``"sbatch"``.

    Returns:
        Absolute command path.

    Raises:
        FileNotFoundError: If command is not available.
    """
    path = shutil.which(name)
    if path is None:
        msg = f"Required command {name!r} not found on PATH. Is SLURM installed on this system?"
        raise FileNotFoundError(msg)
    return path
