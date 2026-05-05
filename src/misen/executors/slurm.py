"""SLURM-backed executor implementation."""

from __future__ import annotations

import logging
import operator
import shlex
import shutil
import subprocess
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

import msgspec

from misen.executor import Executor, Job, JobState
from misen.utils.runtime_events import work_unit_label
from misen.utils.snapshot import LocalSnapshot, NullSnapshot

if TYPE_CHECKING:
    from collections.abc import Sequence

    from misen.utils.work_unit import WorkUnit
    from misen.workspace import Workspace

__all__ = ("SlurmExecutor", "SlurmJob")

logger = logging.getLogger(__name__)


class SlurmJob(Job):
    """Job handle backed by a SLURM job id.

    On the first observation of a terminal state, the job calls
    :meth:`Workspace.finalize_job_log` to capture anything written to
    the ``--output`` file *after* the worker's streaming context closed
    (most importantly, the SLURM epilogue: exit status, resource
    accounting, OOM messages, etc.).
    """

    __slots__ = ("_finalized", "slurm_job_id", "workspace")

    def __init__(
        self,
        work_unit: WorkUnit,
        job_id: str,
        slurm_job_id: str,
        log_path: Path,
        workspace: Workspace,
    ) -> None:
        """Initialize SLURM job wrapper."""
        super().__init__(work_unit=work_unit, job_id=job_id, log_path=log_path)
        self.slurm_job_id = slurm_job_id
        self.workspace = workspace
        self._finalized = False

    def state(self) -> _State:
        """Return the current SLURM state, normalized to a misen job state."""
        return type(self).bulk_state([self]).get(self, "unknown")

    @classmethod
    def bulk_state(cls, jobs: Sequence[Job]) -> dict[Job, JobState]:
        """Return states for many SLURM jobs using one ``squeue`` + one ``sacct`` call.

        ``squeue`` is queried first because it answers from the controller's
        in-memory state (fast). Anything not returned by ``squeue`` is looked
        up in ``sacct`` (slower, hits SlurmDBD), which covers jobs that have
        already left the controller's queue. A single batched call replaces
        the N per-job invocations the default :meth:`Job.bulk_state` would
        make, which matters when the TUI is polling many jobs at once.

        ``jobs`` must be ``SlurmJob`` instances; :func:`bulk_job_states`
        partitions a heterogeneous list by class before dispatching here.
        """
        if not jobs:
            return {}
        slurm_jobs = cast("Sequence[SlurmJob]", jobs)

        # Group by slurm id so duplicate handles to the same job all get the
        # same state (rare but tolerated).
        by_id: dict[str, list[SlurmJob]] = {}
        for job in slurm_jobs:
            by_id.setdefault(job.slurm_job_id, []).append(job)

        states: dict[str, JobState] = dict.fromkeys(by_id, "unknown")
        remaining = set(by_id)

        squeue_out = _run_slurm_query(
            "squeue", ["-h", "-j", ",".join(sorted(remaining)), "-o", "%i %T"]
        )
        for sid, raw in _parse_id_state_rows(squeue_out):
            if sid in by_id:
                states[sid] = _normalize_slurm_state(raw)
                remaining.discard(sid)

        if remaining:
            sacct_out = _run_slurm_query(
                "sacct",
                ["-n", "-X", "-j", ",".join(sorted(remaining)), "--format=JobIDRaw,State"],
            )
            for sid, raw in _parse_id_state_rows(sacct_out):
                if sid in by_id:
                    states[sid] = _normalize_slurm_state(raw)
                    remaining.discard(sid)

        result: dict[Job, JobState] = {}
        for sid, group in by_id.items():
            state = states[sid]
            for job in group:
                if not job._finalized and state in {"done", "failed"} and job.log_path is not None:  # noqa: SLF001
                    job.workspace.finalize_job_log(job.log_path)
                    job._finalized = True  # noqa: SLF001
                result[job] = state
        return result


class SlurmExecutor(Executor[SlurmJob, "LocalSnapshot | NullSnapshot"]):
    """Executor that submits work units to SLURM via ``sbatch``."""

    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    constraint: str | None = None
    default_flags: dict[str, _SetValue] = msgspec.field(default_factory=dict)
    rules: list[_SlurmRule] = msgspec.field(default_factory=list)
    snapshot: bool = True
    snapshots_dir: str | None = None

    def __post_init__(self) -> None:
        """Normalize untyped config into msgspec structs."""
        self.default_flags = msgspec.convert(self.default_flags, type=dict[str, _SetValue])
        self.rules = msgspec.convert(self.rules, type=list[_SlurmRule])

    def _make_snapshot(self, workspace: Workspace) -> LocalSnapshot | NullSnapshot:
        """Return a local snapshot for this workspace, or ``NullSnapshot`` when disabled."""
        if not self.snapshot:
            return NullSnapshot()
        snapshots_dir = Path(self.snapshots_dir) if self.snapshots_dir is not None else None
        return self._make_local_snapshot(workspace=workspace, snapshots_dir=snapshots_dir)

    def _dispatch(
        self,
        work_unit: WorkUnit,
        dependencies: set[SlurmJob],
        workspace: Workspace,
        snapshot: LocalSnapshot | NullSnapshot,
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
            "--nodes",
            "1",
            "--ntasks-per-node",
            "1",
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

        # SLURM cgroups already mask GPUs and pin CPU affinity for the job
        # step, so the worker leaves the inherited environment alone — user
        # code reads ``CUDA_VISIBLE_DEVICES`` / ``os.sched_getaffinity`` to
        # discover its allotment.
        job_id, argv, env_overrides, log_path = snapshot.prepare_job(
            work_unit=work_unit,
            workspace=workspace,
            gpu_runtime=resources["gpu_runtime"],
            cpu_indices=None,
            gpu_indices=None,
        )

        # ``argv`` already carries ``--job-log-path`` so the worker can
        # wrap its lifecycle in ``workspace.streaming_job_log(...)``;
        # ``--output`` points SLURM's stdout capture at the same file.
        wrapped = [
            "env",
            *(f"{key}={value}" for key, value in env_overrides.items()),
            *argv,
        ]
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
        return SlurmJob(
            work_unit=work_unit,
            job_id=job_id,
            slurm_job_id=slurm_job_id,
            log_path=log_path,
            workspace=workspace,
        )


_State = Literal["pending", "running", "done", "failed", "unknown"]
_ResourceKey: TypeAlias = Literal["time", "memory", "cpus", "gpus", "gpu_memory", "gpu_runtime"]
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


def _run_slurm_query(command: str, args: list[str]) -> str:
    """Invoke a SLURM CLI tool and return stdout, or ``""`` if the call failed."""
    try:
        binary = _resolve_slurm_cmd(command)
    except FileNotFoundError:
        return ""
    try:
        result = subprocess.run(  # noqa: S603
            [binary, *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout


def _parse_id_state_rows(output: str) -> list[tuple[str, str]]:
    """Parse ``<JobID> <State>`` rows from squeue/sacct output."""
    rows: list[tuple[str, str]] = []
    for line in output.splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) < 2:  # noqa: PLR2004
            continue
        rows.append((parts[0], parts[1]))
    return rows


def _normalize_slurm_state(raw: str) -> JobState:
    """Strip SLURM annotations like ``"CANCELLED+"`` / ``"CANCELLED by 1"`` and map to misen state."""
    head = raw.upper().split("+", maxsplit=1)[0].split(":", maxsplit=1)[0].split(None, 1)[0]
    return _SLURM_STATE_MAP.get(head, "unknown")
