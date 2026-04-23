"""SLURM-backed executor implementation."""

from __future__ import annotations

import logging
import operator
import shlex
import shutil
import subprocess
from functools import cache
from typing import TYPE_CHECKING, Literal, cast

import msgspec

from misen.executor import Executor, Job
from misen.executors.slurm.parsing import get_assigned_resources_slurm, get_assigned_resources_slurm_per_node
from misen.executors.slurm.rules import ResourceCondition, ResourceKey, ResourcePredicate, SetValue, SlurmRule
from misen.utils.runtime_events import runtime_event, task_label, work_unit_label
from misen.utils.snapshot import LocalSnapshot

if TYPE_CHECKING:
    from pathlib import Path

    from misen.task_metadata import Resources
    from misen.utils.work_unit import WorkUnit
    from misen.workspace import Workspace

_SLURM_STATE_MAP: dict[str, Literal["pending", "running", "failed", "done"]] = {
    **dict.fromkeys(
        ("PENDING", "CONFIGURING", "SUSPENDED", "REQUEUED", "REQUEUED_HOLD", "STAGE_OUT"),
        "pending",
    ),
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
logger = logging.getLogger(__name__)


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

        normalized_state = state.strip().upper().split("+", maxsplit=1)[0].split(":", maxsplit=1)[0]
        return _SLURM_STATE_MAP.get(normalized_state, "unknown")


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

    def _make_snapshot(self, workspace: Workspace) -> LocalSnapshot:
        """Return cached ``LocalSnapshot`` rooted at workspace snapshots dir."""
        return self._make_local_snapshot(workspace=workspace)

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
        logger.info(
            "Submitting SLURM work unit %s with %d dependency job(s).",
            work_unit_label(work_unit),
            len(dependencies),
        )

        sbatch_cmd: list[str] = [
            _resolve_slurm_cmd("sbatch"),
            "--parsable",
            "--job-name",
            f"misen-{work_unit_repr}",
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
        sbatch_cmd.extend(
            _resolve_dynamic_sbatch_flags(
                resources=resources,
                default_flags={
                    **self.default_flags,
                    **{
                        key: value
                        for key, value in (
                            ("partition", self.partition),
                            ("account", self.account),
                            ("qos", self.qos),
                            ("constraint", self.constraint),
                        )
                        if value is not None
                    },
                },
                rules=self.rules,
            )
        )

        if dependencies:
            dep_ids = ":".join(job.slurm_job_id for job in dependencies)
            sbatch_cmd.extend(["--dependency", f"afterok:{dep_ids}"])

        job_id, argv, env_overrides = snapshot.prepare_job(
            work_unit=work_unit,
            workspace=workspace,
            assigned_resources_getter=(
                get_assigned_resources_slurm_per_node if resources["nodes"] > 1 else get_assigned_resources_slurm
            ),
            gpu_runtime=resources["gpu_runtime"],
        )

        job_log_path = workspace.get_job_log(job_id=job_id, work_unit=work_unit)
        sbatch_cmd.extend(["--output", str(job_log_path)])

        sbatch_cmd.extend(["--export", "ALL"])
        env_prefix = ["env", *(f"{key}={value}" for key, value in env_overrides.items())]
        sbatch_cmd.extend(["--wrap", shlex.join([*env_prefix, *argv])])
        logger.debug("sbatch command for %s: %s", work_unit_label(work_unit), shlex.join(sbatch_cmd))

        try:
            result = subprocess.run(sbatch_cmd, check=True, capture_output=True, text=True)  # noqa: S603
        except subprocess.CalledProcessError as e:
            logger.exception("sbatch failed while submitting %s.", work_unit_label(work_unit))
            msg = f"sbatch failed: {(e.stderr or e.stdout or '').strip()}"
            raise RuntimeError(msg) from e

        out = result.stdout.strip()
        slurm_job_id = out.split(";", 1)[0].split(None, 1)[0]
        if not slurm_job_id.isdigit():
            logger.error("Unexpected sbatch output for %s: %s", work_unit_label(work_unit), out)
            msg = f"Unexpected sbatch output: {out!r}"
            raise RuntimeError(msg)
        logger.info(
            "Submitted SLURM work unit %s (job_id=%s, slurm_job_id=%s).",
            work_unit_label(work_unit),
            job_id,
            slurm_job_id,
        )
        runtime_event(
            f"SLURM job submitted: {task_label(work_unit.root, include_hash=False, include_arguments=True)}",
            style="green",
        )
        return SlurmJob(work_unit=work_unit, job_id=job_id, slurm_job_id=slurm_job_id, log_path=job_log_path)


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
      - `when: {"gpu_runtime": "rocm"}` means gpu_runtime == "rocm"
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

    gpu_type = resolved.pop("gpu-type", None)
    if resources["gpus"] > 0:
        resolved["gpus-per-node"] = (
            f"{gpu_type}:{resources['gpus']}" if gpu_type else resources["gpus"]
        )

    return _render_flags(resolved)


def _rule_matches(resources: Resources, when: dict[ResourceKey, ResourceCondition]) -> bool:
    for key, cond in when.items():
        value = resources[key]
        if not _match_condition(value, cond):
            return False
    return True


def _match_condition(value: int | str | None, cond: ResourceCondition) -> bool:
    if isinstance(cond, list):
        cond = cast("list[ResourcePredicate]", cond)
        return all(_match_predicate(value, pred) for pred in cond)
    if isinstance(cond, ResourcePredicate):
        return _match_predicate(value, cond)
    if cond is None:
        return operator.is_(value, None)
    return operator.eq(value, cond)


def _match_predicate(value: int | str | None, pred: ResourcePredicate) -> bool:
    op = getattr(operator, pred.op)
    rhs = pred.value

    if pred.op == "contains":
        if not isinstance(rhs, list):
            msg = "Predicate op='contains' expects `value` to be a list."
            raise TypeError(msg)
        return False if value is None else bool(op(rhs, value))

    if pred.op in {"eq", "ne"}:
        if not isinstance(rhs, (int, str)):
            msg = f"Predicate op={pred.op!r} expects `value` to be an integer or string."
            raise TypeError(msg)
        return False if value is None else bool(op(value, rhs))

    if pred.op in {"lt", "le", "gt", "ge"}:
        if not isinstance(rhs, int):
            msg = f"Predicate op={pred.op!r} expects `value` to be an integer."
            raise TypeError(msg)
        return False if not isinstance(value, int) else bool(op(value, rhs))

    if isinstance(rhs, list):
        msg = f"Predicate op={pred.op!r} does not accept list `value`."
        raise TypeError(msg)
    return bool(op(value, rhs))


def _render_flags(flags: dict[str, SetValue]) -> list[str]:
    """Turn a {flag: value} map into sbatch CLI args."""
    argv: list[str] = []

    # Stable output for reproducibility.
    for flag in sorted(flags.keys()):
        val = flags[flag]

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
