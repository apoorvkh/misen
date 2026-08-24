"""SLURM-backed executor implementation."""

from __future__ import annotations

import logging
import operator
import shlex
import shutil
import subprocess
from functools import cache
from typing import TYPE_CHECKING, ClassVar, Literal, TypeAlias, cast

import msgspec

from misen.exceptions import StatusQueryError, SubmissionError
from misen.executor import Executor, Job, JobState
from misen.utils.dask_runtime import DEFAULT_DASK_STARTUP_TIMEOUT, managed_cluster_script
from misen.utils.runtime_events import work_unit_label
from misen.utils.snapshot import prepare_live_job

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from misen.utils.snapshot import ProjectSnapshot
    from misen.utils.work_unit import WorkUnit
    from misen.workspace import Workspace

__all__ = ("SlurmExecutor", "SlurmJob")

logger = logging.getLogger(__name__)

_SLURM_QUERY_TIMEOUT_S = 15.0
_SLURM_SUBMIT_TIMEOUT_S = 30.0


class SlurmJob(Job):
    """Job handle backed by a SLURM job id.

    On the first observation of a terminal state, the job calls
    :meth:`Workspace.finalize_job_log` to capture anything written to
    the ``--output`` file *after* the worker's streaming context closed
    (most importantly, the SLURM epilogue: exit status, resource
    accounting, OOM messages, etc.).
    """

    __slots__ = ("slurm_job_id", "workspace")

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

    def state(self) -> JobState:
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

        def collect(command: str, args: list[str], *, in_queue: bool) -> None:
            for sid, raw in _parse_id_state_rows(_run_slurm_query(command, args)):
                if sid in by_id:
                    state = _normalize_slurm_state(raw, in_queue=in_queue)
                    states[sid] = state
                    if state == "failed":
                        for job in by_id[sid]:
                            job._record_failure(f"SLURM reported state {raw!r}.")  # noqa: SLF001
                    remaining.discard(sid)

        query_errors: list[StatusQueryError] = []
        for command, in_queue in (("squeue", True), ("sacct", False)):
            if not remaining:
                break
            ids = ",".join(sorted(remaining))
            args = ["-h", "-j", ids, "-o", "%i %T"] if in_queue else ["-n", "-X", "-j", ids, "--format=JobIDRaw,State"]
            try:
                collect(command, args, in_queue=in_queue)
            except StatusQueryError as exc:
                query_errors.append(exc)

        deferred_error: StatusQueryError | None = None
        if remaining and query_errors:
            msg = (
                f"Could not resolve SLURM state for job ids {', '.join(sorted(remaining))}: "
                f"{'; '.join(map(str, query_errors))}"
            )
            error = StatusQueryError(msg, retryable=any(query_error.retryable for query_error in query_errors))
            for query_error in query_errors[:-1]:
                error.add_note(str(query_error))
            if error.retryable:
                first_failure = all(job._unknown_since is None for sid in remaining for job in by_id[sid])  # noqa: SLF001
                logger.log(
                    logging.WARNING if first_failure else logging.DEBUG,
                    "%s; preserving resolved states and retrying unresolved jobs.",
                    error,
                )
            else:
                deferred_error = error

        result: dict[Job, JobState] = {}
        for sid, group in by_id.items():
            state = states[sid]
            for job in group:
                if state in {"done", "failed"}:
                    job._finalize_log(job.workspace, failed=state == "failed")  # noqa: SLF001
                result[job] = state
        if deferred_error is not None:
            cause = query_errors[-1].__cause__ or query_errors[-1]
            raise deferred_error from cause
        return result


class SlurmExecutor(Executor[SlurmJob]):
    """Executor that submits work units to SLURM via ``sbatch``.

    Snapshots are content-addressed project state published to the
    workspace. By default each SLURM job materializes (or reuses) its
    environments in an env store on its own compute node's local disk
    (``env_store_dir``, default ``/tmp/misen-env-store-<user>``); the first
    job per node pays the build, later jobs on that node share it. With
    ``prewarm_envs=True`` and ``env_store_dir`` on a *shared* filesystem,
    environments are instead built once at submission and jobs dispatch
    with direct activation — no worker-side build, no network needed on
    compute nodes.
    """

    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    constraint: str | None = None
    dask_startup_timeout: int = DEFAULT_DASK_STARTUP_TIMEOUT
    default_flags: dict[str, _SetValue] = msgspec.field(default_factory=dict)
    rules: list[_SlurmRule] = msgspec.field(default_factory=list)
    _config_validation_errors: ClassVar[tuple[type[Exception], ...]] = (ValueError,)

    def __post_init__(self) -> None:
        """Normalize config and validate submit/worker filesystem topology."""
        self.default_flags = msgspec.convert(self.default_flags, type=dict[str, _SetValue])
        self.rules = msgspec.convert(self.rules, type=list[_SlurmRule])
        configured_flags = set(self.default_flags)
        for rule in self.rules:
            configured_flags.update(rule.set)
            for condition in rule.when.values():
                predicates = condition if isinstance(condition, list) else [condition]
                for predicate in predicates:
                    if isinstance(predicate, _ResourcePredicate):
                        _validate_predicate(predicate)
        if reserved := sorted(_EXECUTOR_OWNED_SBATCH_FLAGS & configured_flags):
            msg = f"Slurm flags {reserved} are controlled by SlurmExecutor and cannot be overridden."
            raise ValueError(msg)
        if (
            isinstance(self.dask_startup_timeout, bool)
            or not isinstance(self.dask_startup_timeout, int)
            or self.dask_startup_timeout < 1
        ):
            msg = "dask_startup_timeout must be a positive integer number of seconds."
            raise ValueError(msg)
        if self.snapshot and self.prewarm_envs and self.env_store_dir is None:
            msg = (
                "prewarm_envs on SlurmExecutor requires env_store_dir on a shared "
                "filesystem (the default env store is node-local, so envs prewarmed "
                "on the submit host would be invisible to compute nodes)."
            )
            raise ValueError(msg)

    def _dispatch(
        self,
        work_unit: WorkUnit,
        dependencies: set[SlurmJob],
        workspace: Workspace,
        snapshot: ProjectSnapshot | None,
    ) -> SlurmJob:
        """Submit one work unit to SLURM."""
        resources = work_unit.resources
        label = work_unit_label(work_unit)
        logger.info("Submitting SLURM work unit %s with %d dependency job(s).", label, len(dependencies))
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
        matched_resource_keys: set[str] = set()
        for rule in self.rules:
            if all(_condition_matches(resources[key], condition) for key, condition in rule.when.items()):
                matched_resource_keys.update(rule.when)
                flags.update(rule.set)

        if reserved := sorted(_EXECUTOR_OWNED_SBATCH_FLAGS & flags.keys()):
            msg = f"Slurm flags {reserved} are controlled by SlurmExecutor and cannot be overridden."
            raise SubmissionError(msg)

        gpu_type = flags.pop("gpu-type", None)
        if resources["accelerators"] > 0:
            accelerator_type = resources["accelerator_type"]
            if accelerator_type not in {"cuda", "rocm", "xpu"}:
                msg = f"SlurmExecutor does not support {accelerator_type!r} accelerators."
                raise SubmissionError(msg)
            required_keys = {"accelerator_memory"} if resources["accelerator_memory"] is not None else set()
            if accelerator_type != "cuda":
                required_keys.add("accelerator_type")
            if missing := required_keys - matched_resource_keys:
                msg = f"SlurmExecutor rules do not cover {sorted(missing)} for resources {resources!r}."
                raise SubmissionError(msg)
            count = resources["accelerators"]
            flags["gpus-per-node"] = f"{gpu_type}:{count}" if gpu_type else count

        try:
            sbatch_bin = _resolve_slurm_cmd("sbatch")
        except OSError as exc:
            msg = f"Could not submit {label}: {exc}"
            raise SubmissionError(msg) from exc

        sbatch_cmd = [
            sbatch_bin,
            "--parsable",
            "--job-name",
            f"misen-{work_unit.root.task_hash().short_b32()}",
            "--nodes",
            str(resources["nodes"]),
            "--ntasks-per-node",
            "1",
            "--cpus-per-task",
            str(resources["cpus"]),
            "--mem",
            f"{resources['memory']}G",
            "--time",
            str(resources["time"]),
        ]
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
            sbatch_cmd.extend(
                [
                    "--dependency",
                    f"afterok:{':'.join(job.slurm_job_id for job in dependencies)}",
                    "--kill-on-invalid-dep=yes",
                ]
            )

        # SLURM cgroups already mask GPUs and pin CPU affinity for the job
        # step, so the worker leaves the inherited environment alone — user
        # code reads ``CUDA_VISIBLE_DEVICES`` / ``os.sched_getaffinity`` to
        # discover its allotment.
        prepare = snapshot.prepare_job if snapshot is not None else prepare_live_job
        job_id, argv, env_overrides, log_path = prepare(
            work_unit=work_unit,
            workspace=workspace,
            cpu_indices=None,
            accelerator_type=resources["accelerator_type"],
            accelerator_indices=None,
        )

        # ``argv`` already carries ``--job-log-path`` so the worker can
        # wrap its lifecycle in ``workspace.streaming_job_log(...)``;
        # ``--output`` points SLURM's stdout capture at the same file.
        wrapped = ["env", *(f"{key}={value}" for key, value in env_overrides.items()), *argv]
        debug_argv = [*argv]
        if debug_argv[:2] == ["bash", "-c"] and len(debug_argv) > 2:  # noqa: PLR2004
            debug_argv[2] = "<redacted bootstrap script>"
        debug_wrapped = ["env", *(f"{key}={value}" for key, value in env_overrides.items()), *debug_argv]
        if work_unit.uses_dask_client:
            wrapped = [
                "bash",
                "-c",
                managed_cluster_script(
                    argv,
                    [
                        "srun",
                        f"--nodes={resources['nodes']}",
                        f"--ntasks={resources['nodes']}",
                        "--ntasks-per-node=1",
                        f"--cpus-per-task={resources['cpus']}",
                        "--overlap",
                        "--kill-on-bad-exit=1",
                    ],
                    environment=env_overrides,
                    workers=resources["nodes"],
                    cpus=resources["cpus"],
                    memory_gib=resources["memory"],
                    startup_timeout=self.dask_startup_timeout,
                ),
            ]
            debug_wrapped = ["bash", "-c", "<redacted Dask cluster script>"]
        sbatch_cmd.extend(["--output", str(log_path), "--export", "ALL", "--wrap", shlex.join(wrapped)])
        logger.debug("sbatch command for %s: %s", label, shlex.join([*sbatch_cmd[:-1], shlex.join(debug_wrapped)]))

        try:
            result = subprocess.run(  # noqa: S603
                sbatch_cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=_SLURM_SUBMIT_TIMEOUT_S,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.exception("sbatch failed while submitting %s.", label)
            if isinstance(exc, subprocess.TimeoutExpired):
                msg = f"sbatch timed out after {_SLURM_SUBMIT_TIMEOUT_S:g}s while submitting {label}"
            else:
                detail = (
                    (exc.stderr or exc.stdout or "").strip() if isinstance(exc, subprocess.CalledProcessError) else exc
                )
                msg = f"sbatch failed while submitting {label}{f': {detail}' if detail else ''}"
            raise SubmissionError(msg) from exc

        output = result.stdout.strip()
        slurm_job_id = output.split(";", 1)[0].split(None, 1)[0] if output else ""
        if not slurm_job_id.isdigit():
            msg = f"Unexpected sbatch output: {output!r}"
            logger.error("%s", msg)
            raise SubmissionError(msg)

        logger.info("Submitted SLURM work unit %s (job_id=%s, slurm_job_id=%s).", label, job_id, slurm_job_id)
        return SlurmJob(
            work_unit=work_unit,
            job_id=job_id,
            slurm_job_id=slurm_job_id,
            log_path=log_path,
            workspace=workspace,
        )


_ResourceKey: TypeAlias = Literal[
    "time",
    "memory",
    "cpus",
    "nodes",
    "accelerators",
    "accelerator_type",
    "accelerator_memory",
]
_OperatorName: TypeAlias = Literal["eq", "ne", "lt", "le", "gt", "ge", "contains", "is_", "is_not"]
_SetValue: TypeAlias = str | int | float | bool | list[str] | None
_EXECUTOR_OWNED_SBATCH_FLAGS = frozenset(
    """
    cpus-per-gpu cpus-per-task cpus-per-tres dependency export gpus gpus-per-node
    gpus-per-socket gpus-per-task gres job-name kill-on-invalid-dep mem mem-per-cpu
    mem-per-gpu mem-per-tres nodes ntasks ntasks-per-gpu ntasks-per-node output
    parsable time tres-per-job tres-per-node tres-per-socket tres-per-task wrap
    """.split()  # noqa: SIM905
)


class _ResourcePredicate(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    """One predicate against a resource value."""

    op: _OperatorName
    value: int | str | list[int | str] | None = None


_ResourceCondition: TypeAlias = int | str | _ResourcePredicate | list[_ResourcePredicate] | None


class _SlurmRule(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    """One conditional sbatch-flag override rule."""

    when: dict[_ResourceKey, _ResourceCondition] = msgspec.field(default_factory=dict)
    set: dict[str, _SetValue] = msgspec.field(default_factory=dict)


_SLURM_STATE_MAP: dict[str, JobState] = {
    **dict.fromkeys(
        "PENDING CONFIGURING SUSPENDED REQUEUED REQUEUE_HOLD REQUEUE_FED RESV_DEL_HOLD "  # noqa: SIM905
        "EXPEDITING POWER_UP_NODE REVOKED STAGE_OUT".split(),
        "pending",
    ),
    **dict.fromkeys(("RUNNING", "COMPLETING", "RESIZING", "SIGNALING", "STOPPED", "UPDATE_DB"), "running"),
    # ``PREEMPTED`` here is the terminal (out-of-queue) reading; a preempted job
    # still tracked by the controller is requeue-bound and mapped to ``pending``
    # in ``_normalize_slurm_state`` via its ``in_queue`` flag.
    **dict.fromkeys(
        "BOOT_FAIL CANCELLED DEADLINE FAILED LAUNCH_FAILED NODE_FAIL OUT_OF_MEMORY PREEMPTED "  # noqa: SIM905
        "RECONFIG_FAIL TIMEOUT TIMEOUT_SIGNAL SPECIAL_EXIT".split(),
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
    _validate_predicate(predicate)
    op = getattr(operator, predicate.op)
    rhs = predicate.value

    if predicate.op == "contains":
        rhs = cast("list[int | str]", rhs)
        return value is not None and bool(op(rhs, value))

    if predicate.op in {"eq", "ne"}:
        rhs = cast("int | str", rhs)
        return value is not None and bool(op(value, rhs))

    if predicate.op in {"lt", "le", "gt", "ge"}:
        rhs = cast("int", rhs)
        return isinstance(value, int) and bool(op(value, rhs))

    return bool(op(value, rhs))


def _validate_predicate(predicate: _ResourcePredicate) -> None:
    """Validate that a rule operator and its right-hand value agree."""
    rhs = predicate.value
    if predicate.op == "contains" and not isinstance(rhs, list):
        msg = "Predicate op='contains' expects `value` to be a list."
        raise ValueError(msg)
    if predicate.op in {"eq", "ne"} and not isinstance(rhs, (int, str)):
        msg = f"Predicate op={predicate.op!r} expects `value` to be an integer or string."
        raise ValueError(msg)
    if predicate.op in {"lt", "le", "gt", "ge"} and not isinstance(rhs, int):
        msg = f"Predicate op={predicate.op!r} expects `value` to be an integer."
        raise ValueError(msg)
    if predicate.op in {"is_", "is_not"} and isinstance(rhs, list):
        msg = f"Predicate op={predicate.op!r} does not accept list `value`."
        raise ValueError(msg)


@cache
def _resolve_slurm_cmd(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        msg = f"Required command {name!r} not found on PATH. Is SLURM installed on this system?"
        raise FileNotFoundError(msg)
    return path


def _run_slurm_query(command: str, args: list[str]) -> str:
    """Invoke a SLURM CLI query or raise a stable status error."""
    try:
        command_path = _resolve_slurm_cmd(command)
    except FileNotFoundError as exc:
        msg = f"Could not run {command}: {exc}"
        raise StatusQueryError(msg, retryable=False) from exc
    try:
        result = subprocess.run(  # noqa: S603
            [command_path, *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=_SLURM_QUERY_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        msg = f"Could not run {command}: {exc}"
        raise StatusQueryError(msg) from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        msg = f"{command} exited with code {result.returncode}"
        if detail:
            msg += f": {detail}"
        raise StatusQueryError(msg)
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


def _normalize_slurm_state(raw: str, *, in_queue: bool = False) -> JobState:
    """Strip SLURM annotations like ``"CANCELLED+"`` / ``"CANCELLED by 1"`` and map to misen state.

    ``in_queue`` marks states read from ``squeue`` -- i.e. the controller still
    tracks the job. This matters for ``PREEMPTED``: on a ``PreemptMode=REQUEUE``
    cluster a preempted job transiently reports ``PREEMPTED`` (including during
    any preemption ``GraceTime``) and is then requeued under the *same* job id,
    so a ``PREEMPTED`` job still in the queue is non-terminal and reported as
    ``pending``. A ``PREEMPTED`` job that has left the queue (answered only by
    ``sacct``) was not requeued and is a genuine failure -- the map default.
    """
    head = raw.upper().split("+", maxsplit=1)[0].split(":", maxsplit=1)[0].split(None, 1)[0]
    if in_queue and head in {"PREEMPTED", "SPECIAL_EXIT"}:
        return "pending"
    return _SLURM_STATE_MAP.get(head, "unknown")
