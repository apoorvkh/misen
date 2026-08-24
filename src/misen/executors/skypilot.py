"""SkyPilot managed-job executor.

The executor treats SkyPilot as a control plane only. Misen's workspace is
still the data plane: a remote-capable workspace transports the immutable
project snapshot, job payloads, results, locks, and logs, while SkyPilot
provisions compute and owns each durable job lifecycle. Misen submits one
managed job per work unit and enforces dependencies through durable workspace
markers. Independent jobs provision and execute concurrently.
"""

from __future__ import annotations

import importlib
import logging
import math
import re
import shlex
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, cast

import msgspec

from misen.exceptions import ConfigError, MisenError, StatusQueryError, SubmissionError
from misen.executor import Executor, Job, JobState
from misen.task_metadata import AcceleratorType
from misen.utils.dask_runtime import (
    DEFAULT_DASK_SCHEDULER_PORT,
    DEFAULT_DASK_STARTUP_TIMEOUT,
    MAX_DASK_SCHEDULER_PORT,
    MIN_DASK_SCHEDULER_PORT,
    managed_ranked_cluster_script,
)
from misen.utils.job_dependencies import dependency_state_name
from misen.utils.runtime_events import work_unit_label

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from misen.utils.graph import DependencyGraph
    from misen.utils.snapshot import ProjectSnapshot
    from misen.utils.work_unit import WorkUnit
    from misen.workspace import Workspace

__all__ = ("SkyPilotExecutor", "SkyPilotJob")

logger = logging.getLogger(__name__)

_SKYPILOT_INSTALL = 'uv pip install "misen[skypilot]"'
_SKYPILOT_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_QUEUE_FIELDS = ("job_id", "task_id", "status", "failure_reason")
_SKYPILOT_NODE_RANK_ENV = "SKYPILOT_NODE_RANK"
_SKYPILOT_NODE_IPS_ENV = "SKYPILOT_NODE_IPS"
_DONE_MARKER = b"done"

_SKYPILOT_STATE_MAP: dict[str, JobState] = {
    **dict.fromkeys(("PENDING", "SUBMITTED", "STARTING"), "pending"),
    **dict.fromkeys(("RUNNING", "WINDING_DOWN", "RECOVERING", "CANCELLING"), "running"),
    "SUCCEEDED": "done",
    **dict.fromkeys(
        (
            "CANCELLED",
            "FAILED",
            "FAILED_SETUP",
            "FAILED_PRECHECKS",
            "FAILED_NO_RESOURCE",
            "FAILED_CONTROLLER",
        ),
        "failed",
    ),
}


def _load_skypilot() -> Any:
    """Load the optional SkyPilot SDK on first use."""
    try:
        return importlib.import_module("sky")
    except ModuleNotFoundError as exc:
        if exc.name != "sky":
            raise
        msg = f"SkyPilotExecutor requires SkyPilot >=0.12.1; install it with `{_SKYPILOT_INSTALL}`."
        raise ConfigError(msg) from exc


def _field(record: object, name: str, default: Any = None) -> Any:
    """Read one field from a dict or a SkyPilot response model."""
    if isinstance(record, dict):
        return cast("dict[str, Any]", record).get(name, default)
    return getattr(record, name, default)


def _status_name(value: object) -> str:
    """Normalize SkyPilot enum/string status values to an uppercase name."""
    if isinstance(value, Enum):
        value = value.value
    text = str(value or "")
    return text.rsplit(".", 1)[-1].upper()


def _normalize_skypilot_state(value: object) -> JobState:
    """Map a SkyPilot managed-job status to Misen's lifecycle."""
    status = _status_name(value)
    if status.startswith("FAILED"):
        return "failed"
    return _SKYPILOT_STATE_MAP.get(status, "unknown")


def _queue_records(result: object) -> list[object]:
    """Extract queue records from SkyPilot's queue_v2 response."""
    records = result[0] if isinstance(result, tuple) and result else None
    if not isinstance(records, (list, tuple)):
        msg = f"SkyPilot queue_v2 returned an unexpected response: {result!r}"
        raise StatusQueryError(msg, retryable=False)
    return list(records)


class SkyPilotJob(Job):
    """One Misen work unit backed by one SkyPilot managed job."""

    __slots__ = ("_terminal_state", "deadline_minutes", "managed_job_id", "submission_id", "workspace")

    def __init__(
        self,
        *,
        work_unit: WorkUnit,
        job_id: str,
        managed_job_id: int,
        submission_id: str,
        deadline_minutes: int,
        log_path: Path,
        workspace: Workspace,
    ) -> None:
        """Initialize a handle for one managed job."""
        super().__init__(work_unit=work_unit, job_id=job_id, log_path=log_path)
        self.managed_job_id = managed_job_id
        self.submission_id = submission_id
        self.deadline_minutes = deadline_minutes
        self.workspace = workspace
        self._terminal_state: JobState | None = None

    def state(self) -> JobState:
        """Return this managed job's normalized SkyPilot state."""
        return type(self).bulk_state([self]).get(self, "unknown")

    @classmethod
    def bulk_state(cls, jobs: Sequence[Job]) -> dict[Job, JobState]:
        """Query all managed jobs through one SkyPilot request."""
        if not jobs:
            return {}
        skypilot_jobs = cast("Sequence[SkyPilotJob]", jobs)
        result: dict[Job, JobState] = {
            job: job._terminal_state  # noqa: SLF001
            for job in skypilot_jobs
            if job._terminal_state is not None  # noqa: SLF001
        }
        unresolved_jobs = [job for job in skypilot_jobs if job._terminal_state is None]  # noqa: SLF001
        if not unresolved_jobs:
            return result

        managed_ids = sorted({job.managed_job_id for job in unresolved_jobs})
        sky = _load_skypilot()
        try:
            request_id = sky.jobs.queue_v2(
                # Managed-jobs controllers autostop. Refresh makes old handles
                # queryable instead of turning a stopped controller into an
                # indefinitely unknown Misen state.
                refresh=True,
                job_ids=managed_ids,
                fields=_QUEUE_FIELDS,
            )
            records = _queue_records(sky.get(request_id))
        except StatusQueryError:
            raise
        except Exception as exc:
            msg = f"Could not query SkyPilot managed jobs {managed_ids}: {exc}"
            raise StatusQueryError(msg) from exc
        by_job_id: dict[int, object] = {}
        for record in records:
            raw_job_id = _field(record, "job_id")
            raw_task_id = _field(record, "task_id")
            if isinstance(raw_job_id, int) and raw_task_id in (0, None):
                by_job_id[raw_job_id] = record

        for job in unresolved_jobs:
            record = by_job_id.get(job.managed_job_id)
            state = _normalize_skypilot_state(_field(record, "status"))
            raw_status = _status_name(_field(record, "status"))

            # A controller failure may mask a Misen result that committed
            # first. The workspace remains the authoritative success boundary.
            if state == "failed":
                committed = False
                try:
                    committed = (
                        job.workspace.read_job_file(
                            job.submission_id,
                            dependency_state_name(cast("str", job.job_id)),
                        )
                        == _DONE_MARKER
                    )
                except FileNotFoundError:
                    pass
                except (MisenError, OSError) as exc:
                    msg = f"Could not verify the completion marker for failed {job.label}: {exc}"
                    raise StatusQueryError(msg) from exc
                if not committed:
                    try:
                        committed = job.work_unit.done(workspace=job.workspace)
                    except (MisenError, OSError) as exc:
                        msg = f"Could not verify workspace completion for failed {job.label}: {exc}"
                        raise StatusQueryError(msg) from exc
                if committed:
                    state = "done"

            if state == "failed":
                raw_status = raw_status or "FAILED"
                reason = _field(record, "failure_reason")
                detail = f": {reason}" if isinstance(reason, str) and reason else ""
                job._record_failure(  # noqa: SLF001
                    f"SkyPilot managed job {job.managed_job_id} reported {raw_status}{detail}."
                )
            if state in {"done", "failed"}:
                try:
                    job.workspace.put_job_file(
                        job.submission_id,
                        dependency_state_name(cast("str", job.job_id)),
                        state.encode(),
                    )
                except (MisenError, OSError) as exc:
                    msg = f"Could not publish terminal dependency state for {job.label}: {exc}"
                    raise StatusQueryError(msg) from exc
                job._finalize_log(job.workspace, failed=state == "failed")  # noqa: SLF001
                job._terminal_state = state  # noqa: SLF001
            result[job] = state
        return result


class SkyPilotExecutor(Executor[SkyPilotJob]):
    """Run Misen work units as dependency-aware SkyPilot managed jobs.

    ``infra`` is passed through to SkyPilot and accepts any infrastructure
    understood by the installed SkyPilot version (for example ``"aws"``,
    ``"azure/eastus"``, ``"k8s/my-context"``, ``"ssh/my-pool"``, or
    ``"slurm/my-cluster"``), or an ordered list of alternatives. SkyPilot
    remains responsible for provider dependencies, credentials, configuration,
    and backend-specific feature checks. CPU and memory requests are forwarded
    as per-node minimums.

    ``job_recovery`` selects SkyPilot's infrastructure-recovery strategy.
    This adapter deliberately accepts only the strategy name, leaving
    application-error restarts disabled so task side effects are not repeated
    implicitly.

    Misen accelerator types describe programming backends (``cuda``, ``tpu``,
    etc.), whereas SkyPilot requires concrete hardware names. Configure the
    candidate mapping explicitly with ``accelerators``. If tasks declare a
    minimum per-device memory, also provide each candidate's capacity in
    ``accelerator_memory`` so the executor can filter safely.

    Arbitrary Misen DAGs are supported by submitting one managed job per work
    unit. Every job is accepted eagerly, while a worker-side workspace gate
    delays user code until its parents finish. This preserves parallelism and
    durable dependency gates for jobs that reach worker code, at the cost of
    provisioning descendants while they wait. A pre-worker failure requires
    status observation for prompt propagation; otherwise descendants fail at
    their cumulative timeout. Multi-node tasks normally execute the Misen
    worker once on rank zero. A work unit requesting ``DASK_CLIENT`` instead
    receives one private Dask worker per node, with its scheduler and Misen
    coordinator on rank zero.
    """

    infra: str | list[str] = "aws"
    instance_type: str | None = None
    accelerators: dict[AcceleratorType, list[str]] = msgspec.field(default_factory=dict)
    accelerator_memory: dict[str, int] = msgspec.field(default_factory=dict)
    use_spot: bool = False
    image_id: str | None = None
    disk_size: int | None = None
    max_hourly_cost: float | None = None
    job_recovery: str | None = None
    dask_startup_timeout: int = DEFAULT_DASK_STARTUP_TIMEOUT
    dask_scheduler_port: int = DEFAULT_DASK_SCHEDULER_PORT
    name_prefix: str = "misen"
    _config_validation_errors: ClassVar[tuple[type[Exception], ...]] = (ValueError,)

    def __post_init__(self) -> None:
        """Normalize configuration and reject modes unsafe on remote workers."""
        self.accelerators = msgspec.convert(self.accelerators, type=dict[AcceleratorType, list[str]])
        self.accelerator_memory = msgspec.convert(self.accelerator_memory, type=dict[str, int])
        infras = [self.infra] if isinstance(self.infra, str) else self.infra
        if not infras or any(not isinstance(infra, str) or not infra.strip() for infra in infras):
            msg = "infra must be a non-empty SkyPilot infrastructure string or list of strings."
            raise ValueError(msg)
        normalized_infras = [infra.strip() for infra in infras]
        if len(set(normalized_infras)) != len(normalized_infras):
            msg = "infra must not contain duplicate alternatives."
            raise ValueError(msg)
        self.infra = normalized_infras[0] if isinstance(self.infra, str) else normalized_infras
        if not self.snapshot:
            msg = "SkyPilotExecutor requires snapshot=True; live project paths are not visible on remote workers."
            raise ValueError(msg)
        if self.prewarm_envs:
            msg = "SkyPilotExecutor requires prewarm_envs=False; submit-host environments are not worker-visible."
            raise ValueError(msg)
        for field_name in ("instance_type", "image_id", "job_recovery"):
            value = getattr(self, field_name)
            if value is not None:
                if not value.strip():
                    msg = f"{field_name} must be a non-empty string when set."
                    raise ValueError(msg)
                setattr(self, field_name, value.strip())
        if not _SKYPILOT_NAME.fullmatch(self.name_prefix) or len(self.name_prefix) > 30:  # noqa: PLR2004
            msg = "name_prefix must be at most 30 lowercase letters, digits, or single hyphen-separated words."
            raise ValueError(msg)
        for accelerator_type, models in self.accelerators.items():
            if not models or any(not model.strip() for model in models):
                msg = f"accelerators[{accelerator_type!r}] must contain one or more non-empty model names."
                raise ValueError(msg)
            normalized_models = [model.strip() for model in models]
            if len(set(normalized_models)) != len(normalized_models):
                msg = f"accelerators[{accelerator_type!r}] must not contain duplicate model names."
                raise ValueError(msg)
            self.accelerators[accelerator_type] = normalized_models
        if any(not model.strip() for model in self.accelerator_memory):
            msg = "accelerator_memory keys must be non-empty SkyPilot model names."
            raise ValueError(msg)
        normalized_memory = {model.strip(): memory for model, memory in self.accelerator_memory.items()}
        if len(normalized_memory) != len(self.accelerator_memory):
            msg = "accelerator_memory must not contain duplicate model names after trimming whitespace."
            raise ValueError(msg)
        self.accelerator_memory = normalized_memory
        if any(isinstance(memory, bool) or memory < 1 for memory in self.accelerator_memory.values()):
            msg = "accelerator_memory values must be positive integer GiB capacities."
            raise ValueError(msg)
        if self.disk_size is not None and (isinstance(self.disk_size, bool) or self.disk_size < 1):
            msg = "disk_size must be a positive integer number of GiB."
            raise ValueError(msg)
        if self.max_hourly_cost is not None and (
            isinstance(self.max_hourly_cost, bool)
            or not math.isfinite(self.max_hourly_cost)
            or self.max_hourly_cost <= 0
        ):
            msg = "max_hourly_cost must be positive."
            raise ValueError(msg)
        if (
            isinstance(self.dask_startup_timeout, bool)
            or not isinstance(self.dask_startup_timeout, int)
            or self.dask_startup_timeout < 1
        ):
            msg = "dask_startup_timeout must be a positive integer number of seconds."
            raise ValueError(msg)
        if (
            isinstance(self.dask_scheduler_port, bool)
            or not isinstance(self.dask_scheduler_port, int)
            or not MIN_DASK_SCHEDULER_PORT <= self.dask_scheduler_port <= MAX_DASK_SCHEDULER_PORT
        ):
            msg = "dask_scheduler_port must be an integer between 1024 and 65535."
            raise ValueError(msg)

    def _validate_submission(
        self,
        *,
        work_graph: DependencyGraph[WorkUnit],
        pending_work_units: Sequence[WorkUnit],
        workspace: Workspace,
    ) -> None:
        """Reject local-only storage and validate every remote resource request."""
        del work_graph
        transport = workspace.bootstrap_transport()
        if transport is None:
            msg = (
                "SkyPilotExecutor requires a remotely fetchable workspace transport; "
                "use CloudWorkspace with worker IAM/service-account access."
            )
            raise ConfigError(msg)
        if workspace.get_temp_dir().is_absolute():
            msg = (
                "SkyPilotExecutor requires a relative workspace cache_dir so worker payload/log paths are valid "
                "on ephemeral hosts (for CloudWorkspace, use cache_dir='.cache/misen')."
            )
            raise ConfigError(msg)
        if not workspace.supports_job_file_reads():
            msg = "SkyPilotExecutor requires a workspace that supports submission-file coordination reads."
            raise ConfigError(msg)
        sky = _load_skypilot()
        for work_unit in pending_work_units:
            try:
                self._resource_options(sky, work_unit)
            except SubmissionError:
                raise
            except Exception as exc:
                msg = f"Invalid SkyPilot resources for {work_unit_label(work_unit)}: {exc}"
                raise SubmissionError(msg) from exc

    def _accelerator_models(self, work_unit: WorkUnit) -> Sequence[str | None]:
        """Resolve a generic accelerator request to concrete SkyPilot models."""
        requested = work_unit.resources
        if not requested["accelerators"]:
            return [None]

        accelerator_type = requested["accelerator_type"]
        models = list(self.accelerators.get(accelerator_type, ()))
        if not models:
            msg = (
                f"No SkyPilot accelerator models are configured for {accelerator_type!r}; "
                f"set executor.accelerators.{accelerator_type}."
            )
            raise SubmissionError(msg)
        if minimum_memory := requested["accelerator_memory"]:
            models = [model for model in models if self.accelerator_memory.get(model, 0) >= minimum_memory]
            if not models:
                msg = (
                    f"No configured {accelerator_type!r} SkyPilot accelerator has the requested "
                    f"minimum {minimum_memory} GiB/device; configure matching accelerator_memory capacities."
                )
                raise SubmissionError(msg)
        return models

    def _resource_options(self, sky: Any, work_unit: WorkUnit) -> object:
        """Translate one generic Misen request into SkyPilot resource choices."""
        requested = work_unit.resources
        infras = [self.infra] if isinstance(self.infra, str) else self.infra
        models = self._accelerator_models(work_unit)
        validate_locally = sky.server.common.is_api_server_local()

        common_options: dict[str, object] = {
            "cpus": f"{requested['cpus']}+",
            "memory": f"{requested['memory']}+",
            "use_spot": self.use_spot,
        }
        for key in ("instance_type", "image_id", "disk_size", "max_hourly_cost", "job_recovery"):
            if (value := getattr(self, key)) is not None:
                common_options[key] = value

        options: list[object] = []
        for infra in infras:
            for model in models:
                resource_options = {"infra": infra, **common_options}
                if model is not None:
                    resource_options["accelerators"] = {model: requested["accelerators"]}
                option = sky.Resources(**resource_options)
                # Full validation consults local Kubernetes contexts, SSH node
                # pools, and Slurm configuration. A remote API server owns
                # those settings and validates them in jobs.launch instead.
                if validate_locally:
                    option.validate()
                options.append(option)
        return options[0] if len(options) == 1 else options

    @staticmethod
    def _run_command(
        argv: list[str],
        env: dict[str, str],
        log_path: Path,
        *,
        time_minutes: int,
        nodes: int,
        cpus: int,
        memory_gib: int,
        uses_dask_client: bool,
        dask_startup_timeout: int,
        dask_scheduler_port: int,
    ) -> str:
        """Render one bounded rank-aware SkyPilot worker command."""
        command = (
            managed_ranked_cluster_script(
                argv,
                environment=env,
                workers=nodes,
                cpus=cpus,
                memory_gib=memory_gib,
                startup_timeout=dask_startup_timeout,
                node_rank_env=_SKYPILOT_NODE_RANK_ENV,
                node_ips_env=_SKYPILOT_NODE_IPS_ENV,
                scheduler_port=dask_scheduler_port,
            )
            if uses_dask_client
            else shlex.join(["env", *(f"{key}={value}" for key, value in env.items()), *argv])
        )
        lines = ["set -o pipefail"]
        if nodes > 1 and not uses_dask_client:
            lines.append('if [[ "${SKYPILOT_NODE_RANK:-0}" != "0" ]]; then exit 0; fi')
        lines.extend(
            (
                f"mkdir -p {shlex.quote(str(log_path.parent))}",
                (
                    f"timeout --signal=TERM --kill-after=30s {time_minutes}m bash -c {shlex.quote(command)} "
                    f"2>&1 | tee -a {shlex.quote(str(log_path))}"
                ),
            )
        )
        return "\n".join(lines)

    def _dispatch(
        self,
        work_unit: WorkUnit,
        dependencies: set[SkyPilotJob],
        workspace: Workspace,
        snapshot: ProjectSnapshot | None,
    ) -> SkyPilotJob:
        """Submit one managed job with durable gates for its parent jobs."""
        if snapshot is None:  # guarded by __post_init__; retain the boundary invariant
            msg = "SkyPilotExecutor cannot dispatch without a project snapshot."
            raise SubmissionError(msg)

        resources = work_unit.resources
        dependency_jobs = {dependency.work_unit: cast("str", dependency.job_id) for dependency in dependencies}
        job_id, argv, env, log_path = snapshot.prepare_job(
            work_unit=work_unit,
            workspace=workspace,
            cpu_indices=None,
            accelerator_type=resources["accelerator_type"],
            accelerator_indices=None,
            dependency_jobs=dependency_jobs,
        )
        deadline_minutes = resources["time"] + max(
            (dependency.deadline_minutes for dependency in dependencies),
            default=0,
        )
        name = f"{self.name_prefix}-{snapshot.submission_id.lower()}-{job_id.lower()}"
        sky = _load_skypilot()
        try:
            task = sky.Task(
                name=name,
                run=self._run_command(
                    argv,
                    env,
                    log_path,
                    time_minutes=deadline_minutes,
                    nodes=resources["nodes"],
                    cpus=resources["cpus"],
                    memory_gib=resources["memory"],
                    uses_dask_client=work_unit.uses_dask_client,
                    dask_startup_timeout=self.dask_startup_timeout,
                    dask_scheduler_port=self.dask_scheduler_port,
                ),
                num_nodes=resources["nodes"],
                resources=self._resource_options(sky, work_unit),
                api_server_access=False,
            )
            request_id = sky.jobs.launch(task, name=name)
            launch_result = sky.get(request_id)
        except SubmissionError:
            raise
        except Exception as exc:
            msg = f"SkyPilot failed to submit managed job {name!r}: {exc}"
            raise SubmissionError(msg) from exc

        managed_ids = launch_result[0] if isinstance(launch_result, tuple) and launch_result else None
        if not isinstance(managed_ids, (list, tuple)) or len(managed_ids) != 1 or not isinstance(managed_ids[0], int):
            msg = (
                f"SkyPilot accepted launch request {request_id!r} for job {name!r}, "
                f"but returned an unexpected result: {launch_result!r}. Use the job name to inspect it."
            )
            raise SubmissionError(msg)

        managed_job_id = managed_ids[0]
        logger.info("Submitted SkyPilot managed job %s (managed_job_id=%d).", name, managed_job_id)
        return SkyPilotJob(
            work_unit=work_unit,
            job_id=job_id,
            managed_job_id=managed_job_id,
            submission_id=snapshot.submission_id,
            deadline_minutes=deadline_minutes,
            log_path=log_path,
            workspace=workspace,
        )
