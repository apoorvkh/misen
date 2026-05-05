"""SkyPilot-backed executor implementation.

Dispatches each work unit as a SkyPilot **managed job**: ``sky.jobs.launch``
hands the job to SkyPilot's managed-jobs controller, which provisions an
ephemeral cluster, runs the worker, captures the result, and tears the
cluster down. The controller transparently retries on spot preemption or
transient infrastructure failure, so misen can treat compute as ephemeral
work over the persistent :class:`~misen.workspace.Workspace` -- the only
state that needs to survive is what the worker writes back through the
workspace.

The worker on the cluster re-opens the workspace from the cloudpickle
payload (the orchestrator's workspace instance is serialized into the
payload and rehydrated on the remote node), so all results, logs, and
intermediate state flow back through the workspace's durable storage.
The executor never needs direct filesystem access to the cluster.
"""

from __future__ import annotations

import logging
import shlex
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast

import msgspec

from misen.executor import Executor, Job, JobState
from misen.utils.assigned_resources import AssignedResources
from misen.utils.runtime_events import work_unit_label
from misen.utils.snapshot import CloudSnapshot, token_base32

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from misen.task_metadata import GpuRuntime, Resources
    from misen.utils.snapshot import Snapshot
    from misen.utils.work_unit import WorkUnit
    from misen.workspace import Workspace

__all__ = ("SkyPilotExecutor", "SkyPilotJob")

logger = logging.getLogger(__name__)

_GPU_RUNTIME_DEFAULTS: dict[str, str] = {
    "cuda": "T4",
    "rocm": "MI100",
    "xpu": "Gaudi2",
}

# Bash bootstrap that runs on the cluster before the worker. Idempotent: the
# cluster is fresh per job, but we keep it idempotent so users can opt into
# longer-lived clusters later without rewriting setup. The script overlays the
# orchestrator-staged manifest files onto whatever ``workdir`` rsynced over,
# so users with gitignored env files / lockfiles still get them.
_REMOTE_SETUP_SCRIPT = r"""
set -euo pipefail
cd "$HOME/sky_workdir"

MANIFEST_DIR="$HOME/.misen/manifest"
for f in pyproject.toml uv.lock pixi.toml pixi.lock .env .env.local; do
    if [ -f "$MANIFEST_DIR/$f" ]; then
        cp "$MANIFEST_DIR/$f" "./$f"
    fi
done

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

uv sync --no-install-workspace
uv sync --no-editable --no-cache

if [ -f pixi.toml ] && [ -f pixi.lock ]; then
    if ! command -v pixi >/dev/null 2>&1; then
        curl -fsSL https://pixi.sh/install.sh | bash
    fi
    export PATH="$HOME/.pixi/bin:$PATH"
    pixi install --frozen
fi

mkdir -p "$HOME/.misen/job_logs"
"""


class SkyPilotJob(Job):
    """Job handle backed by a SkyPilot managed-job id.

    The handle holds a Future for the ``sky.jobs.launch`` call so
    independent work units can be submitted to the managed-jobs
    controller concurrently. ``state()`` consults the future first
    (pending while not yet submitted, failed if submission itself
    raised), then ``sky.jobs.queue()`` for the actual managed-job
    state -- which already accounts for cluster provisioning,
    spot recovery, retries, and final completion.
    """

    __slots__ = ("_cached_state", "_launch_future", "_lock", "_managed_job_id", "managed_job_name", "workspace")

    def __init__(
        self,
        work_unit: WorkUnit,
        managed_job_name: str,
        workspace: Workspace,
    ) -> None:
        """Initialize a pending SkyPilot managed-job handle.

        ``managed_job_name`` is decided up front (a unique name per work
        unit) so it appears on the handle even before the launch future
        runs. The integer ``managed_job_id`` returned by SkyPilot is
        recorded once the launch future resolves.
        """
        super().__init__(work_unit=work_unit, job_id=None, log_path=None)
        self.managed_job_name = managed_job_name
        self.workspace = workspace
        self._launch_future: Future[int] | None = None
        self._managed_job_id: int | None = None
        self._cached_state: JobState = "pending"
        self._lock = threading.Lock()

    def attach_launch_future(self, future: Future[int]) -> None:
        """Attach the future returned by the launcher thread pool."""
        with self._lock:
            self._launch_future = future

    def state(self) -> JobState:
        """Return the current SkyPilot-mapped job state."""
        return type(self).bulk_state([self]).get(self, "unknown")

    @classmethod
    def bulk_state(cls, jobs: Sequence[Job]) -> dict[Job, JobState]:
        """Resolve states for many ``SkyPilotJob`` instances.

        Managed jobs share one global queue, so we can answer N jobs in
        a single ``sky.jobs.queue()`` call regardless of how many
        clusters are involved -- a meaningful win over per-cluster
        polling when a UI is watching dozens of jobs at once.

        Jobs whose launch future hasn't resolved yet (or resolved with
        an exception) are short-circuited without any SkyPilot calls.
        """
        if not jobs:
            return {}
        sky_jobs = cast("Sequence[SkyPilotJob]", jobs)
        result: dict[Job, JobState] = {}
        to_query: list[SkyPilotJob] = []
        for job in sky_jobs:
            with job._lock:  # noqa: SLF001
                if job._cached_state in {"done", "failed"}:  # noqa: SLF001
                    result[job] = job._cached_state  # noqa: SLF001
                    continue
                future = job._launch_future  # noqa: SLF001
                if future is None or not future.done():
                    result[job] = "pending"
                    continue
                exc = future.exception()
                if exc is not None:
                    job._cached_state = "failed"  # noqa: SLF001
                    result[job] = "failed"
                    continue
                if job._managed_job_id is None:  # noqa: SLF001
                    job._managed_job_id = future.result()  # noqa: SLF001
            to_query.append(job)

        if not to_query:
            return result

        try:
            state_by_id = _query_managed_job_states()
        except Exception:
            logger.exception("sky.jobs.queue failed during bulk state query.")
            for job in to_query:
                result[job] = "unknown"
            return result

        for job in to_query:
            managed_id = job._managed_job_id  # noqa: SLF001
            state: JobState = state_by_id.get(managed_id, "unknown") if managed_id is not None else "unknown"
            if state in {"done", "failed"}:
                with job._lock:  # noqa: SLF001
                    job._cached_state = state  # noqa: SLF001
            result[job] = state
        return result


class SkyPilotExecutor(Executor[SkyPilotJob, CloudSnapshot]):
    """Executor that submits each work unit as a SkyPilot managed job.

    Each managed job runs on its own ephemeral cluster, with the
    SkyPilot managed-jobs controller handling provisioning, spot
    recovery, and teardown. Misen's only responsibility is to honor
    the dependency graph (deps must reach ``done`` before a dependent
    is submitted) and to report state back to the runtime UI.
    """

    job_name_prefix: str = "misen"
    cloud: str | None = None
    region: str | None = None
    zone: str | None = None
    instance_type: str | None = None
    disk_size: int | None = None
    use_spot: bool = False
    image_id: str | None = None
    gpu_types: dict[str, str] = msgspec.field(default_factory=lambda: dict(_GPU_RUNTIME_DEFAULTS))
    setup_commands: list[str] = msgspec.field(default_factory=list)
    envs: dict[str, str] = msgspec.field(default_factory=dict)
    max_concurrent_launches: int = 4

    # Persistent cross-job caches: maps a path on the cluster to a SkyPilot
    # Storage name. Each entry becomes a ``sky.Storage(name=..., persistent=True,
    # mode=MOUNT_CACHED)`` mounted at the path, so that data written by one job
    # (model weights, wheel caches, etc.) is visible to every future job that
    # mounts the same Storage name. ``MOUNT_CACHED`` adds a local-SSD cache
    # layer over the FUSE mount, which materially helps the import-heavy
    # access patterns of HF and uv caches without losing writeback.
    # Typical entries:
    #   {"~/.cache/huggingface": "misen-hf",
    #    "~/.cache/uv": "misen-uv"}
    # The HF case lets jobs share downloaded models. The uv case lets ``uv
    # sync`` skip wheel downloads on every cluster, so setup drops from minutes
    # to seconds for cached projects -- without trying to mount the venv
    # itself, which would be fragile across heterogeneous instance types.
    # Storage names are namespaced per-user in SkyPilot's account; pick names
    # that won't collide with other projects.
    persistent_caches: dict[str, str] = msgspec.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Lazily load SkyPilot, validate config, build a launcher pool."""
        try:
            import sky
            import sky.jobs

            _ = (sky, sky.jobs)
        except ImportError as exc:
            msg = (
                "SkyPilotExecutor requires the SkyPilot package. "
                "Install it via the optional extra: `pip install 'misen[skypilot]'`."
            )
            raise ImportError(msg) from exc

        if self.max_concurrent_launches < 1:
            msg = "max_concurrent_launches must be a positive integer."
            raise ValueError(msg)

        self._launch_pool = ThreadPoolExecutor(
            max_workers=self.max_concurrent_launches,
            thread_name_prefix="misen-skypilot-launcher",
        )

    def _make_snapshot(self, workspace: Workspace) -> CloudSnapshot:
        """Build a :class:`CloudSnapshot` after verifying the workspace is remote-accessible."""
        if not workspace.supports_remote_executor:
            msg = (
                f"SkyPilotExecutor requires a workspace whose state is reachable from a remote "
                f"cluster (e.g. CloudWorkspace). {type(workspace).__name__} does not advertise "
                f"`supports_remote_executor=True`."
            )
            raise RuntimeError(msg)
        snapshots_dir = (workspace.get_temp_dir() / "snapshots").resolve()
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        return CloudSnapshot(snapshots_dir=snapshots_dir)

    def _dispatch(
        self,
        work_unit: WorkUnit,
        dependencies: set[SkyPilotJob],
        workspace: Workspace,
        snapshot: CloudSnapshot,
    ) -> SkyPilotJob:
        """Build a SkyPilot Task for the work unit and submit it to the launcher pool."""
        resources = work_unit.resources
        label = work_unit_label(work_unit)
        managed_job_name = f"{self.job_name_prefix}-{token_base32(4).lower()}"

        job = SkyPilotJob(work_unit=work_unit, managed_job_name=managed_job_name, workspace=workspace)

        # ``assigned_resources`` is fixed: the cluster is dedicated to this
        # one job, so the worker should claim everything that was requested.
        assigned = AssignedResources(
            cpu_indices=list(range(resources["cpus"])),
            gpu_indices=list(range(resources["gpus"])),
            memory=resources["memory"],
            gpu_memory=resources["gpu_memory"],
        )
        job_id, argv, _env_overrides, log_path = snapshot.prepare_job(
            work_unit=work_unit,
            workspace=workspace,
            assigned_resources_getter=partial(_constant_assigned_resources, assigned=assigned),
            gpu_runtime=resources["gpu_runtime"],
            # SkyPilot already grants the worker access to all GPUs on the
            # cluster; emitting CUDA_VISIBLE_DEVICES on top would mask them.
            bind_gpu_env=False,
        )
        job.job_id = job_id
        job.log_path = log_path

        sky_resources = self._build_sky_resources(resources)
        sky_task = self._build_sky_task(
            argv=argv,
            sky_resources=sky_resources,
            num_nodes=resources["nodes"],
            snapshot=snapshot,
            payload_filename=f"{job_id}.pkl",
        )

        future: Future[int] = self._launch_pool.submit(
            self._wait_and_launch,
            sky_task=sky_task,
            managed_job_name=managed_job_name,
            dependencies=dependencies,
            label=label,
        )
        job.attach_launch_future(future)
        logger.info(
            "Queued SkyPilot work unit %s (job_id=%s, managed_job_name=%s, deps=%d).",
            label,
            job_id,
            managed_job_name,
            len(dependencies),
        )
        return job

    def cleanup_snapshot(self, snapshot: Snapshot | None) -> None:
        """Clean up the local snapshot dir.

        The launcher pool is intentionally not torn down here — a single
        executor instance may serve multiple ``submit()`` calls in one
        Python process, and the pool is reused. Pool threads are released
        on interpreter shutdown.
        """
        super().cleanup_snapshot(snapshot)

    def _build_sky_resources(self, resources: Resources) -> object:
        """Translate misen ``Resources`` into a ``sky.Resources`` instance."""
        import sky

        kwargs: dict[str, object] = {}
        if self.cloud is not None:
            kwargs["cloud"] = sky.clouds.CLOUD_REGISTRY.from_str(self.cloud)
        if self.region is not None:
            kwargs["region"] = self.region
        if self.zone is not None:
            kwargs["zone"] = self.zone
        if self.image_id is not None:
            kwargs["image_id"] = self.image_id
        if self.use_spot:
            kwargs["use_spot"] = True
        if self.disk_size is not None:
            kwargs["disk_size"] = self.disk_size

        if self.instance_type is not None:
            kwargs["instance_type"] = self.instance_type
        else:
            if resources["cpus"] > 0:
                kwargs["cpus"] = f"{resources['cpus']}+"
            if resources["memory"] > 0:
                kwargs["memory"] = f"{resources['memory']}+"

        if resources["gpus"] > 0:
            gpu_type = self.gpu_types.get(resources["gpu_runtime"])
            if gpu_type is None:
                msg = (
                    f"No SkyPilot accelerator name configured for gpu_runtime="
                    f"{resources['gpu_runtime']!r}. Set `gpu_types` on SkyPilotExecutor."
                )
                raise ValueError(msg)
            kwargs["accelerators"] = f"{gpu_type}:{resources['gpus']}"

        return sky.Resources(**kwargs)

    def _build_sky_task(
        self,
        *,
        argv: list[str],
        sky_resources: object,
        num_nodes: int,
        snapshot: CloudSnapshot,
        payload_filename: str,
    ) -> object:
        """Construct the ``sky.Task`` for one work unit."""
        import sky

        run_command = "\n".join([
            'cd "$HOME/sky_workdir"',
            'export PATH="$HOME/.local/bin:$HOME/.pixi/bin:$PATH"',
            'export VIRTUAL_ENV="$HOME/sky_workdir/.venv"',
            shlex.join(argv),
        ])

        setup = "\n".join([_REMOTE_SETUP_SCRIPT, *self.setup_commands]) if self.setup_commands else _REMOTE_SETUP_SCRIPT

        task = sky.Task(
            setup=setup,
            run=run_command,
            envs=dict(self.envs),
            workdir=str(Path.cwd()),
        )
        task.set_resources(sky_resources)
        if num_nodes > 1:
            task.num_nodes = num_nodes

        # File mounts ship the orchestrator's staged manifest + per-job
        # payload to the cluster. The manifest dir is mounted as a whole;
        # per-job payloads land at deterministic paths so each cluster
        # only carries the bytes for its one job.
        local_payload = snapshot.payload_dir / payload_filename
        remote_payload = snapshot.REMOTE_PAYLOAD_DIR / payload_filename
        file_mounts: dict[str, str] = {
            str(snapshot.REMOTE_MANIFEST_DIR): str(snapshot.manifest_dir),
            str(remote_payload): str(local_payload),
        }
        task.set_file_mounts(file_mounts)

        # Persistent caches: each entry becomes a SkyPilot Storage that's
        # provisioned once and re-mounted on every future cluster, so model
        # downloads and wheel caches survive across jobs. MOUNT_CACHED layers
        # a local-SSD cache over the FUSE mount for faster repeat reads.
        if self.persistent_caches:
            storage_mounts = {
                path: sky.Storage(name=name, persistent=True, mode=sky.StorageMode.MOUNT_CACHED)
                for path, name in self.persistent_caches.items()
            }
            task.set_storage_mounts(storage_mounts)

        return task

    def _wait_and_launch(
        self,
        *,
        sky_task: object,
        managed_job_name: str,
        dependencies: set[SkyPilotJob],
        label: str,
    ) -> int:
        """Block on dependency completion, then submit a managed job.

        Runs in the launcher thread pool. Concurrent calls for
        independent work units submit to the managed-jobs controller in
        parallel; the controller itself parallelizes cluster
        provisioning across submissions.
        """
        for dep in dependencies:
            dep.wait()
            if dep.state() == "failed":
                msg = f"Dependency {dep.label} failed before {label} could launch."
                raise RuntimeError(msg)

        import sky.jobs

        logger.info("sky.jobs.launch starting for %s (name=%s).", label, managed_job_name)
        result = sky.jobs.launch(sky_task, name=managed_job_name, detach_run=True, stream_logs=False)
        managed_job_id = _extract_managed_job_id(result)
        logger.info("sky.jobs.launch returned managed_job_id=%s for %s.", managed_job_id, label)
        return managed_job_id


def _constant_assigned_resources(
    env: Mapping[str, str] | None = None,
    *,
    gpu_runtime: GpuRuntime = "cuda",  # noqa: ARG001
    assigned: AssignedResources,
) -> AssignedResources:
    """Picklable getter that returns a pre-computed AssignedResources.

    Defined at module scope (not a lambda or nested closure) so cloudpickle
    can serialize it cheaply for transport in the work-unit payload.
    """
    _ = env
    return assigned


# Managed-job statuses reported by ``sky.jobs.queue``. A managed job
# stays in RECOVERING / RETRYING during transient failures; misen
# folds those into ``running`` since the controller will eventually
# resolve them. Only terminal failures map to ``failed``.
_MANAGED_STATE_MAP: dict[str, JobState] = {
    "PENDING": "pending",
    "SUBMITTED": "pending",
    "STARTING": "pending",
    "PROVISIONING": "pending",
    "SETTING_UP": "pending",
    "RUNNING": "running",
    "RECOVERING": "running",
    "RETRYING": "running",
    "SUCCEEDED": "done",
    "FAILED": "failed",
    "FAILED_DRIVER": "failed",
    "FAILED_SETUP": "failed",
    "FAILED_PRECHECKS": "failed",
    "FAILED_NO_RESOURCE": "failed",
    "FAILED_CONTROLLER": "failed",
    "CANCELLED": "failed",
    "CANCELLING": "failed",
}


def _query_managed_job_states() -> dict[int, JobState]:
    """Snapshot every managed job's current state in one ``sky.jobs.queue`` call.

    The managed-jobs controller exposes one global queue for the user's
    account, so we don't need a per-cluster query. Returns
    ``managed_job_id -> normalized JobState``; callers map missing ids to
    ``"unknown"``.
    """
    import sky.jobs

    try:
        rows = sky.jobs.queue(refresh=False)
    except (ValueError, RuntimeError):
        return {}

    states: dict[int, JobState] = {}
    for row in rows:
        managed_job_id = row.get("job_id") if isinstance(row, dict) else getattr(row, "job_id", None)
        raw_status = row.get("status") if isinstance(row, dict) else getattr(row, "status", None)
        if managed_job_id is None or raw_status is None:
            continue
        status_str = getattr(raw_status, "value", None) or str(raw_status)
        states[int(managed_job_id)] = _MANAGED_STATE_MAP.get(status_str.upper(), "unknown")
    return states


def _extract_managed_job_id(result: object) -> int:
    """Pull the managed-job id out of ``sky.jobs.launch``'s return value.

    SkyPilot's API has shifted shape across versions: older releases
    returned ``(job_id, handle)``, newer releases sometimes return a
    request id resolvable via ``sky.get`` or just the bare id. Accept
    the common shapes.
    """
    if isinstance(result, tuple) and len(result) >= 1 and isinstance(result[0], int):
        return result[0]
    if isinstance(result, int):
        return result
    job_id = getattr(result, "job_id", None)
    if isinstance(job_id, int):
        return job_id
    msg = f"Unrecognized sky.jobs.launch return shape: {result!r}"
    raise RuntimeError(msg)


