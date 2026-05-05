"""SkyPilot executor behavior that doesn't require an actual cloud."""

from __future__ import annotations

import os
import sys
import types
from concurrent.futures import Future
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest

from misen import Task, meta
from misen.executor import Executor
from misen.utils.snapshot import CloudSnapshot
from misen.utils.work_unit import WorkUnit
from misen.workspace import Workspace
from misen.workspaces.memory import InMemoryWorkspace

if TYPE_CHECKING:
    from collections.abc import Iterator


@meta(id="skypilot_test_task", cache=False)
def _skypilot_test_task(x: int = 0) -> int:
    return x


def _stage_project(tmp_path: Path) -> Path:
    """Create a minimal CWD-shaped project tree the snapshot can read from."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "0.1"\n')
    (tmp_path / "uv.lock").write_text("# placeholder\n")
    return tmp_path


@pytest.fixture
def fake_sky(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.SimpleNamespace]:
    """Install a minimal stub `sky` (and `sky.jobs`) module."""
    sky = types.ModuleType("sky")
    sky_jobs = types.ModuleType("sky.jobs")

    class _Resources:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class _Task:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.resources: object | None = None
            self.file_mounts: dict[str, str] | None = None
            self.num_nodes = 1

        def set_resources(self, r: object) -> None:
            self.resources = r

        def set_file_mounts(self, m: dict[str, str]) -> None:
            self.file_mounts = dict(m)

    managed_launches: list[dict[str, Any]] = []
    managed_queue: list[dict[str, Any]] = []

    def _jobs_launch(task: object, **kwargs: Any) -> tuple[int, object]:
        managed_launches.append({"task": task, **kwargs})
        return (len(managed_launches), object())

    def _jobs_queue(refresh: bool = False) -> list[dict[str, Any]]:  # noqa: ARG001
        return list(managed_queue)

    class _Cloud:
        def __init__(self, name: str) -> None:
            self.name = name

    class _CloudRegistry:
        def from_str(self, name: str) -> _Cloud:
            return _Cloud(name)

    sky.Resources = _Resources  # type: ignore[attr-defined]
    sky.Task = _Task  # type: ignore[attr-defined]
    sky.clouds = types.SimpleNamespace(CLOUD_REGISTRY=_CloudRegistry())  # type: ignore[attr-defined]
    sky.jobs = sky_jobs  # type: ignore[attr-defined]
    sky_jobs.launch = _jobs_launch  # type: ignore[attr-defined]
    sky_jobs.queue = _jobs_queue  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "sky", sky)
    monkeypatch.setitem(sys.modules, "sky.jobs", sky_jobs)
    yield types.SimpleNamespace(
        module=sky,
        jobs_module=sky_jobs,
        launches=managed_launches,
        queue=managed_queue,
    )
    sys.modules.pop("sky", None)
    sys.modules.pop("sky.jobs", None)


def test_alias_resolves_to_skypilot_executor() -> None:
    cls = Executor.resolve_type("skypilot")
    assert cls.__module__ == "misen.executors.skypilot"
    assert cls.__name__ == "SkyPilotExecutor"


def test_workspace_capability_flag() -> None:
    from misen.workspaces.cloud import CloudWorkspace
    from misen.workspaces.disk import DiskWorkspace

    assert Workspace.supports_remote_executor is False
    assert CloudWorkspace.supports_remote_executor is True
    assert DiskWorkspace.supports_remote_executor is False
    assert InMemoryWorkspace.supports_remote_executor is False


def test_executor_init_without_sky_installed_raises_friendly_error() -> None:
    sys.modules.pop("sky", None)
    from misen.executors.skypilot import SkyPilotExecutor

    with pytest.raises(ImportError, match=r"misen\[skypilot\]"):
        SkyPilotExecutor()


def test_cloud_snapshot_stages_manifest_and_env_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _stage_project(tmp_path / "project")
    (project / ".env").write_text("FOO=bar\n")
    (project / ".env.local").write_text("SECRET=baz\n")
    monkeypatch.chdir(project)

    snap = CloudSnapshot(snapshots_dir=tmp_path / "snapshots")
    try:
        names = sorted(p.name for p in snap.manifest_dir.iterdir())
        assert names == [".env", ".env.local", "pyproject.toml", "uv.lock"]
        assert snap.staged_env_files == [".env", ".env.local"]
        assert snap.staged_pixi is False
        env_local = snap.manifest_dir / ".env.local"
        # Permission tightening only applies on POSIX; skip on Windows-y FS modes.
        if os.name == "posix":
            assert (env_local.stat().st_mode & 0o077) == 0
    finally:
        snap.cleanup()
    assert not snap.snapshot_dir.exists()


def test_cloud_snapshot_requires_pyproject(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="pyproject.toml"):
        CloudSnapshot(snapshots_dir=tmp_path / "snapshots")


def test_cloud_snapshot_rejects_pixi_lock_without_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "0.1"\n')
    (project / "uv.lock").write_text("# placeholder\n")
    (project / "pixi.lock").write_text("version: 6\n")
    monkeypatch.chdir(project)

    with pytest.raises(RuntimeError, match="pixi.lock"):
        CloudSnapshot(snapshots_dir=tmp_path / "snapshots")


def test_cloud_snapshot_prepare_job_emits_remote_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "0.1"\n')
    (project / "uv.lock").write_text("# placeholder\n")
    (project / ".env").write_text("FOO=bar\n")
    monkeypatch.chdir(project)

    snap = CloudSnapshot(snapshots_dir=tmp_path / "snapshots")
    try:
        ws = InMemoryWorkspace(directory=str(tmp_path / "ws"))
        work_unit = WorkUnit(root=Task(_skypilot_test_task, x=1), dependencies=set())

        job_id, argv, env_overrides, log_path = snap.prepare_job(
            work_unit=work_unit,
            workspace=ws,
            assigned_resources_getter=lambda: None,
            gpu_runtime="cuda",
            bind_gpu_env=False,
        )

        # job_id surfaces as the payload filename and the suffix of log filename.
        payload_pkl = snap.payload_dir / f"{job_id}.pkl"
        assert payload_pkl.exists()
        assert payload_pkl.stat().st_size > 0
        assert env_overrides == {}

        argv_str = " ".join(argv)
        # Argv references the deterministic remote layout, never local paths.
        assert str(snap.REMOTE_PAYLOAD_DIR / f"{job_id}.pkl") in argv_str
        assert str(snap.REMOTE_MANIFEST_DIR / ".env") in argv_str
        # No pixi wrap when no pixi.toml was staged.
        assert "pixi" not in argv_str
        # Worker entry, with --no-bind-gpu-env propagated.
        assert "--payload" in argv
        assert "misen.utils.execute" in argv
        assert "--no-bind-gpu-env" in argv
        # Returned log_path is the *remote* path (~/.misen/job_logs/...).
        assert str(snap.REMOTE_LOG_DIR) in str(log_path)
    finally:
        snap.cleanup()
        ws.close()


def test_executor_rejects_workspace_without_remote_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_sky: types.SimpleNamespace
) -> None:
    _ = fake_sky
    project = _stage_project(tmp_path / "project")
    monkeypatch.chdir(project)

    from misen.executors.skypilot import SkyPilotExecutor

    executor = SkyPilotExecutor()
    ws = InMemoryWorkspace(directory=str(tmp_path / "ws"))
    try:
        with pytest.raises(RuntimeError, match="supports_remote_executor"):
            executor._make_snapshot(workspace=ws)  # noqa: SLF001
    finally:
        ws.close()


def test_executor_make_snapshot_accepts_remote_capable_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_sky: types.SimpleNamespace
) -> None:
    _ = fake_sky
    project = _stage_project(tmp_path / "project")
    monkeypatch.chdir(project)

    # Pose as a remote-capable workspace by tagging an InMemoryWorkspace.
    class _RemoteWorkspace(InMemoryWorkspace):
        supports_remote_executor = True

    from misen.executors.skypilot import SkyPilotExecutor

    executor = SkyPilotExecutor()
    ws = _RemoteWorkspace(directory=str(tmp_path / "ws"))
    try:
        snap = executor._make_snapshot(workspace=ws)  # noqa: SLF001
        try:
            assert isinstance(snap, CloudSnapshot)
            assert snap.snapshot_dir.exists()
        finally:
            snap.cleanup()
    finally:
        ws.close()


def test_skypilot_job_state_pending_until_future_resolves() -> None:
    from misen.executors.skypilot import SkyPilotJob

    work_unit = WorkUnit(root=Task(_skypilot_test_task, x=2), dependencies=set())
    workspace = cast("Workspace", MagicMock(spec=Workspace))
    job = SkyPilotJob(work_unit=work_unit, managed_job_name="misen-test", workspace=workspace)

    # No future attached: job is pending.
    assert job.state() == "pending"

    future: Future[int] = Future()
    job.attach_launch_future(future)
    # Future not done: still pending.
    assert job.state() == "pending"

    # Future resolves with success: state then comes from sky.jobs.queue.
    future.set_result(7)
    fake_sky_mod = types.ModuleType("sky")
    fake_jobs_mod = types.ModuleType("sky.jobs")
    fake_jobs_mod.queue = lambda refresh=False: [{"job_id": 7, "status": "RUNNING"}]  # type: ignore[attr-defined]
    fake_sky_mod.jobs = fake_jobs_mod  # type: ignore[attr-defined]
    sys.modules["sky"] = fake_sky_mod
    sys.modules["sky.jobs"] = fake_jobs_mod
    try:
        assert job.state() == "running"

        # Terminal state caches.
        fake_jobs_mod.queue = lambda refresh=False: [{"job_id": 7, "status": "SUCCEEDED"}]  # type: ignore[attr-defined]
        assert job.state() == "done"
        # Even after the queue forgets the job, the cached terminal state holds.
        fake_jobs_mod.queue = lambda refresh=False: []  # type: ignore[attr-defined]
        assert job.state() == "done"
    finally:
        sys.modules.pop("sky", None)
        sys.modules.pop("sky.jobs", None)


def test_skypilot_job_bulk_state_uses_one_queue_call_for_many_jobs() -> None:
    """Managed jobs share one global queue, so N jobs cost one ``sky.jobs.queue`` call."""
    from misen.executors.skypilot import SkyPilotJob

    workspace = cast("Workspace", MagicMock(spec=Workspace))
    jobs: list[SkyPilotJob] = []
    for i in range(5):
        wu = WorkUnit(root=Task(_skypilot_test_task, x=i), dependencies=set())
        job = SkyPilotJob(work_unit=wu, managed_job_name=f"misen-{i}", workspace=workspace)
        future: Future[int] = Future()
        future.set_result(100 + i)
        job.attach_launch_future(future)
        jobs.append(job)

    queue_data = [
        {"job_id": 100, "status": "RUNNING"},
        {"job_id": 101, "status": "SUCCEEDED"},
        {"job_id": 102, "status": "FAILED"},
        {"job_id": 103, "status": "PENDING"},
        # 104 deliberately omitted -> "unknown"
    ]
    queue_calls = [0]

    def _queue(refresh: bool = False) -> list[dict[str, Any]]:  # noqa: ARG001
        queue_calls[0] += 1
        return list(queue_data)

    fake_sky_mod = types.ModuleType("sky")
    fake_jobs_mod = types.ModuleType("sky.jobs")
    fake_jobs_mod.queue = _queue  # type: ignore[attr-defined]
    fake_sky_mod.jobs = fake_jobs_mod  # type: ignore[attr-defined]
    sys.modules["sky"] = fake_sky_mod
    sys.modules["sky.jobs"] = fake_jobs_mod
    try:
        states = SkyPilotJob.bulk_state(jobs)
    finally:
        sys.modules.pop("sky", None)
        sys.modules.pop("sky.jobs", None)

    assert queue_calls[0] == 1
    assert states[jobs[0]] == "running"
    assert states[jobs[1]] == "done"
    assert states[jobs[2]] == "failed"
    assert states[jobs[3]] == "pending"
    assert states[jobs[4]] == "unknown"


def test_skypilot_job_state_failed_when_launch_future_raises() -> None:
    from misen.executors.skypilot import SkyPilotJob

    work_unit = WorkUnit(root=Task(_skypilot_test_task, x=3), dependencies=set())
    workspace = cast("Workspace", MagicMock(spec=Workspace))
    job = SkyPilotJob(work_unit=work_unit, managed_job_name="misen-test", workspace=workspace)

    future: Future[int] = Future()
    job.attach_launch_future(future)
    future.set_exception(RuntimeError("provisioning failed"))

    assert job.state() == "failed"
    # Cached after first observation.
    assert job.state() == "failed"
