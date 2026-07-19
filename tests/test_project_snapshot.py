"""Tests for ``ProjectSnapshot`` staging, workspace publication, and the bootstrap.

Staging tests assert what a snapshot stages and publishes (and that no env
is built without ``prewarm``); bootstrap tests run the worker-side env
build for real against a tiny generated workspace project, capturing the
final ``exec`` instead of replacing the test process and then running the
captured command to prove the envs work. Cloud-workspace publication is
exercised hermetically through obstore's ``MemoryStore``.
"""
# ruff: noqa: ARG002, D103, FBT001, PLR2004, S101, SLF001

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import tyro

from misen.utils import bootstrap_env
from misen.utils import snapshot as snapshot_mod
from misen.utils.snapshot import (
    BOOTSTRAP_MODULE,
    BOOTSTRAP_TRANSPORT_ENV,
    JOB_LOG_PATH_ARG,
    ProjectSnapshot,
    _is_pure_wheel,
    _misen_bootstrap_requirement,
    _uv_bin,
)
from misen.workspace import Workspace
from misen.workspaces.disk import DiskWorkspace
from tests.test_cloud_workspace import _MemoryCloudWorkspace
from tests.test_shared_env_store import _write_project

ExecCall = tuple[str, list[str], dict[str, str]]


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Module-shared tiny workspace project (tests must not mutate it)."""
    root = tmp_path_factory.mktemp("proj_parent") / "project"
    _write_project(root)
    return root


@pytest.fixture
def in_project(project: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(project)
    return project


@pytest.fixture
def counted_run(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Record every subprocess argv issued by the snapshot module."""
    recorded: list[list[str]] = []
    real_run = subprocess.run

    def recording_run(argv: list[str], *args: object, **kwargs: object) -> object:
        recorded.append(list(argv))
        return real_run(argv, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(snapshot_mod.subprocess, "run", recording_run)
    return recorded


@pytest.fixture
def captured_exec(monkeypatch: pytest.MonkeyPatch) -> list[ExecCall]:
    """Capture ``os.execve`` calls issued by the bootstrap instead of exec'ing."""
    captured: list[ExecCall] = []
    monkeypatch.setattr(os, "execve", lambda path, argv, env: captured.append((path, list(argv), dict(env))))
    return captured


def _env_build_calls(recorded: list[list[str]]) -> list[list[str]]:
    """Subprocess calls that create or populate an environment."""
    return [argv for argv in recorded if "sync" in argv or "venv" in argv or "install" in argv]


class _StubHash:
    def b32(self) -> str:
        return "STUBHASH"


class _StubRoot:
    def task_hash(self) -> _StubHash:
        return _StubHash()


class _StubWorkUnit:
    root = _StubRoot()

    def as_payload(self, *, workspace: object, job_id: str) -> bytes:
        return b"payload"


# ---------- staging + publication ----------


@pytest.mark.usefixtures("in_project")
def test_staging_contents_and_key_stability(tmp_path: Path, counted_run: list[list[str]]) -> None:
    workspace = DiskWorkspace(directory=str(tmp_path / ".misen"))
    snapshot = ProjectSnapshot(workspace=workspace, prewarm=False)

    # No environment is built without prewarm.
    assert _env_build_calls(counted_run) == []
    assert snapshot.prewarmed is None

    project_dir = snapshot.project_dir
    assert workspace.has_snapshot(snapshot.snapshot_key)
    assert project_dir == workspace.fetch_snapshot(snapshot.snapshot_key)
    for name in ("pyproject.toml", "uv.lock", "requirements.txt"):
        assert (project_dir / name).is_file()

    # Local packages become artifacts (all pure -> wheels); remote deps
    # become requirement pins.
    artifact_names = sorted(p.name for p in (project_dir / "packages").iterdir())
    assert [name.split("-")[0] for name in artifact_names] == ["mainpkg", "member", "pathdep"]
    assert all(name.endswith(".whl") for name in artifact_names)
    requirement_lines = [
        line
        for line in (project_dir / "requirements.txt").read_text().splitlines()
        if line and not line[0].isspace() and not line.startswith("#")
    ]
    assert len(requirement_lines) == 1
    assert requirement_lines[0].startswith("iniconfig==")

    # No misen in the lock: bootstrap dispatch must fail with a clear error.
    assert snapshot.misen_requirement is None
    with pytest.raises(RuntimeError, match="installable misen"):
        snapshot.prepare_job(
            _StubWorkUnit(),  # type: ignore[arg-type]
            workspace,
            "cuda",
            cpu_indices=None,
            gpu_indices=None,
        )

    # Identical code -> identical content key (SOURCE_DATE_EPOCH pins the
    # artifact bytes), so a resubmission republishes nothing.
    second = ProjectSnapshot(workspace=workspace, prewarm=False)
    assert second.snapshot_key == snapshot.snapshot_key


@pytest.mark.usefixtures("in_project")
def test_env_files_become_submission_job_files(tmp_path: Path, project: Path) -> None:
    workspace = DiskWorkspace(directory=str(tmp_path / ".misen"))
    (project / ".env").write_text("A=1\n")
    (project / ".env.local").write_text("SECRET=1\n")
    try:
        snapshot = ProjectSnapshot(workspace=workspace, prewarm=False)
        # Env files are submission-scoped job files, never snapshot content.
        assert not (snapshot.project_dir / ".env").exists()
        assert len(snapshot.env_file_refs) == 2
        paths = [Path(ref) for ref in snapshot.env_file_refs]
        assert [p.name for p in paths] == [".env", ".env.local"]
        assert all(oct(p.stat().st_mode & 0o777) == "0o600" for p in paths)
        # Env-file copies are retained (0600) until pruning; nothing
        # deletes them when a submission ends.
    finally:
        (project / ".env").unlink()
        (project / ".env.local").unlink()


def test_cloud_workspace_publication_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Snapshots publish to the object store and re-materialize on a fresh host."""
    root = tmp_path / "project"
    _write_project(root)
    monkeypatch.chdir(root)
    workspace = _MemoryCloudWorkspace(backend="s3", bucket="snap-test", cache_dir=str(tmp_path / "cache-a"))

    snapshot = ProjectSnapshot(workspace=workspace, prewarm=False)
    assert workspace.has_snapshot(snapshot.snapshot_key)

    # A second workspace over the same bucket with a cold cache = a worker.
    worker_side = _MemoryCloudWorkspace(backend="s3", bucket="snap-test", cache_dir=str(tmp_path / "cache-b"))
    fetched = worker_side.fetch_snapshot(snapshot.snapshot_key)
    assert (fetched / "requirements.txt").read_bytes() == (snapshot.project_dir / "requirements.txt").read_bytes()
    assert sorted(p.name for p in (fetched / "packages").iterdir()) == sorted(
        p.name for p in (snapshot.project_dir / "packages").iterdir()
    )


# ---------- artifact selection ----------


@pytest.mark.parametrize(
    ("name", "pure"),
    [
        ("mainpkg-0.1.0-py3-none-any.whl", True),
        ("pkg-1.0-py2.py3-none-any.whl", True),
        ("nativepkg-2.0-cp313-cp313-macosx_11_0_arm64.whl", False),
        ("nativepkg-2.0-cp313-abi3-manylinux_2_17_x86_64.whl", False),
    ],
)
def test_is_pure_wheel(name: str, pure: bool) -> None:
    assert _is_pure_wheel(Path(name)) is pure


def test_broken_wheel_build_falls_back_to_sdist(tmp_path: Path) -> None:
    """A package whose wheel cannot build here still stages as an sdist."""
    package_dir = tmp_path / "brokenwheel"
    package_dir.mkdir()
    # hatchling can always build the sdist, but the wheel target fails
    # because no package directory matches.
    (package_dir / "pyproject.toml").write_text(
        "[project]\n"
        'name = "brokenwheel"\nversion = "0.1.0"\nrequires-python = ">=3.11"\n'
        "[build-system]\n"
        'requires = ["hatchling"]\nbuild-backend = "hatchling.build"\n'
    )
    packages_dir = tmp_path / "packages"
    packages_dir.mkdir()
    snapshot_mod._stage_local_package(package_dir, packages_dir)
    staged = sorted(p.name for p in packages_dir.iterdir())
    assert staged == ["brokenwheel-0.1.0.tar.gz"]


# ---------- misen bootstrap requirement resolution ----------


def _requirement_for_lock(tmp_path: Path, lock_body: str, *, paths: bool = True) -> str | None:
    project_dir = tmp_path / "staged"
    (project_dir / "packages").mkdir(parents=True, exist_ok=True)
    (project_dir / "uv.lock").write_text(f"version = 1\n{lock_body}")

    class _StubWorkspace:
        job_files_are_paths = paths

    return _misen_bootstrap_requirement(project_dir, workspace=_StubWorkspace())  # type: ignore[arg-type]


def test_misen_requirement_registry_pin(tmp_path: Path) -> None:
    body = '[[package]]\nname = "misen"\nversion = "0.0.9"\n[package.source]\nregistry = "https://pypi.org/simple"\n'
    assert _requirement_for_lock(tmp_path, body) == "misen==0.0.9"


def test_misen_requirement_local_checkout_uses_staged_artifact(tmp_path: Path) -> None:
    body = '[[package]]\nname = "misen"\nversion = "0.0.9"\n[package.source]\neditable = "."\n'
    project_dir = tmp_path / "staged"
    (project_dir / "packages").mkdir(parents=True)
    (project_dir / "uv.lock").write_text(f"version = 1\n{body}")
    wheel = project_dir / "packages" / "misen-0.0.9-py3-none-any.whl"
    wheel.write_bytes(b"wheel")

    class _PathsWorkspace:
        job_files_are_paths = True

    class _RemoteWorkspace:
        job_files_are_paths = False

    assert _misen_bootstrap_requirement(project_dir, workspace=_PathsWorkspace()) == str(wheel)  # type: ignore[arg-type]
    # No shared filesystem -> local misen is unusable (released misen required).
    assert _misen_bootstrap_requirement(project_dir, workspace=_RemoteWorkspace()) is None  # type: ignore[arg-type]


def test_misen_requirement_absent_or_git(tmp_path: Path) -> None:
    assert _requirement_for_lock(tmp_path, "") is None
    git_body = '[[package]]\nname = "misen"\nversion = "0.0.9"\n[package.source]\ngit = "https://x.invalid/misen"\n'
    assert _requirement_for_lock(tmp_path, git_body) is None


# ---------- bootstrap dispatch argv ----------


def test_bootstrap_transports(tmp_path: Path) -> None:
    disk = DiskWorkspace(directory=str(tmp_path / ".misen"))
    assert disk.bootstrap_transport() == {"kind": "path"}

    cloud = _MemoryCloudWorkspace(backend="s3", bucket="transport-test", cache_dir=str(tmp_path / "cache"))
    transport = cloud.bootstrap_transport()
    assert transport["kind"] == "obstore"
    assert transport["backend"] == "s3"
    assert transport["bucket"] == "transport-test"

    class _NoTransportWorkspace:
        job_files_are_paths = False
        bootstrap_transport = Workspace.bootstrap_transport

    with pytest.raises(NotImplementedError, match="bootstrap transport"):
        _NoTransportWorkspace().bootstrap_transport()


@pytest.mark.usefixtures("in_project")
def test_bootstrap_dispatch_argv(tmp_path: Path) -> None:
    """Path-transport workspaces dispatch the bootstrap with plain paths."""
    workspace = DiskWorkspace(directory=str(tmp_path / ".misen"))
    snapshot = ProjectSnapshot(workspace=workspace, env_store_dir="/scratch/envs", prewarm=False)
    snapshot.misen_requirement = "misen==0.0.9"  # the tiny project has no misen dep
    _job_id, argv, env_overrides, log_path = snapshot.prepare_job(
        _StubWorkUnit(),  # type: ignore[arg-type]
        workspace,
        "cuda",
        cpu_indices=[0, 1],
        gpu_indices=None,
    )
    # Path transport: no env var, no refs — everything is a path.
    assert env_overrides == {}
    assert "--snapshot-key" not in argv
    assert BOOTSTRAP_TRANSPORT_ENV not in os.environ

    assert argv[0] == _uv_bin()
    assert argv[1:3] == ["run", "--no-project"]
    assert argv[argv.index("--with") + 1] == "misen==0.0.9"
    assert argv[argv.index("-m") + 1] == BOOTSTRAP_MODULE
    assert argv[argv.index("--project-dir") + 1] == str(snapshot.project_dir)
    assert argv[argv.index("--env-store-root") + 1] == "/scratch/envs"
    payload_path = argv[argv.index("--payload") + 1]
    assert Path(payload_path).read_bytes() == b"payload"
    assert argv[argv.index("--cpu-indices") + 1 : argv.index("--cpu-indices") + 3] == ["0", "1"]
    assert argv[argv.index(JOB_LOG_PATH_ARG) + 1] == str(log_path)
    assert "misen.utils.execute" not in argv  # inner argv is built by the bootstrap


# ---------- bootstrap (worker side, real uv) ----------


def _run_bootstrap(snapshot: ProjectSnapshot, store_root: Path | str, payload_path: str) -> None:
    """Invoke the bootstrap in path mode, as a path-transport dispatch would."""
    tyro.cli(
        bootstrap_env.main,
        args=[
            "--project-dir",
            str(snapshot.project_dir),
            "--env-store-root",
            str(store_root),
            "--payload",
            payload_path,
            "--gpu-runtime",
            "cuda",
            JOB_LOG_PATH_ARG,
            str(Path(store_root) / "job.log"),
        ],
    )


@pytest.mark.usefixtures("in_project")
def test_bootstrap_builds_reuses_and_execs(
    tmp_path: Path, counted_run: list[list[str]], captured_exec: list[ExecCall]
) -> None:
    workspace = DiskWorkspace(directory=str(tmp_path / ".misen"))
    store_root = tmp_path / "env-store"
    snapshot = ProjectSnapshot(workspace=workspace, prewarm=False)
    payload_ref = workspace.put_job_file(snapshot.submission_id, "JOB.pkl", b"payload")

    _run_bootstrap(snapshot, store_root, payload_ref)
    path, argv, env = captured_exec[-1]
    assert path == argv[0] == _uv_bin()
    assert "misen.utils.execute" in argv
    assert argv[argv.index("--payload") + 1] == payload_ref  # disk refs are paths

    overlay_venv = Path(env["VIRTUAL_ENV"])
    assert overlay_venv.is_relative_to(store_root / "overlay-envs")
    deps_bin = Path(env["PATH"].split(os.pathsep)[0])
    assert deps_bin.name == "bin"
    assert deps_bin.parent.is_relative_to(store_root / "python-envs")
    assert Path(env["PYTHONPATH"].split(os.pathsep)[0]).is_relative_to(overlay_venv)

    # The exec'd env really runs: local wheels + registry deps import.
    check = (
        "import iniconfig, mainpkg, member, pathdep\n"
        "import importlib.metadata as m\n"
        "assert m.version('member') == '0.2.0'\n"
        f"assert __import__('sys').prefix == {str(overlay_venv)!r}\n"
    )
    subprocess.run(  # noqa: S603
        [argv[0], "run", "--no-project", "python", "-c", check], env=env, check=True, capture_output=True
    )

    # A second bootstrap (same snapshot, same host) reuses both entries.
    builds = len(_env_build_calls(counted_run))
    _run_bootstrap(snapshot, store_root, payload_ref)
    assert len(_env_build_calls(counted_run)) == builds
    assert captured_exec[-1][2]["VIRTUAL_ENV"] == str(overlay_venv)

    # Prewarming afterwards reuses the same store entries too: one build
    # path regardless of which side runs it.
    prewarmed = ProjectSnapshot(workspace=workspace, env_store_dir=str(store_root), prewarm=True)
    assert prewarmed.prewarmed is not None
    assert prewarmed.prewarmed.overlay_venv_dir == overlay_venv
    assert len(_env_build_calls(counted_run)) == builds


def test_bootstrap_normalizes_relative_store_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, captured_exec: list[ExecCall]
) -> None:
    """Relative store roots pin down once, before builds run under other CWDs."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname = "virtualproj"\nversion = "0.0.0"\nrequires-python = ">=3.11"\ndependencies = []\n'
    )
    monkeypatch.chdir(root)
    subprocess.run(["uv", "lock"], check=True, capture_output=True)  # noqa: S607
    workspace = DiskWorkspace(directory=str(tmp_path / ".misen"))

    snapshot = ProjectSnapshot(workspace=workspace, prewarm=False)
    assert list((snapshot.project_dir / "packages").iterdir()) == []  # virtual root stages nothing

    payload_ref = workspace.put_job_file(snapshot.submission_id, "JOB.pkl", b"payload")
    _run_bootstrap(snapshot, "env-store", payload_ref)  # relative to CWD
    _, _, env = captured_exec[-1]
    assert Path(env["VIRTUAL_ENV"]).is_relative_to((root / "env-store").absolute())


def test_obstore_bootstrap_fetches_and_execs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, captured_exec: list[ExecCall]
) -> None:
    """Ref-mode bootstrap: fetch snapshot + job files via the obstore transport.

    Hermetic: the transport's store builder is patched to return the shared
    ``MemoryStore``, so this exercises exactly the code path a real worker
    runs against S3/GCS/Azure — reconstruction from the transport env var,
    tarball fetch into the env store, job-file fetch, activation, exec.
    """
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname = "virtualproj"\nversion = "0.0.0"\nrequires-python = ">=3.11"\ndependencies = []\n'
    )
    monkeypatch.chdir(root)
    subprocess.run(["uv", "lock"], check=True, capture_output=True)  # noqa: S607

    workspace = _MemoryCloudWorkspace(backend="s3", bucket="boot-test", cache_dir=str(tmp_path / "cache"))
    snapshot = ProjectSnapshot(workspace=workspace, prewarm=False)
    assert snapshot.transport is not None
    assert snapshot.transport["kind"] == "obstore"
    payload_ref = workspace.put_job_file(snapshot.submission_id, "JOB.pkl", b"payload")
    env_ref = workspace.put_job_file(snapshot.submission_id, ".env", b"A=1\n")

    from misen.workspaces import cloud as cloud_mod

    monkeypatch.setattr(cloud_mod, "_build_obstore_store", lambda *_args, **_kwargs: workspace._store)
    monkeypatch.setenv(BOOTSTRAP_TRANSPORT_ENV, json.dumps(snapshot.transport))
    store_root = tmp_path / "env-store"
    tyro.cli(
        bootstrap_env.main,
        args=[
            "--snapshot-key",
            snapshot.snapshot_key,
            "--env-store-root",
            str(store_root),
            "--payload-ref",
            payload_ref,
            "--env-file-ref",
            env_ref,
            "--gpu-runtime",
            "cuda",
            JOB_LOG_PATH_ARG,
            str(store_root / "job.log"),
        ],
    )

    _, argv, env = captured_exec[-1]
    # Snapshot fetched (content-addressed) and job files fetched into the env store root.
    payload_path = Path(argv[argv.index("--payload") + 1])
    assert payload_path.is_relative_to(store_root / "job-files")
    assert payload_path.read_bytes() == b"payload"
    env_file_path = Path(argv[argv.index("--env-file") + 1])
    assert env_file_path.read_bytes() == b"A=1\n"
    assert oct(env_file_path.stat().st_mode & 0o777) == "0o600"
    assert (store_root / "snapshots" / snapshot.snapshot_key / "requirements.txt").is_file()
    assert Path(env["VIRTUAL_ENV"]).is_relative_to(store_root / "overlay-envs")
    assert BOOTSTRAP_TRANSPORT_ENV not in env


# ---------- executor plumbing ----------


@pytest.mark.usefixtures("in_project")
def test_executor_validation(tmp_path: Path) -> None:
    from misen.executors.slurm import SlurmExecutor

    disk = DiskWorkspace(directory=str(tmp_path / ".misen"))
    cloud = _MemoryCloudWorkspace(backend="s3", bucket="val-test", cache_dir=str(tmp_path / "cache"))

    # SLURM prewarm needs an explicit (shared) env store.
    with pytest.raises(ValueError, match="env_store_dir"):
        SlurmExecutor(prewarm_envs=True)._make_snapshot(disk)
    # Prewarm needs path-addressable job files.
    with pytest.raises(ValueError, match="worker-visible paths"):
        SlurmExecutor(prewarm_envs=True, env_store_dir=str(tmp_path / "s"))._make_snapshot(cloud)

    # snapshot=False -> live dispatch, no snapshot object at all.
    assert SlurmExecutor(snapshot=False)._make_snapshot(disk) is None


def test_in_process_executor_warns_on_snapshot_config(caplog: pytest.LogCaptureFixture) -> None:
    from misen.executors.in_process import InProcessExecutor

    with caplog.at_level("WARNING", logger="misen.executors.in_process"):
        executor = InProcessExecutor(snapshot=True)
    assert executor.snapshot is True
    assert any("ignores snapshot=True" in record.message for record in caplog.records)

    caplog.clear()
    with caplog.at_level("WARNING", logger="misen.executors.in_process"):
        InProcessExecutor()
    assert not caplog.records


@pytest.mark.usefixtures("in_project")
def test_slurm_executor_bootstrap_dispatch(tmp_path: Path) -> None:
    """Default SLURM config produces a bootstrap dispatch through the workspace."""
    from misen.executors.slurm import SlurmExecutor

    workspace = DiskWorkspace(directory=str(tmp_path / ".misen"))
    executor = SlurmExecutor(env_store_dir="/mnt/local/misen-envs")
    snapshot = executor._make_snapshot(workspace)
    assert isinstance(snapshot, ProjectSnapshot)
    assert snapshot.prewarmed is None
    assert snapshot.env_store_dir == "/mnt/local/misen-envs"
