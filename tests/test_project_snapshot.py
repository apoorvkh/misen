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

import hashlib
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import obstore.store as obstore_store
import pytest
import tyro

from misen.utils import materialize_env
from misen.utils import snapshot as snapshot_mod
from misen.utils.bootstrap_transport import render_python_transport, worker_bootstrap_script
from misen.utils.snapshot import (
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


def _test_python_transport(context: dict[str, object], operation: str, ref: str, destination: Path) -> None:
    destination.write_text(f"{context['prefix']}:{operation}:{ref}")


_CAPTURED_TRANSPORT_VALUE = "captured"


def _capturing_python_transport(context: dict[str, object], operation: str, ref: str, destination: Path) -> None:
    del context, operation, ref
    destination.write_text(_CAPTURED_TRANSPORT_VALUE)


# ---------- staging + publication ----------


@pytest.mark.usefixtures("in_project")
def test_staging_contents_and_key_stability(tmp_path: Path, counted_run: list[list[str]]) -> None:
    workspace = DiskWorkspace(directory=str(tmp_path / ".misen"))
    snapshot = ProjectSnapshot(workspace=workspace, prewarm=False)

    # No environment is built without prewarm.
    assert _env_build_calls(counted_run) == []
    assert snapshot.prewarmed is None

    project_dir = snapshot.project_dir
    assert project_dir == workspace.fetch_snapshot(snapshot.snapshot_key)
    for name in ("pyproject.toml", "uv.lock"):
        assert (project_dir / name).is_file()
    assert not (project_dir / "requirements.txt").exists()

    # Local packages become artifacts (all pure -> wheels); remote deps stay
    # represented by the frozen uv project.
    artifact_names = sorted(p.name for p in (project_dir / "packages").iterdir())
    assert [name.split("-")[0] for name in artifact_names] == ["mainpkg", "member", "pathdep"]
    assert all(name.endswith(".whl") for name in artifact_names)
    lock = (project_dir / "uv.lock").read_text()
    assert 'name = "iniconfig"' in lock

    # No misen in the lock: bootstrap dispatch must fail with a clear error.
    assert snapshot.misen_requirement is None
    with pytest.raises(RuntimeError, match="installable misen"):
        snapshot.prepare_job(
            _StubWorkUnit(),  # type: ignore[arg-type]
            workspace,
            cpu_indices=None,
            accelerator_type="cuda",
            accelerator_indices=None,
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
    # A second workspace over the same bucket with a cold cache = a worker.
    worker_side = _MemoryCloudWorkspace(backend="s3", bucket="snap-test", cache_dir=str(tmp_path / "cache-b"))
    fetched = worker_side.fetch_snapshot(snapshot.snapshot_key)
    assert (fetched / "pyproject.toml").read_bytes() == (snapshot.project_dir / "pyproject.toml").read_bytes()
    assert (fetched / "uv.lock").read_bytes() == (snapshot.project_dir / "uv.lock").read_bytes()
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


def _requirement_for_lock(tmp_path: Path, lock_body: str, *, paths_visible: bool = True) -> str | None:
    project_dir = tmp_path / "staged"
    (project_dir / "packages").mkdir(parents=True, exist_ok=True)
    (project_dir / "uv.lock").write_text(f"version = 1\n{lock_body}")

    return _misen_bootstrap_requirement(project_dir, paths_visible=paths_visible)


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

    assert _misen_bootstrap_requirement(project_dir, paths_visible=True) == str(wheel)
    # No shared filesystem -> a local checkout artifact is not worker-visible.
    assert _misen_bootstrap_requirement(project_dir, paths_visible=False) is None


def test_misen_requirement_absent_or_unresolved_git(tmp_path: Path) -> None:
    assert _requirement_for_lock(tmp_path, "") is None
    git_body = '[[package]]\nname = "misen"\nversion = "0.0.9"\n[package.source]\ngit = "https://x.invalid/misen"\n'
    assert _requirement_for_lock(tmp_path, git_body) is None


def test_misen_requirement_git_uses_locked_commit(tmp_path: Path) -> None:
    commit = "0123456789abcdef0123456789abcdef01234567"
    body = (
        '[[package]]\nname = "misen"\nversion = "0.0.9"\n[package.source]\n'
        f'git = "https://github.com/example/misen.git?branch=dev#{commit}"\n'
    )
    assert _requirement_for_lock(tmp_path, body) == f"misen @ git+https://github.com/example/misen.git@{commit}"


def test_misen_requirement_git_preserves_subdirectory(tmp_path: Path) -> None:
    commit = "fedcba9876543210fedcba9876543210fedcba98"
    body = (
        '[[package]]\nname = "misen"\nversion = "0.0.9"\n[package.source]\n'
        f'git = "git+ssh://git@example.com/mono.git?rev=main&subdirectory=packages%2Fmisen#{commit}"\n'
    )
    assert _requirement_for_lock(tmp_path, body) == (
        f"misen @ git+ssh://git@example.com/mono.git@{commit}#subdirectory=packages/misen"
    )


# ---------- bootstrap dispatch argv ----------


def test_worker_shell_bootstrap_resolves_configured_tools(tmp_path: Path) -> None:
    """The Bash root resolves required tools before entering uv."""
    echo_bin = shutil.which("echo")
    true_bin = shutil.which("true")
    bash_bin = shutil.which("bash")
    assert echo_bin is not None
    assert true_bin is not None
    assert bash_bin is not None
    script = worker_bootstrap_script(
        uv_bin=echo_bin,
        pixi_bin=true_bin,
        requires_pixi=True,
        transport_script=None,
        misen_requirement="misen==0.0.9",
        python_version="3.11",
        store_root=tmp_path / "store",
        project_dir=tmp_path / "project",
        snapshot_key=None,
        payload=str(tmp_path / "payload"),
        env_files=[],
        worker_args=[JOB_LOG_PATH_ARG, str(tmp_path / "job.log")],
    )
    assert '${env_file_paths[@]+"${env_file_paths[@]}"}' in script
    result = subprocess.run(  # noqa: S603
        [bash_bin, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "misen.utils.materialize_env" in result.stdout


def test_python_transport_renderer_extracts_and_invokes_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dependency_transport = render_python_transport(
        _test_python_transport,
        requirements=("example-package==1.2.3",),
        context={"prefix": "workspace-a"},
    )
    assert "example-package==1.2.3" in dependency_transport

    transport = render_python_transport(_test_python_transport, context={"prefix": "workspace-a"})
    assert "def _test_python_transport" in transport
    destination = tmp_path / "transport-output"
    bash_bin = shutil.which("bash")
    assert bash_bin is not None
    monkeypatch.setenv("MISEN_UV_BIN", _uv_bin())
    monkeypatch.setenv("MISEN_TRANSPORT_OPERATION", "job-file")
    monkeypatch.setenv("MISEN_TRANSPORT_REF", "opaque-ref")
    monkeypatch.setenv("MISEN_TRANSPORT_DEST", str(destination))
    subprocess.run([bash_bin, "-c", transport], check=True)  # noqa: S603
    assert destination.read_text() == "workspace-a:job-file:opaque-ref"


def test_python_transport_renderer_rejects_captured_globals() -> None:
    with pytest.raises(ValueError, match="module globals"):
        render_python_transport(_capturing_python_transport, context={})


def test_bootstrap_transports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    disk = DiskWorkspace(directory=str(tmp_path / ".misen"))
    assert disk.bootstrap_transport() is None

    cloud = _MemoryCloudWorkspace(backend="s3", bucket="transport-test", cache_dir=str(tmp_path / "cache"))
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "project.txt").write_text("snapshot")
    cloud.publish_snapshot("snapshot-key", staged)
    job_ref = cloud.put_job_file("submission", "job.pkl", b"payload")
    transport = cloud.bootstrap_transport()
    assert "obstore==" in transport
    assert "--with misen" not in transport.lower()
    assert "from misen" not in transport.lower()
    assert '"backend":"s3"' in transport
    assert '"bucket":"transport-test"' in transport
    subprocess.run(["bash", "-n"], input=transport, text=True, check=True)  # noqa: S607
    command = shlex.split(transport)
    code = compile(command[command.index("-c") + 1], "<cloud-bootstrap-transport>", "exec")

    # Execute the exact inline program against the hermetic MemoryStore. This
    # tests both operation branches without invoking uv or a real cloud API.
    monkeypatch.setattr(obstore_store, "S3Store", lambda **_kwargs: cloud._store)
    snapshot_dest = tmp_path / "fetched-snapshot"
    monkeypatch.setenv("MISEN_TRANSPORT_OPERATION", "snapshot")
    monkeypatch.setenv("MISEN_TRANSPORT_REF", "snapshot-key")
    monkeypatch.setenv("MISEN_TRANSPORT_DEST", str(snapshot_dest))
    exec(code, {})  # noqa: S102
    assert (snapshot_dest / "project.txt").read_text() == "snapshot"

    job_dest = tmp_path / "fetched-job.pkl"
    monkeypatch.setenv("MISEN_TRANSPORT_OPERATION", "job-file")
    monkeypatch.setenv("MISEN_TRANSPORT_REF", job_ref)
    monkeypatch.setenv("MISEN_TRANSPORT_DEST", str(job_dest))
    exec(code, {})  # noqa: S102
    assert job_dest.read_bytes() == b"payload"

    configured = _MemoryCloudWorkspace(
        backend="s3",
        bucket="configured-transport",
        config={"secret_access_key": "do-not-embed"},
        cache_dir=str(tmp_path / "configured-cache"),
    )
    with pytest.raises(ValueError, match="ambient worker environment"):
        configured.bootstrap_transport()

    class _NoTransportWorkspace:
        bootstrap_transport = Workspace.bootstrap_transport

    with pytest.raises(NotImplementedError):
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
        cpu_indices=[0, 1],
        accelerator_type="rocm",
        accelerator_indices=[],
    )
    # Path transport: one self-contained shell program, with no transport.
    assert env_overrides == {}
    assert argv[:2] == ["bash", "-c"]
    shell = argv[2]
    assert "misen.utils.bootstrap_env" not in shell
    assert "misen.utils.materialize_env" in shell
    assert str(snapshot.project_dir) in shell
    assert "/scratch/envs" in shell
    assert "--cpu-indices 0 1" in shell
    assert "--accelerator-type rocm" in shell
    assert "--accelerator-indices" in shell
    assert str(log_path) in shell


# ---------- bootstrap (worker side, real uv) ----------


def _run_materializer(
    snapshot: ProjectSnapshot,
    store_root: Path | str,
    payload_path: str,
    *,
    accelerator_type: str = "cuda",
    accelerator_indices: list[int] | None = None,
) -> None:
    """Invoke the path-only worker materializer directly."""
    accelerator_args = (
        []
        if accelerator_indices is None
        else ["--accelerator-type", accelerator_type, "--accelerator-indices", *map(str, accelerator_indices)]
    )
    tyro.cli(
        materialize_env.main,
        args=[
            "--project-dir",
            str(snapshot.project_dir),
            "--env-store-root",
            str(store_root),
            "--payload",
            payload_path,
            *accelerator_args,
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

    _run_materializer(snapshot, store_root, payload_ref, accelerator_type="rocm", accelerator_indices=[])
    path, argv, env = captured_exec[-1]
    assert path == argv[0] == _uv_bin()
    assert "misen.utils.execute" in argv
    assert argv[argv.index("--payload") + 1] == payload_ref  # disk refs are paths
    assert argv[argv.index("--accelerator-type") + 1] == "rocm"
    accelerator_flag = argv.index("--accelerator-indices")
    assert argv[accelerator_flag + 1].startswith("--")

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
    _run_materializer(snapshot, store_root, payload_ref)
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
    _run_materializer(snapshot, "env-store", payload_ref)  # relative to CWD
    _, _, env = captured_exec[-1]
    assert Path(env["VIRTUAL_ENV"]).is_relative_to((root / "env-store").absolute())


@pytest.mark.usefixtures("in_project")
def test_materializer_removes_corrupt_transported_snapshot(tmp_path: Path) -> None:
    """A failed integrity check leaves the cache ready for a clean refetch."""
    workspace = DiskWorkspace(directory=str(tmp_path / ".misen"))
    snapshot = ProjectSnapshot(workspace=workspace, prewarm=False)
    store_root = tmp_path / "env-store"
    corrupt = store_root / "snapshots" / snapshot.snapshot_key
    corrupt.mkdir(parents=True)
    (corrupt / "wrong.txt").write_text("wrong")
    payload = workspace.put_job_file(snapshot.submission_id, "JOB.pkl", b"payload")

    with pytest.raises(RuntimeError, match="expected"):
        tyro.cli(
            materialize_env.main,
            args=[
                "--project-dir",
                str(corrupt),
                "--snapshot-key",
                snapshot.snapshot_key,
                "--env-store-root",
                str(store_root),
                "--payload",
                payload,
                JOB_LOG_PATH_ARG,
                str(store_root / "job.log"),
            ],
        )
    assert not corrupt.exists()


def test_bash_transport_fetches_before_materialization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The single submitted shell fetches every ref before materialization."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname = "virtualproj"\nversion = "0.0.0"\nrequires-python = ">=3.11"\ndependencies = []\n'
    )
    monkeypatch.chdir(root)
    subprocess.run(["uv", "lock"], check=True, capture_output=True)  # noqa: S607

    workspace = DiskWorkspace(directory=str(tmp_path / ".misen"))
    snapshot = ProjectSnapshot(workspace=workspace, prewarm=False)
    payload_ref = workspace.put_job_file(snapshot.submission_id, "JOB.pkl", b"payload")
    env_ref = workspace.put_job_file(snapshot.submission_id, ".env", b"A=1\n")
    script = (
        'if [[ "$MISEN_TRANSPORT_OPERATION" == "snapshot" ]]; then\n'
        f'  cp -a -- {shlex.quote(str(snapshot.project_dir))} "$MISEN_TRANSPORT_DEST"\n'
        "else\n"
        '  cp -- "$MISEN_TRANSPORT_REF" "$MISEN_TRANSPORT_DEST"\n'
        "fi\n"
    )
    store_root = tmp_path / "env-store"
    echo_bin = shutil.which("echo")
    bash_bin = shutil.which("bash")
    assert echo_bin is not None
    assert bash_bin is not None
    bootstrap = worker_bootstrap_script(
        uv_bin=echo_bin,
        pixi_bin=None,
        requires_pixi=False,
        transport_script=script,
        misen_requirement="misen==0.0.9",
        python_version="3.11",
        store_root=store_root,
        project_dir=None,
        snapshot_key=snapshot.snapshot_key,
        payload=payload_ref,
        env_files=[env_ref],
        worker_args=[JOB_LOG_PATH_ARG, str(store_root / "job.log")],
    )
    assert '${env_file_refs[@]+"${!env_file_refs[@]}"}' in bootstrap
    result = subprocess.run([bash_bin, "-c", bootstrap], check=True, capture_output=True, text=True)  # noqa: S603

    transport_root = store_root / "job-files" / hashlib.sha256(script.encode()).hexdigest()
    payload_path = transport_root / hashlib.sha256(payload_ref.encode()).hexdigest()
    assert payload_path.read_bytes() == b"payload"
    env_file_path = transport_root / hashlib.sha256(env_ref.encode()).hexdigest()
    assert env_file_path.read_bytes() == b"A=1\n"
    assert oct(env_file_path.stat().st_mode & 0o777) == "0o600"
    assert (store_root / "snapshots" / snapshot.snapshot_key / "uv.lock").is_file()
    assert "misen.utils.materialize_env" in result.stdout
    assert "misen.utils.bootstrap_env" not in result.stdout


# ---------- executor plumbing ----------


@pytest.mark.usefixtures("in_project")
def test_executor_validation(tmp_path: Path) -> None:
    from misen.executors.slurm import SlurmExecutor

    cloud = _MemoryCloudWorkspace(backend="s3", bucket="val-test", cache_dir=str(tmp_path / "cache"))

    # SLURM prewarm needs an explicit (shared) env store.
    with pytest.raises(ValueError, match="env_store_dir"):
        SlurmExecutor(prewarm_envs=True)
    # Prewarm needs path-addressable job files.
    with pytest.raises(ValueError, match="worker-visible paths"):
        ProjectSnapshot(workspace=cloud, env_store_dir=str(tmp_path / "s"), prewarm=True)

    # snapshot=False is valid without a shared env-store path.
    assert SlurmExecutor(snapshot=False, prewarm_envs=True).snapshot is False


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
    snapshot = ProjectSnapshot(
        workspace=workspace,
        env_store_dir=executor.env_store_dir,
        prewarm=executor.prewarm_envs,
    )
    assert snapshot.prewarmed is None
    assert snapshot.env_store_dir == "/mnt/local/misen-envs"
