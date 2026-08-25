"""Tests for the env store and overlay venvs behind ``ProjectSnapshot``.

Protocol tests exercise ``_ensure_store_entry`` with synthetic build
functions (no uv); integration tests run real uv against a tiny generated
workspace project (one root package, one workspace member with a console
script, one non-workspace path dependency, one registry dependency),
prewarming environments through ``ProjectSnapshot``.

Reuse is asserted via markers and subprocess call counts, never via inode
counts: uv links with reflink on macOS, so shared inodes are Linux-only.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from misen.exceptions import SnapshotError
from misen.utils import snapshot as snapshot_mod
from misen.utils.locks import NFSLock
from misen.utils.snapshot import (
    ProjectSnapshot,
    _ensure_store_entry,
    _local_package_paths,
    _uv_cache_env,
)
from misen.workspaces.disk import DiskWorkspace

if TYPE_CHECKING:
    from collections.abc import Callable

_UV_BUILD_REQUIREMENT = 'requires = ["uv_build>=0.11.7,<0.13"]\nbuild-backend = "uv_build"\n'


def _write_project(root: Path) -> None:
    """Write a tiny uv workspace project: root + member + path dep."""
    member_dir = root / "packages" / "member"
    pathdep_dir = root.parent / "pathdep"
    for pkg_dir, name in ((root, "mainpkg"), (member_dir, "member"), (pathdep_dir, "pathdep")):
        (pkg_dir / "src" / name).mkdir(parents=True)

    (root / "pyproject.toml").write_text(
        "[project]\n"
        'name = "mainpkg"\n'
        'version = "0.1.0"\n'
        'requires-python = ">=3.11"\n'
        'dependencies = ["iniconfig", "member", "pathdep"]\n'
        "[tool.uv.sources]\n"
        "member = { workspace = true }\n"
        'pathdep = { path = "../pathdep" }\n'
        "[tool.uv.workspace]\n"
        'members = ["packages/member"]\n'
        f"[build-system]\n{_UV_BUILD_REQUIREMENT}"
    )
    (member_dir / "pyproject.toml").write_text(
        "[project]\n"
        'name = "member"\n'
        'version = "0.2.0"\n'
        'requires-python = ">=3.11"\n'
        "dependencies = []\n"
        "[project.scripts]\n"
        'member-cli = "member:main"\n'
        f"[build-system]\n{_UV_BUILD_REQUIREMENT}"
    )
    (pathdep_dir / "pyproject.toml").write_text(
        "[project]\n"
        'name = "pathdep"\n'
        'version = "0.3.0"\n'
        'requires-python = ">=3.11"\n'
        "dependencies = []\n"
        f"[build-system]\n{_UV_BUILD_REQUIREMENT}"
    )

    (root / "src" / "mainpkg" / "__init__.py").write_text("X = 1\n")
    (member_dir / "src" / "member" / "__init__.py").write_text('def main() -> None:\n    print("member-cli ok")\n')
    (pathdep_dir / "src" / "pathdep" / "__init__.py").write_text("Y = 2\n")

    subprocess.run(["uv", "lock"], cwd=root, check=True, capture_output=True)


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


def _env_build_calls(recorded: list[list[str]]) -> list[list[str]]:
    """Subprocess calls that create or populate an environment."""
    return [argv for argv in recorded if "sync" in argv or "venv" in argv or "install" in argv]


# ---------- protocol tests (synthetic builds, no uv) ----------


def _synthetic_build(calls: list[int], delay: float = 0.0) -> Callable[[Path], None]:
    def build(entry_dir: Path) -> None:
        calls.append(1)
        if delay:
            time.sleep(delay)
        entry_dir.mkdir(parents=True)
        (entry_dir / "sane").touch()

    return build


def _private_builds(store: Path, key: str = "KEY") -> list[Path]:
    root = store / f".{key}.builds"
    return list(root.iterdir()) if root.is_dir() else []


def test_ensure_store_entry_builds_then_reuses(tmp_path: Path) -> None:
    store = tmp_path / "store"
    calls: list[int] = []
    first = _ensure_store_entry(
        store=store, key="KEY", build=_synthetic_build(calls), sanity_path="sane", label="test env"
    )
    second = _ensure_store_entry(
        store=store, key="KEY", build=_synthetic_build(calls), sanity_path="sane", label="test env"
    )
    assert first == second == store / "KEY"
    assert calls == [1]
    assert (store / "KEY.complete").is_file()


def test_residue_without_marker_rebuilt(tmp_path: Path) -> None:
    store = tmp_path / "store"
    residue = store / "KEY" / "half-written"
    residue.parent.mkdir(parents=True)
    residue.touch()

    calls: list[int] = []
    entry = _ensure_store_entry(
        store=store, key="KEY", build=_synthetic_build(calls), sanity_path="sane", label="test env"
    )
    assert calls == [1]
    assert not (entry / "half-written").exists()
    assert (entry / "sane").is_file()
    assert (store / "KEY.complete").is_file()


def test_marker_without_entry_heals(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    (store / "KEY.complete").write_text("stale marker\n")

    calls: list[int] = []
    entry = _ensure_store_entry(
        store=store, key="KEY", build=_synthetic_build(calls), sanity_path="sane", label="test env"
    )
    assert calls == [1]
    assert (entry / "sane").is_file()
    assert (store / "KEY.complete").is_file()


def test_unmarked_link_healing_removes_private_build(tmp_path: Path) -> None:
    store = tmp_path / "store"
    orphaned_build = store / ".KEY.builds" / "ORPHAN"
    orphaned_build.mkdir(parents=True)
    (orphaned_build / "sane").touch()
    (store / "KEY").symlink_to(orphaned_build.relative_to(store), target_is_directory=True)

    calls: list[int] = []
    entry = _ensure_store_entry(
        store=store, key="KEY", build=_synthetic_build(calls), sanity_path="sane", label="test env"
    )

    assert calls == [1]
    assert not orphaned_build.exists()
    assert (entry / "sane").is_file()
    assert (store / "KEY.complete").is_file()


def test_next_publisher_cleans_hard_crashed_private_build(tmp_path: Path) -> None:
    store = tmp_path / "store"
    orphaned_build = store / ".KEY.builds" / "CRASHED"
    orphaned_build.mkdir(parents=True)
    (orphaned_build / "partial").touch()
    (store / "KEY").symlink_to(orphaned_build.relative_to(store), target_is_directory=True)

    entry = _ensure_store_entry(
        store=store,
        key="KEY",
        build=_synthetic_build([]),
        sanity_path="sane",
        label="test env",
    )

    assert not orphaned_build.exists()
    assert _private_builds(store) == [entry.resolve()]


def test_concurrent_same_key_single_build(tmp_path: Path) -> None:
    store = tmp_path / "store"
    calls: list[int] = []
    build = _synthetic_build(calls, delay=0.3)
    results: list[Path] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            results.append(
                _ensure_store_entry(store=store, key="KEY", build=build, sanity_path="sane", label="test env")
            )
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors
    assert calls == [1]
    assert results[0] == results[1]


def test_lost_lease_blocks_marker(tmp_path: Path) -> None:
    store = tmp_path / "store"
    lockfile = store / "KEY.lock"
    replacement = store / ".KEY.builds" / "NEWOWNER"

    def build(entry_dir: Path) -> None:
        entry_dir.mkdir(parents=True)
        (entry_dir / "sane").touch()
        # Simulate an extreme stall: another process breaks the (apparently
        # stale) lease mid-build. The builder must then refuse to publish.
        stale = time.time() - 3600
        os.utime(lockfile, (stale, stale))
        thief = NFSLock(lockfile, lifetime=1)
        thief.acquire(blocking=False)
        thief.release()
        replacement.mkdir()
        (replacement / "active").touch()

    with pytest.raises(SnapshotError, match="Lost the build lock"):
        _ensure_store_entry(store=store, key="KEY", build=build, sanity_path="sane", label="test env")
    assert not (store / "KEY.complete").exists()
    assert not (store / "KEY").exists()
    assert (replacement / "active").is_file()


def test_lost_lease_after_link_publication_blocks_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = tmp_path / "store"
    lockfile = store / "KEY.lock"
    real_publish_entry_link = snapshot_mod._publish_entry_link
    stole_once = False

    def publish_then_steal(store_dir: Path, entry_dir: Path, build_dir: Path) -> None:
        nonlocal stole_once
        real_publish_entry_link(store_dir, entry_dir, build_dir)
        if not stole_once:
            stole_once = True
            stale = time.time() - 3600
            os.utime(lockfile, (stale, stale))
            thief = NFSLock(lockfile, lifetime=1)
            thief.acquire(blocking=False)
            thief.release()

    monkeypatch.setattr(snapshot_mod, "_publish_entry_link", publish_then_steal)

    with pytest.raises(SnapshotError, match="Lost the build lock"):
        _ensure_store_entry(
            store=store,
            key="KEY",
            build=_synthetic_build([]),
            sanity_path="sane",
            label="test env",
        )

    assert not (store / "KEY.complete").exists()
    assert (store / "KEY").is_symlink()  # stale owner must not mutate the shared pointer
    assert not _private_builds(store)

    calls: list[int] = []
    entry = _ensure_store_entry(
        store=store, key="KEY", build=_synthetic_build(calls), sanity_path="sane", label="test env"
    )
    assert calls == [1]
    assert (entry / "sane").is_file()
    assert (store / "KEY.complete").is_file()


def test_marker_generation_rejects_pointer_swapped_during_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = tmp_path / "store"
    marker = store / "KEY.complete"
    lockfile = store / "KEY.lock"
    replacement = store / ".KEY.builds" / "NEWOWNER"
    real_replace = snapshot_mod.os.replace
    stole_once = False

    def replace_after_theft(src: str | Path, dst: str | Path) -> None:
        nonlocal stole_once
        if Path(dst).parent == store and Path(dst).name.startswith(".KEY.complete.") and not stole_once:
            stole_once = True
            stale = time.time() - 3600
            os.utime(lockfile, (stale, stale))
            thief = NFSLock(lockfile, lifetime=1)
            thief.acquire(blocking=False)
            replacement.mkdir()
            (replacement / "sane").touch()  # partial trees can contain the sanity path early
            replacement_link = store / ".KEY.replacement"
            replacement_link.symlink_to(replacement.relative_to(store), target_is_directory=True)
            real_replace(replacement_link, store / "KEY")
            thief.release()
        real_replace(src, dst)

    monkeypatch.setattr(snapshot_mod.os, "replace", replace_after_theft)

    with pytest.raises(SnapshotError, match="Build pointer changed"):
        _ensure_store_entry(
            store=store,
            key="KEY",
            build=_synthetic_build([]),
            sanity_path="sane",
            label="test env",
        )

    assert (store / "KEY").resolve() == replacement
    assert not marker.exists()  # the stale generation cannot bless NEWOWNER
    calls: list[int] = []
    entry = _ensure_store_entry(
        store=store,
        key="KEY",
        build=_synthetic_build(calls),
        sanity_path="sane",
        label="test env",
    )
    assert calls == [1]
    assert (entry / "sane").is_file()


def test_stale_shared_marker_cannot_invalidate_committed_generation(tmp_path: Path) -> None:
    store = tmp_path / "store"
    first = store / ".KEY.builds" / "FIRST"
    winner = store / ".KEY.builds" / "WINNER"
    for target in (first, winner):
        target.mkdir(parents=True)
        (target / "sane").touch()
        snapshot_mod._private_completion_marker(store, "KEY", target).write_text("complete\n")
    (store / "KEY").symlink_to(winner.relative_to(store), target_is_directory=True)
    (store / "KEY.complete").symlink_to(snapshot_mod._private_completion_marker(store, "KEY", first).relative_to(store))

    calls: list[int] = []
    entry = _ensure_store_entry(
        store=store,
        key="KEY",
        build=_synthetic_build(calls),
        sanity_path="sane",
        label="test env",
    )

    assert calls == []
    assert entry.resolve() == winner
    assert winner.is_dir()


def test_marker_publication_failure_cleans_link_and_private_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = tmp_path / "store"
    real_publish_marker = snapshot_mod._publish_marker

    def publish_then_fail(
        store_dir: Path,
        marker: Path,
        *,
        target: Path,
        before_commit: Callable[[], None] | None = None,
    ) -> None:
        real_publish_marker(store_dir, marker, target=target, before_commit=before_commit)
        msg = "marker publication failed"
        raise OSError(msg)

    monkeypatch.setattr(snapshot_mod, "_publish_marker", publish_then_fail)

    with pytest.raises(SnapshotError, match="Could not publish test env KEY") as raised:
        _ensure_store_entry(
            store=store,
            key="KEY",
            build=_synthetic_build([]),
            sanity_path="sane",
            label="test env",
        )

    assert isinstance(raised.value.__cause__, OSError)
    assert str(raised.value.__cause__) == "marker publication failed"
    assert not (store / "KEY.complete").exists()
    assert (store / "KEY").is_symlink()
    assert len(_private_builds(store)) == 1

    calls: list[int] = []
    entry = _ensure_store_entry(
        store=store,
        key="KEY",
        build=_synthetic_build(calls),
        sanity_path="sane",
        label="test env",
    )
    assert calls == []
    assert (entry / "sane").is_file()
    assert (store / "KEY.complete").is_file()


def test_syncfs_failure_blocks_and_cleans_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = tmp_path / "store"

    def fail_syncfs(_fd: int) -> None:
        msg = "syncfs failed"
        raise OSError(msg)

    monkeypatch.setattr(snapshot_mod.os, "syncfs", fail_syncfs, raising=False)

    with pytest.raises(SnapshotError, match="Could not publish test env KEY") as raised:
        _ensure_store_entry(
            store=store,
            key="KEY",
            build=_synthetic_build([]),
            sanity_path="sane",
            label="test env",
        )

    assert isinstance(raised.value.__cause__, OSError)
    assert str(raised.value.__cause__) == "syncfs failed"
    assert not (store / "KEY.complete").exists()
    assert (store / "KEY").is_symlink()
    assert not _private_builds(store)


def test_failed_build_removes_entry_and_leaves_no_marker(tmp_path: Path) -> None:
    store = tmp_path / "store"

    def failing_build(entry_dir: Path) -> None:
        entry_dir.mkdir(parents=True)
        msg = "boom"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="boom"):
        _ensure_store_entry(store=store, key="KEY", build=failing_build, sanity_path="sane", label="test env")
    assert not (store / "KEY.complete").exists()
    assert not (store / "KEY").exists()

    # The next attempt starts clean and succeeds.
    calls: list[int] = []
    _ensure_store_entry(store=store, key="KEY", build=_synthetic_build(calls), sanity_path="sane", label="test env")
    assert calls == [1]
    assert (store / "KEY.complete").is_file()


# ---------- uv.lock parsing ----------


def test_local_package_paths_classification(in_project: Path) -> None:
    paths = _local_package_paths(in_project / "uv.lock")
    assert (in_project).resolve() in paths  # root project (editable ".")
    assert (in_project / "packages" / "member").resolve() in paths
    assert (in_project.parent / "pathdep").resolve() in paths
    assert len(paths) == 3  # iniconfig (registry) excluded


def test_local_package_paths_rejects_unknown_source(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text(
        "version = 1\n"
        'requires-python = ">=3.11"\n'
        "[[package]]\n"
        'name = "mystery"\n'
        'version = "1.0"\n'
        "[package.source]\n"
        'teleport = "elsewhere"\n'
    )
    with pytest.raises(SnapshotError, match="Unrecognized source"):
        _local_package_paths(lock)


def test_local_package_paths_rejects_unknown_lock_version(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text("version = 2\n")
    with pytest.raises(SnapshotError, match=r"Unsupported uv\.lock version"):
        _local_package_paths(lock)


# ---------- cache-dir policy ----------


def test_cache_dir_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _uv_cache_env.cache_clear()
    try:
        # Explicit env var: always respected, never overridden.
        monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "mine"))
        assert _uv_cache_env(tmp_path / "store-a") == {}

        # Effective cache on the same filesystem: left alone (it's warm).
        monkeypatch.delenv("UV_CACHE_DIR")
        monkeypatch.setattr(snapshot_mod, "_same_filesystem", lambda _a, _b: True)
        assert _uv_cache_env(tmp_path / "store-b") == {}

        # Cross-filesystem effective cache: co-locate with the store.
        monkeypatch.setattr(snapshot_mod, "_same_filesystem", lambda _a, _b: False)
        store = tmp_path / "store-c"
        assert _uv_cache_env(store) == {"UV_CACHE_DIR": str(store / "uv-cache")}
    finally:
        _uv_cache_env.cache_clear()


# ---------- integration tests (real uv against the tiny project) ----------


@pytest.fixture(scope="module")
def module_workspace(tmp_path_factory: pytest.TempPathFactory) -> DiskWorkspace:
    return DiskWorkspace(directory=str(tmp_path_factory.mktemp("ws") / ".misen"))


@pytest.fixture(scope="module")
def module_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("env-store")


@pytest.fixture(scope="module")
def built_snapshot(project: Path, module_workspace: DiskWorkspace, module_store: Path) -> ProjectSnapshot:
    """One real prewarmed snapshot against module-scoped stores (read-only)."""
    cwd = Path.cwd()
    os.chdir(project)
    try:
        return ProjectSnapshot(workspace=module_workspace, env_store_dir=str(module_store), prewarm=True)
    finally:
        os.chdir(cwd)


@pytest.mark.usefixtures("in_project")
def test_envs_reused_across_snapshots(tmp_path: Path, counted_run: list[list[str]]) -> None:
    workspace = DiskWorkspace(directory=str(tmp_path / ".misen"))
    store = tmp_path / "env-store"
    first = ProjectSnapshot(workspace=workspace, env_store_dir=str(store), prewarm=True)
    builds_first = len(_env_build_calls(counted_run))
    assert builds_first > 0
    sync_calls = [argv for argv in counted_run if "sync" in argv]
    assert len(sync_calls) == 1
    assert "--frozen" in sync_calls[0]
    assert "--no-install-local" in sync_calls[0]
    assert "--compile-bytecode" in sync_calls[0]
    assert "pip" not in sync_calls[0]

    second = ProjectSnapshot(workspace=workspace, env_store_dir=str(store), prewarm=True)
    assert len(_env_build_calls(counted_run)) == builds_first  # reused, not rebuilt

    assert first.prewarmed is not None
    assert second.prewarmed is not None
    # Identical code state: same snapshot key, same deps env, same overlay.
    assert first.snapshot_key == second.snapshot_key
    assert first.prewarmed.deps_env_dir == second.prewarmed.deps_env_dir
    assert first.prewarmed.overlay_venv_dir == second.prewarmed.overlay_venv_dir
    assert first.prewarmed.deps_env_dir.is_relative_to(store / "python-envs")
    assert (store / "python-envs" / f"{first.prewarmed.deps_env_dir.name}.complete").is_file()


def test_code_change_rekeys_overlay_not_deps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "project"
    _write_project(root)
    monkeypatch.chdir(root)
    workspace = DiskWorkspace(directory=str(tmp_path / ".misen"))
    store = tmp_path / "env-store"

    first = ProjectSnapshot(workspace=workspace, env_store_dir=str(store), prewarm=True)
    member_source = root / "packages" / "member" / "src" / "member" / "__init__.py"
    member_source.write_text(member_source.read_text() + "\n# changed\n")
    second = ProjectSnapshot(workspace=workspace, env_store_dir=str(store), prewarm=True)

    assert first.prewarmed is not None
    assert second.prewarmed is not None
    assert first.snapshot_key != second.snapshot_key
    # Local-package edits change the overlay, not the remote-deps env.
    assert first.prewarmed.deps_env_dir == second.prewarmed.deps_env_dir
    assert first.prewarmed.overlay_venv_dir != second.prewarmed.overlay_venv_dir
    assert first.prewarmed.overlay_venv_dir.is_dir()  # both keys coexist in the store
    assert second.prewarmed.overlay_venv_dir.is_dir()


def test_overlay_contents(built_snapshot: ProjectSnapshot) -> None:
    assert built_snapshot.prewarmed is not None
    venv_dir = built_snapshot.prewarmed.overlay_venv_dir
    deps_env_dir = built_snapshot.prewarmed.deps_env_dir

    # Local packages import from the overlay; registry deps via the .pth
    # chain into the deps env; dist metadata resolves for local packages.
    check = (
        "import iniconfig, mainpkg, member, pathdep\n"
        "import importlib.metadata as m\n"
        "assert m.version('member') == '0.2.0'\n"
        "import sys\n"
        f"assert any(p.startswith({str(venv_dir)!r}) for p in sys.path if 'site-packages' in p)\n"
    )
    subprocess.run([str(venv_dir / "bin" / "python"), "-c", check], check=True)

    # Entry-point scripts of local packages get real launchers in the overlay.
    member_cli = venv_dir / "bin" / "member-cli"
    assert member_cli.is_file()
    assert os.access(member_cli, os.X_OK)
    result = subprocess.run([str(member_cli)], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "member-cli ok"

    # The deps env stays free of local packages.
    deps_sites = list(deps_env_dir.glob("lib/python*/site-packages"))
    assert deps_sites
    for name in ("mainpkg", "member", "pathdep"):
        assert not list(deps_sites[0].glob(f"{name}*"))
    assert list(deps_sites[0].glob("iniconfig*"))


def test_env_overrides_composition(built_snapshot: ProjectSnapshot, module_workspace: DiskWorkspace) -> None:
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

    _, argv, env_overrides, log_path = built_snapshot.prepare_job(
        _StubWorkUnit(),  # type: ignore[arg-type]
        module_workspace,
    )
    assert built_snapshot.prewarmed is not None
    assert env_overrides["VIRTUAL_ENV"] == str(built_snapshot.prewarmed.overlay_venv_dir)
    assert env_overrides["PATH"].split(os.pathsep)[0] == str(built_snapshot.prewarmed.deps_env_dir / "bin")
    assert env_overrides["PATH"].endswith(os.environ["PATH"])
    assert env_overrides["PYTHONPATH"].split(os.pathsep)[0] == str(built_snapshot.prewarmed.overlay_site_dir)
    # Direct dispatch: no bootstrap wrapper, straight to the worker module.
    assert "misen.utils.bootstrap_env" not in argv
    assert "misen.utils.execute" in argv
    assert "--accelerator-type" not in argv
    assert "--accelerator-indices" not in argv
    payload_path = Path(argv[argv.index("--payload") + 1])
    assert payload_path.read_bytes() == b"payload"
    assert argv[argv.index("--job-log-path") + 1] == str(log_path)


@pytest.mark.usefixtures("in_project")
def test_submission_artifacts_are_retained(tmp_path: Path) -> None:
    """No per-submission deletion: payloads must outlive scheduler requeues."""
    workspace = DiskWorkspace(directory=str(tmp_path / ".misen"))
    store = tmp_path / "env-store"
    snapshot = ProjectSnapshot(workspace=workspace, env_store_dir=str(store), prewarm=True)
    assert snapshot.prewarmed is not None
    deps_env_dir = snapshot.prewarmed.deps_env_dir
    snapshot_key = snapshot.snapshot_key
    ref = workspace.put_job_file(snapshot.submission_id, "probe.pkl", b"x")

    del snapshot  # a submit call ending deletes nothing
    assert Path(ref).exists()
    assert workspace.fetch_snapshot(snapshot_key).is_dir()
    assert (deps_env_dir.parent / f"{deps_env_dir.name}.complete").is_file()


def test_python_env_key_components(
    built_snapshot: ProjectSnapshot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    project_dir = built_snapshot.project_dir
    monkeypatch.delenv("UV_PYTHON", raising=False)
    base = snapshot_mod._python_env_key(project_dir)
    assert base == snapshot_mod._python_env_key(project_dir)  # deterministic

    monkeypatch.setenv("UV_PYTHON", "3.12")
    assert base != snapshot_mod._python_env_key(project_dir)  # interpreter selection
    monkeypatch.delenv("UV_PYTHON")

    # The frozen project controls the sync, so either metadata input changes
    # the key (checked on copies: published snapshots are immutable).
    copy_dir = tmp_path / "keyprobe"
    shutil.copytree(project_dir, copy_dir)
    (copy_dir / "uv.lock").write_bytes((copy_dir / "uv.lock").read_bytes() + b"\n# perturbed\n")
    assert base != snapshot_mod._python_env_key(copy_dir)

    metadata_dir = tmp_path / "metadataprobe"
    shutil.copytree(project_dir, metadata_dir)
    (metadata_dir / "pyproject.toml").write_bytes((metadata_dir / "pyproject.toml").read_bytes() + b"\n# perturbed\n")
    assert base != snapshot_mod._python_env_key(metadata_dir)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only store")
def test_store_ignores_foreign_entries(tmp_path: Path) -> None:
    """Lock/claim files, probes, and marker temps never match residue checks."""
    store = tmp_path / "store"
    calls: list[int] = []
    _ensure_store_entry(store=store, key="KEY", build=_synthetic_build(calls), sanity_path="sane", label="test env")
    # Foreign files that legitimately appear in a busy store.
    (store / "OTHER.lock").touch()
    (store / ".misen.freshprobe.left.tmp").touch()
    (store / ".KEY.complete.stray.tmp").touch()

    _ensure_store_entry(store=store, key="KEY", build=_synthetic_build(calls), sanity_path="sane", label="test env")
    assert calls == [1]  # still a pure reuse
