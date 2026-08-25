# ruff: noqa: ANN001, D100, D103, S101, S603
import contextlib
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import cloudpickle

import misen.utils.execute as execute_mod
from misen.workspace import Workspace


def _stub_workspace() -> Workspace:
    """Return a minimal stub workspace with a no-op streaming_job_log."""
    return cast(
        "Workspace",
        SimpleNamespace(streaming_job_log=lambda _path: contextlib.nullcontext()),
    )


class _RecordingWorkspace:
    def __init__(self, marker_path: Path) -> None:
        self.marker_path = marker_path

    @contextlib.contextmanager
    def streaming_job_log(self, path: Path) -> Iterator[None]:
        self.marker_path.write_text(str(path), encoding="utf-8")
        yield


def test_execute_reexecs_after_loading_env_files(tmp_path) -> None:
    module_dir = tmp_path / "modules"
    module_dir.mkdir()
    (module_dir / "injected.py").write_text("VALUE = 'loaded'\n")
    first = tmp_path / ".env"
    second = tmp_path / ".env.local"
    first.write_text(
        f"PYTHONPATH={module_dir}\n"
        "FROM_FILES=first\n"
        "INHERITED=from-file\n"
        "OMP_NUM_THREADS=from-file\n"
        "CUDA_VISIBLE_DEVICES=from-file\n"
    )
    second.write_text("FROM_FILES=second\n")
    marker = tmp_path / "environment.txt"

    def payload_fn() -> None:
        injected = __import__("injected")
        marker.write_text(
            f"{injected.VALUE}:{os.environ['FROM_FILES']}:{os.environ['INHERITED']}:"
            f"{os.environ['OMP_NUM_THREADS']}:{os.environ['CUDA_VISIBLE_DEVICES']}"
        )

    payload = tmp_path / "payload.pkl"
    payload.write_bytes(cloudpickle.dumps({"workspace": _stub_workspace(), "fn": payload_fn}))
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("MISEN_ENV_FILES_LOADED", None)
    env["INHERITED"] = "worker"
    env["OMP_NUM_THREADS"] = "2"
    env["CUDA_VISIBLE_DEVICES"] = "1"

    subprocess.run(
        [sys.executable, "-m", "misen.utils.execute", "--payload", str(payload), "--env-file", str(first), str(second)],
        env=env,
        check=True,
    )

    assert marker.read_text() == "loaded:second:worker:2:1"


def test_execute_places_active_venv_first_on_path(tmp_path, monkeypatch) -> None:
    venv = tmp_path / "overlay"
    monkeypatch.setenv("VIRTUAL_ENV", str(venv))
    monkeypatch.setenv("PATH", "/conda/bin:/deps/bin:/system/bin")
    marker = tmp_path / "path.txt"

    def payload_fn() -> None:
        marker.write_text(os.environ["PATH"])

    payload = tmp_path / "payload.pkl"
    payload.write_bytes(cloudpickle.dumps({"workspace": _stub_workspace(), "fn": payload_fn}))

    execute_mod.execute(payload=payload)

    assert marker.read_text().split(os.pathsep) == [
        str(venv / "bin"),
        "/conda/bin",
        "/deps/bin",
        "/system/bin",
    ]


def test_execute_streams_explicit_job_log_path(tmp_path) -> None:
    marker_path = tmp_path / "streamed-path.txt"
    workspace = _RecordingWorkspace(marker_path)
    payload_path = tmp_path / "payload.pkl"
    payload_marker = tmp_path / "payload-ran.txt"
    log_path = tmp_path / "job.log"

    def payload_fn() -> None:
        payload_marker.write_text("ran", encoding="utf-8")

    payload_path.write_bytes(cloudpickle.dumps({"workspace": workspace, "fn": payload_fn}))

    execute_mod.execute(payload=payload_path, job_log_path=log_path)

    assert marker_path.read_text(encoding="utf-8") == str(log_path)
    assert payload_marker.read_text(encoding="utf-8") == "ran"
