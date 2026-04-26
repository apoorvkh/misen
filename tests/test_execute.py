# ruff: noqa: ANN001, D100, D103, S101
import base64
import contextlib
import os
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import cloudpickle
import pytest

import misen.utils.execute as execute_mod
from misen.utils.assigned_resources import AssignedResources
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


def test_execute_uses_encoded_assigned_resources_getter(tmp_path, monkeypatch) -> None:
    expected = AssignedResources(
        cpu_indices=[1, 2],
        gpu_indices=[0],
        memory=8,
        gpu_memory=16,
    )
    calls: list[dict[str, object]] = []
    payload_marker = tmp_path / "payload-ran.txt"

    def fake_apply_resource_binding(*, assigned_resources: object, gpu_runtime: str) -> None:
        calls.append({"assigned_resources": assigned_resources, "gpu_runtime": gpu_runtime})

    monkeypatch.setattr(execute_mod, "apply_resource_binding", fake_apply_resource_binding)
    monkeypatch.setenv("MISEN_TEST_GETTER", "1")

    def payload_fn(*, assigned_resources: AssignedResources | None) -> None:
        assert assigned_resources == expected
        payload_marker.write_text("ran", encoding="utf-8")

    def getter() -> AssignedResources:
        if os.environ.get("MISEN_TEST_GETTER") != "1":
            msg = "getter did not read worker environment"
            raise AssertionError(msg)
        return expected

    payload_path = tmp_path / "payload.pkl"
    payload_path.write_bytes(cloudpickle.dumps({"workspace": _stub_workspace(), "fn": payload_fn}))

    encoded_getter = base64.urlsafe_b64encode(cloudpickle.dumps(getter)).decode("ascii")

    execute_mod.execute(payload=payload_path, assigned_resources_getter=encoded_getter)

    assert calls == [{"assigned_resources": expected, "gpu_runtime": "cuda"}]
    assert payload_marker.read_text(encoding="utf-8") == "ran"


def test_execute_streams_explicit_job_log_path(tmp_path) -> None:
    marker_path = tmp_path / "streamed-path.txt"
    workspace = _RecordingWorkspace(marker_path)
    payload_path = tmp_path / "payload.pkl"
    payload_marker = tmp_path / "payload-ran.txt"
    log_path = tmp_path / "job.log"

    def payload_fn(*, assigned_resources: object) -> None:
        assert assigned_resources is None
        payload_marker.write_text("ran", encoding="utf-8")

    payload_path.write_bytes(cloudpickle.dumps({"workspace": workspace, "fn": payload_fn}))
    encoded_getter = base64.urlsafe_b64encode(cloudpickle.dumps(lambda: None)).decode("ascii")

    execute_mod.execute(payload=payload_path, assigned_resources_getter=encoded_getter, job_log_path=log_path)

    assert marker_path.read_text(encoding="utf-8") == str(log_path)
    assert payload_marker.read_text(encoding="utf-8") == "ran"


def test_execute_rejects_invalid_encoded_assigned_resources_getter(tmp_path) -> None:
    payload_path = tmp_path / "payload.pkl"
    payload_path.write_bytes(cloudpickle.dumps({"workspace": _stub_workspace(), "fn": lambda **_: None}))

    with pytest.raises(ValueError, match="expected URL-safe base64"):
        execute_mod.execute(payload=payload_path, assigned_resources_getter="not-base64@@")
