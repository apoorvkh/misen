# ruff: noqa: ANN001, D100, D103, S101
import contextlib
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


def test_execute_forwards_assigned_resources_to_binding(tmp_path, monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    payload_marker = tmp_path / "payload-ran.txt"

    def fake_apply_resource_binding(
        *, cpu_indices: object, accelerator_type: object, accelerator_indices: object
    ) -> None:
        calls.append(
            {
                "cpu_indices": cpu_indices,
                "accelerator_type": accelerator_type,
                "accelerator_indices": accelerator_indices,
            }
        )

    monkeypatch.setattr(execute_mod, "apply_resource_binding", fake_apply_resource_binding)

    def payload_fn() -> None:
        payload_marker.write_text("ran", encoding="utf-8")

    payload_path = tmp_path / "payload.pkl"
    payload_path.write_bytes(cloudpickle.dumps({"workspace": _stub_workspace(), "fn": payload_fn}))
    loads = execute_mod.cloudpickle.loads

    def checked_loads(data: bytes) -> object:
        assert calls  # Binding must happen before payload imports/deserialization.
        return loads(data)

    monkeypatch.setattr(execute_mod.cloudpickle, "loads", checked_loads)

    execute_mod.execute(
        payload=payload_path,
        cpu_indices=[1, 2],
        accelerator_type="rocm",
        accelerator_indices=[],
    )

    assert calls == [{"cpu_indices": [1, 2], "accelerator_type": "rocm", "accelerator_indices": []}]
    assert payload_marker.read_text(encoding="utf-8") == "ran"


def test_execute_passes_none_indices_when_scheduler_isolates(tmp_path, monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    payload_path = tmp_path / "payload.pkl"
    payload_path.write_bytes(cloudpickle.dumps({"workspace": _stub_workspace(), "fn": lambda: None}))

    def fake_apply_resource_binding(
        *, cpu_indices: object, accelerator_type: object, accelerator_indices: object
    ) -> None:
        calls.append(
            {
                "cpu_indices": cpu_indices,
                "accelerator_type": accelerator_type,
                "accelerator_indices": accelerator_indices,
            }
        )

    monkeypatch.setattr(execute_mod, "apply_resource_binding", fake_apply_resource_binding)

    execute_mod.execute(payload=payload_path)

    assert calls == [{"cpu_indices": None, "accelerator_type": "cuda", "accelerator_indices": None}]


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
