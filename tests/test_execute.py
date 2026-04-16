import base64
import cloudpickle
import os
import pytest

import misen.utils.execute as execute_mod
from misen.utils.assigned_resources import AssignedResources


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
    payload_path.write_bytes(cloudpickle.dumps(payload_fn))

    encoded_getter = base64.urlsafe_b64encode(cloudpickle.dumps(getter)).decode("ascii")

    execute_mod.execute(payload=payload_path, assigned_resources_getter=encoded_getter)

    assert calls == [{"assigned_resources": expected, "gpu_runtime": "cuda"}]
    assert payload_marker.read_text(encoding="utf-8") == "ran"


def test_execute_rejects_invalid_encoded_assigned_resources_getter(tmp_path) -> None:
    payload_path = tmp_path / "payload.pkl"
    payload_path.write_bytes(cloudpickle.dumps(lambda: None))

    with pytest.raises(ValueError, match="expected URL-safe base64"):
        execute_mod.execute(payload=payload_path, assigned_resources_getter="not-base64@@")
