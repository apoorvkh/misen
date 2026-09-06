# ruff: noqa: ANN001, ANN201, ANN202, D100, D103, S101
import contextlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import misen.executors.skypilot as sky_mod
import misen.utils.execute as execute_mod
from misen.exceptions import ExecutionError
from misen.workspaces.disk import DiskWorkspace

_MAX_LOCK_KEY_LENGTH = 100


class _RecordingWorkspace:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict[str, int | str]]] = []
        self.events: list[str] = []
        self.fail_state: str | None = None
        self.files: dict[tuple[str, str], bytes] = {}
        self.lock_held = True

    def put_job_file(self, run_id: str, name: str, data: bytes) -> None:
        record = json.loads(data)
        if record["state"] == self.fail_state:
            msg = "private storage credential must not appear in an exception note"
            raise OSError(msg)
        self.records.append((run_id, name, record))
        self.events.append(record["state"])
        self.files[run_id, name] = data

    def read_job_file(self, run_id: str, name: str) -> bytes:
        try:
            return self.files[run_id, name]
        except KeyError:
            raise FileNotFoundError(name) from None

    def lock(self, namespace: str, key: str):
        assert namespace == "job"
        assert key.startswith("execution-")
        assert len(key) < _MAX_LOCK_KEY_LENGTH

        def context(*, timeout):
            assert timeout == sky_mod._CLAIM_LOCK_TIMEOUT  # noqa: SLF001 -- verify bounded internal locking
            return contextlib.nullcontext()

        return SimpleNamespace(
            context=context,
            is_locked=lambda: self.lock_held,
        )

    @contextlib.contextmanager
    def streaming_job_log(self, _path: Path):
        self.events.append("log-open")
        yield
        self.events.append("log-close")


@pytest.fixture(autouse=True)
def _clean_coordination_environment(monkeypatch):
    monkeypatch.delenv("MISEN_RUN_ID", raising=False)
    monkeypatch.delenv("MISEN_ATTEMPT_ID", raising=False)
    monkeypatch.delenv("MISEN_ENV_FILES_LOADED", raising=False)
    monkeypatch.setattr(execute_mod, "run_role_from_env", lambda: False)


@pytest.fixture
def execution(tmp_path, monkeypatch):
    payload = tmp_path / "payload.pkl"
    payload.write_bytes(b"test payload")
    workspace = _RecordingWorkspace()
    bundle = {"workspace": workspace, "fn": lambda: workspace.events.append("execute")}
    monkeypatch.setattr(execute_mod.cloudpickle, "loads", lambda _data: bundle)
    return payload, workspace, bundle


def _enable_attempt(monkeypatch):
    monkeypatch.setenv("MISEN_RUN_ID", "run_123")
    monkeypatch.setenv("MISEN_ATTEMPT_ID", "attempt-456")


def test_execution_without_attempt_environment_has_no_new_workspace_requirements(execution):
    payload, workspace, bundle = execution
    bundle["workspace"] = SimpleNamespace()

    execute_mod.execute(payload)

    assert workspace.events == ["execute"]
    assert workspace.records == []


def test_attempt_markers_bracket_payload_and_precede_log_finalization(execution, monkeypatch, tmp_path):
    payload, workspace, bundle = execution
    _enable_attempt(monkeypatch)

    def payload_fn():
        assert "MISEN_RUN_ID" not in os.environ
        assert "MISEN_ATTEMPT_ID" not in os.environ
        workspace.events.append("execute")

    bundle["fn"] = payload_fn
    execute_mod.execute(payload, job_log_path=tmp_path / "job.log")

    assert workspace.events == ["log-open", "claimed", "running", "execute", "done", "log-close"]
    assert workspace.records == [
        (
            "run_123",
            "attempt-attempt-456.execution.json",
            {"version": 1, "run_id": "run_123", "attempt_id": "attempt-456", "state": "claimed"},
        ),
        (
            "run_123",
            "attempt-attempt-456.started.json",
            {"version": 1, "run_id": "run_123", "attempt_id": "attempt-456", "state": "running"},
        ),
        (
            "run_123",
            "attempt-attempt-456.result.json",
            {"version": 1, "run_id": "run_123", "attempt_id": "attempt-456", "state": "done"},
        ),
    ]


@pytest.mark.parametrize("exception_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_payload_failure_is_recorded_without_exception_message_and_reraised(execution, monkeypatch, exception_type):
    payload, workspace, bundle = execution
    _enable_attempt(monkeypatch)
    secret = "secret-credential-value"  # noqa: S105 -- sentinel verifies that marker data excludes credentials
    error = exception_type(secret)

    def payload_fn():
        raise error

    bundle["fn"] = payload_fn
    with pytest.raises(exception_type) as caught:
        execute_mod.execute(payload)

    assert caught.value is error
    assert any(entry.name == "payload_fn" for entry in caught.traceback)
    assert workspace.events == ["claimed", "running", "failed"]
    result = workspace.records[-1][2]
    assert result["state"] == "failed"
    assert result["error_type"] == exception_type.__name__
    assert secret not in json.dumps(workspace.records)


def test_failed_marker_storage_error_does_not_replace_payload_error(execution, monkeypatch):
    payload, workspace, bundle = execution
    _enable_attempt(monkeypatch)
    workspace.fail_state = "failed"
    error = RuntimeError("original failure")

    def payload_fn():
        raise error

    bundle["fn"] = payload_fn
    with pytest.raises(RuntimeError) as caught:
        execute_mod.execute(payload)

    assert caught.value is error
    assert error.__notes__ == ["Additionally, publishing the failed attempt result did not succeed."]
    assert workspace.events == ["claimed", "running"]


@pytest.mark.parametrize("state", ["running", "done"])
def test_marker_storage_error_fails_closed_without_publishing_false_failure(execution, monkeypatch, state):
    payload, workspace, _bundle = execution
    _enable_attempt(monkeypatch)
    workspace.fail_state = state

    with pytest.raises(OSError, match="private storage credential"):
        execute_mod.execute(payload)

    assert workspace.events == (["claimed"] if state == "running" else ["claimed", "running", "execute"])


@pytest.mark.parametrize(
    ("run_id", "attempt_id"),
    [
        (None, "attempt"),
        ("run", None),
        ("", "attempt"),
        ("run", ""),
        ("../private-credential", "attempt"),
        ("run", "../attempt"),
        ("run/other", "attempt"),
        ("run", "attempt\nsecret"),
        ("a" * 129, "attempt"),
        ("run", "a" * 129),
        ("run", "non-ascii-\N{LATIN SMALL LETTER E WITH ACUTE}"),
    ],
)
def test_invalid_or_partial_identity_rejected_without_echoing_values(execution, monkeypatch, run_id, attempt_id):
    payload, workspace, _bundle = execution
    if run_id is not None:
        monkeypatch.setenv("MISEN_RUN_ID", run_id)
    if attempt_id is not None:
        monkeypatch.setenv("MISEN_ATTEMPT_ID", attempt_id)

    with pytest.raises(ValueError, match="must both contain valid bounded coordination identifiers"):
        execute_mod.execute(payload)

    assert workspace.events == []
    assert "MISEN_RUN_ID" not in os.environ
    assert "MISEN_ATTEMPT_ID" not in os.environ


def test_maximum_length_safe_identifiers_are_accepted(execution, monkeypatch):
    payload, workspace, _bundle = execution
    monkeypatch.setenv("MISEN_RUN_ID", "r" * 128)
    monkeypatch.setenv("MISEN_ATTEMPT_ID", "a" * 128)

    execute_mod.execute(payload)

    assert workspace.events == ["claimed", "running", "execute", "done"]


def test_completed_attempt_is_not_executed_again(execution, monkeypatch):
    payload, workspace, _bundle = execution
    _enable_attempt(monkeypatch)
    execute_mod.execute(payload)
    records = list(workspace.records)
    _enable_attempt(monkeypatch)

    execute_mod.execute(payload)

    assert workspace.events == ["claimed", "running", "execute", "done"]
    assert workspace.records == records


def test_existing_execution_claim_prevents_uncertain_replay(execution, monkeypatch):
    payload, workspace, _bundle = execution
    _enable_attempt(monkeypatch)
    workspace.files["run_123", "attempt-attempt-456.execution.json"] = b"prior claim"

    with pytest.raises(ExecutionError, match="without committed success"):
        execute_mod.execute(payload)

    assert workspace.events == []
    assert workspace.files == {("run_123", "attempt-attempt-456.execution.json"): b"prior claim"}


@pytest.mark.parametrize(
    "result_data",
    [
        b"not-json",
        b"x" * 4097,
        b"null",
        b'{"version": 1, "run_id": "run_123", "attempt_id": "attempt-456", "state": "failed"}',
        b'{"version": 1, "run_id": "other", "attempt_id": "attempt-456", "state": "done"}',
        b'{"version": true, "run_id": "run_123", "attempt_id": "attempt-456", "state": "done"}',
    ],
)
def test_nonmatching_or_failed_outcome_prevents_replay(execution, monkeypatch, result_data):
    payload, workspace, _bundle = execution
    _enable_attempt(monkeypatch)
    workspace.files["run_123", "attempt-attempt-456.result.json"] = result_data

    with pytest.raises(ExecutionError, match="unsuccessful or invalid outcome"):
        execute_mod.execute(payload)

    assert workspace.events == []
    assert workspace.files == {("run_123", "attempt-attempt-456.result.json"): result_data}


def test_lost_claim_lock_prevents_execution(execution, monkeypatch):
    payload, workspace, _bundle = execution
    _enable_attempt(monkeypatch)
    workspace.lock_held = False

    with pytest.raises(ExecutionError, match="Lost the execution claim lock"):
        execute_mod.execute(payload)

    assert workspace.events == []


def test_noncoordinator_dask_role_does_not_load_payload_or_publish_markers(execution, monkeypatch):
    payload, workspace, _bundle = execution
    _enable_attempt(monkeypatch)
    monkeypatch.setattr(execute_mod, "run_role_from_env", lambda: True)

    def unexpected_load(_data):
        pytest.fail("Dask runtime roles must not load the user payload")

    monkeypatch.setattr(execute_mod.cloudpickle, "loads", unexpected_load)
    execute_mod.execute(payload)

    assert workspace.records == []
    assert "MISEN_RUN_ID" not in os.environ
    assert "MISEN_ATTEMPT_ID" not in os.environ


def test_attempt_identity_survives_environment_reexec(execution, monkeypatch, tmp_path):
    payload, workspace, _bundle = execution
    _enable_attempt(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text("EXTRA_ENV_VALUE=example\n", encoding="utf-8")
    captured = {}

    def capture_exec(_executable, _argv, env):
        captured.update(env)

    monkeypatch.setattr(execute_mod.os, "execve", capture_exec)
    execute_mod.execute(payload, env_file=(env_file,))

    assert captured["MISEN_RUN_ID"] == "run_123"
    assert captured["MISEN_ATTEMPT_ID"] == "attempt-456"
    assert captured["MISEN_ENV_FILES_LOADED"] == "1"
    assert workspace.records == []


def test_disk_workspace_execution_claim_and_completion_survive_a_second_invocation(tmp_path, monkeypatch):
    workspace = DiskWorkspace(directory=str(tmp_path / "workspace"))
    marker = tmp_path / "invocations.txt"

    def payload_fn():
        assert "MISEN_RUN_ID" not in os.environ
        assert "MISEN_ATTEMPT_ID" not in os.environ
        with marker.open("a", encoding="utf-8") as output:
            output.write("executed\n")

    payload = tmp_path / "payload.pkl"
    payload.write_bytes(execute_mod.cloudpickle.dumps({"workspace": workspace, "fn": payload_fn}))
    _enable_attempt(monkeypatch)
    execute_mod.execute(payload)
    _enable_attempt(monkeypatch)
    execute_mod.execute(payload)

    assert marker.read_text(encoding="utf-8") == "executed\n"
    assert json.loads(workspace.read_job_file("run_123", "attempt-attempt-456.execution.json"))["state"] == "claimed"
    assert json.loads(workspace.read_job_file("run_123", "attempt-attempt-456.started.json"))["state"] == "running"
    assert json.loads(workspace.read_job_file("run_123", "attempt-attempt-456.result.json"))["state"] == "done"
