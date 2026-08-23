"""Contract tests for command-line exception presentation."""

# ruff: noqa: D103, EM101, PLR2004, S101, SLF001, TRY003, TRY301

from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any

import pytest

import misen.cli as misen_cli
import misen.utils.cli.experiment as experiment_cli_module
import misen.utils.cli.tui as tui_module
from misen import (
    CliUsageError,
    ConfigError,
    Experiment,
    JobFailedError,
    JobFailure,
    StorageError,
    StatusQueryError,
    SubmissionError,
)
from misen.utils.cli.errors import render_cli_error, run_cli
from misen.utils.graph import DependencyGraph


class _EmptyExperiment(Experiment):
    def tasks(self) -> set[Any]:
        return set()


class _AcceptedJob:
    job_id = "12345"
    log_path = Path("train.log")
    label = "train"

    def state(self) -> str:
        return "running"

    def wait(self, poll_s: float = 0.5) -> None:
        _ = poll_s


def _raise_config_error() -> None:
    raise ConfigError("invalid test configuration")


def _raise_chained_config_error() -> None:
    try:
        raise ValueError("invalid TOML")
    except ValueError as exc:
        raise ConfigError("could not load configuration") from exc


def test_expected_error_is_concise_by_default(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = run_cli(_raise_config_error)

    stderr = capsys.readouterr().err
    assert exit_code == 1
    assert stderr == "misen: configuration error: invalid test configuration\n"
    assert "Traceback" not in stderr


def test_debug_mode_preserves_exception_chain(capsys: pytest.CaptureFixture[str]) -> None:
    try:
        _raise_chained_config_error()
    except ConfigError as exc:
        exit_code = render_cli_error(exc, debug=True)

    stderr = capsys.readouterr().err
    assert exit_code == 1
    assert "Traceback" in stderr
    assert "ValueError: invalid TOML" in stderr
    assert "ConfigError: could not load configuration" in stderr


def test_unexpected_errors_are_not_hidden() -> None:
    def fail() -> None:
        raise ValueError("programmer error")

    with pytest.raises(ValueError, match="programmer error"):
        run_cli(fail)


def test_missing_textual_is_a_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def import_without_textual(name: str, *args: object, **kwargs: object) -> Any:
        if name == "textual.app":
            msg = "No module named 'textual'"
            raise ModuleNotFoundError(msg, name="textual")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_textual)

    with pytest.raises(ConfigError, match="Textual is required") as raised:
        tui_module._run_textual_task_monitor(
            named_tasks={},
            job_graph=None,  # type: ignore[arg-type]
            workspace=None,  # type: ignore[arg-type]
            poll_interval_s=0.2,
            state_poll_interval_s=2.0,
        )
    assert isinstance(raised.value.__cause__, ModuleNotFoundError)


def test_textual_transitive_import_errors_are_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def broken_textual_import(name: str, *args: object, **kwargs: object) -> Any:
        if name == "textual.app":
            msg = "No module named 'textual._broken'"
            raise ModuleNotFoundError(msg, name="textual._broken")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_textual_import)

    with pytest.raises(ModuleNotFoundError, match="textual._broken"):
        tui_module._run_textual_task_monitor(
            named_tasks={},
            job_graph=None,  # type: ignore[arg-type]
            workspace=None,  # type: ignore[arg-type]
            poll_interval_s=0.2,
            state_poll_interval_s=2.0,
        )


def test_tui_propagates_state_query_errors_after_teardown(monkeypatch: pytest.MonkeyPatch) -> None:
    from textual.app import App

    def fail_run(app: Any) -> None:
        app._monitor_error = StatusQueryError("controller unavailable")  # noqa: SLF001

    monkeypatch.setattr(App, "run", fail_run)

    with pytest.raises(StatusQueryError, match="controller unavailable"):
        tui_module._run_textual_task_monitor(
            named_tasks={},
            job_graph=DependencyGraph(),
            workspace=None,  # type: ignore[arg-type]
            poll_interval_s=0.2,
            state_poll_interval_s=2.0,
        )


def test_tui_renders_log_storage_errors_without_crashing(monkeypatch: pytest.MonkeyPatch) -> None:
    from textual.app import App

    messages: list[str] = []

    class Viewer:
        def clear(self) -> None:
            messages.clear()

        def write(self, value: Any) -> None:
            messages.append(value.plain)

    viewer = Viewer()

    def run_with_log_failure(app: Any) -> None:
        def fail_log_resolution(_entry: object) -> None:
            raise StorageError("log backend unavailable")

        monkeypatch.setattr(app, "query_one", lambda *_args, **_kwargs: viewer)
        monkeypatch.setattr(app, "_resolve_log_source", fail_log_resolution)
        app._cursor_entry = object()
        app._stream_log_chunk()

    monkeypatch.setattr(App, "run", run_with_log_failure)

    tui_module._run_textual_task_monitor(
        named_tasks={},
        job_graph=DependencyGraph(),
        workspace=None,  # type: ignore[arg-type]
        poll_interval_s=0.2,
        state_poll_interval_s=2.0,
    )

    assert messages == ["(log unavailable: StorageError)"]


def test_keyboard_interrupt_has_conventional_exit_code(capsys: pytest.CaptureFixture[str]) -> None:
    def interrupt() -> None:
        raise KeyboardInterrupt

    assert run_cli(interrupt) == 130
    assert capsys.readouterr().err == "misen: interrupted\n"


def test_cli_usage_error_has_conventional_exit_code(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = run_cli(lambda: (_ for _ in ()).throw(CliUsageError("unknown task 'nope'")))

    assert exit_code == 2
    assert capsys.readouterr().err == "misen: usage error: unknown task 'nope'\n"


def test_partial_submission_lists_already_accepted_jobs(capsys: pytest.CaptureFixture[str]) -> None:
    accepted_job = _AcceptedJob()

    def fail() -> None:
        raise SubmissionError("later dispatch failed", submitted_jobs=[accepted_job])

    assert run_cli(fail) == 1
    assert capsys.readouterr().err == (
        "misen: submission error: later dispatch failed\n"
        "  already submitted jobs:\n"
        "    - train (job_id=12345, log=train.log)\n"
    )


def test_experiment_command_input_error_is_concise(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = misen_cli.main(["experiment", "tests/test_experiment_cli.py:CliExperiment", "tree", "nope"])

    stderr = capsys.readouterr().err
    assert exit_code == 2
    assert "misen: usage error: Experiment has no task named 'nope'." in stderr
    assert "Traceback" not in stderr


def test_top_level_cli_uses_shared_error_boundary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_experiment(*, argv: list[str] | None = None) -> int:
        _ = argv
        raise ConfigError("bad CLI configuration")

    monkeypatch.setattr(misen_cli, "experiment", fail_experiment)

    exit_code = misen_cli.main(["experiment", "tests.fake:Experiment"])

    assert exit_code == 1
    assert capsys.readouterr().err == "misen: configuration error: bad CLI configuration\n"


def test_failed_job_produces_nonzero_clean_cli_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    failure = JobFailure(label="train", job_id="42", log_path="train.log", reason="Process exited with code 1.")

    def fail_experiment(*, argv: list[str] | None = None) -> int:
        _ = argv
        raise JobFailedError("1 failed job: train (log=train.log)", failures=[failure])

    monkeypatch.setattr(misen_cli, "experiment", fail_experiment)

    exit_code = misen_cli.main(["experiment", "tests.fake:Experiment"])

    stderr = capsys.readouterr().err
    assert exit_code == 1
    assert stderr == "misen: job failed: 1 failed job: train (log=train.log)\n"
    assert "Traceback" not in stderr


def test_experiment_cli_method_uses_shared_error_boundary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_experiment_cli(_experiment: object) -> None:
        raise ConfigError("bad direct CLI configuration")

    monkeypatch.setattr(experiment_cli_module, "experiment_cli", fail_experiment_cli)

    with pytest.raises(SystemExit) as raised:
        _EmptyExperiment.cli()

    assert raised.value.code == 1
    assert capsys.readouterr().err == "misen: configuration error: bad direct CLI configuration\n"
