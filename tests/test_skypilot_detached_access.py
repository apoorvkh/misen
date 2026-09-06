"""Fail-closed detached coordinator credential and endpoint preflight."""
# ruff: noqa: D103, SLF001

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

import misen.executors.skypilot as graph_module
import misen.executors.skypilot as sky_module
from misen.exceptions import ConfigError, ExecutionError
from misen.executors.skypilot import GraphSkyPilotExecutor, RunManifest, SkyPilotCapacity


def _executor() -> GraphSkyPilotExecutor:
    return GraphSkyPilotExecutor(
        lifecycle="detached",
        manage_api_server=False,
        coordinator=SkyPilotCapacity(infra="aws", dedicated=True),
    )


def _workspace() -> MagicMock:
    workspace = MagicMock()
    workspace.supports_job_file_reads.return_value = True
    workspace.bootstrap_transport.return_value = "fetch-command"
    workspace.get_temp_dir.return_value = Path(".cache/workspace/tmp")
    return workspace


def _sky(monkeypatch: pytest.MonkeyPatch, health: Any, *, local: bool = False) -> SimpleNamespace:
    sky = SimpleNamespace(
        server=SimpleNamespace(common=SimpleNamespace(is_api_server_local=MagicMock(return_value=local))),
        api_info=MagicMock(return_value=health),
    )
    monkeypatch.setattr(sky_module, "_load_skypilot", MagicMock(return_value=sky))
    return sky


@pytest.mark.parametrize("enabled", [False, None, "true", 1])
def test_detached_requires_explicit_service_account_capability(monkeypatch: pytest.MonkeyPatch, enabled: Any) -> None:
    _sky(monkeypatch, {"status": "healthy", "api_version": "42", "service_account_token_enabled": enabled})
    workspace = _workspace()
    with pytest.raises(ConfigError, match="service accounts enabled"):
        _executor()._validate_submission(work_graph=None, pending_work_units=[], workspace=workspace)
    workspace.put_job_file.assert_not_called()


@pytest.mark.parametrize("version", [None, "", "41", "invalid", True])
def test_detached_requires_api_access_injection_version(monkeypatch: pytest.MonkeyPatch, version: Any) -> None:
    _sky(monkeypatch, {"status": "healthy", "api_version": version, "service_account_token_enabled": True})
    with pytest.raises(ConfigError, match="version >=42"):
        _executor()._validate_submission(work_graph=None, pending_work_units=[], workspace=_workspace())


@pytest.mark.parametrize("as_model", [False, True])
def test_detached_capability_accepts_sdk_model_or_json(monkeypatch: pytest.MonkeyPatch, *, as_model: bool) -> None:
    values = {"status": "healthy", "api_version": "42", "service_account_token_enabled": True}
    sky = _sky(monkeypatch, SimpleNamespace(**values) if as_model else values)
    _executor()._validate_submission(work_graph=None, pending_work_units=[], workspace=_workspace())
    sky.api_info.assert_called_once_with()


def test_detached_local_api_is_rejected_before_health_request(monkeypatch: pytest.MonkeyPatch) -> None:
    sky = _sky(monkeypatch, {}, local=True)
    with pytest.raises(ConfigError, match="stable remote"):
        _executor()._validate_submission(work_graph=None, pending_work_units=[], workspace=_workspace())
    sky.api_info.assert_not_called()


@pytest.mark.parametrize("health", [{"status": "needs_auth"}, {"status": "unhealthy"}])
def test_detached_preflight_requires_authenticated_health(
    monkeypatch: pytest.MonkeyPatch, health: dict[str, Any]
) -> None:
    _sky(monkeypatch, health)
    with pytest.raises(ConfigError, match="authenticated healthy"):
        _executor()._validate_submission(work_graph=None, pending_work_units=[], workspace=_workspace())


@pytest.mark.parametrize("missing", ["SKYPILOT_API_SERVER_ENDPOINT", "SKYPILOT_SERVICE_ACCOUNT_TOKEN"])
def test_remote_entry_missing_injection_cannot_initialize_backend(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    monkeypatch.setenv("SKYPILOT_API_SERVER_ENDPOINT", "https://sky.example.test")
    monkeypatch.setenv("SKYPILOT_SERVICE_ACCOUNT_TOKEN", "test-token")
    monkeypatch.delenv(missing)
    loader = MagicMock(side_effect=AssertionError("SDK must not load without injected credentials"))
    backend = MagicMock()
    monkeypatch.setattr(sky_module, "_load_skypilot", loader)
    monkeypatch.setattr(graph_module, "_SkyCapacityBackend", backend)
    workspace = _workspace()
    with pytest.raises(ExecutionError, match="missing its injected"):
        graph_module._run_remote(_executor(), RunManifest("run", "snapshot", [], []), workspace)
    loader.assert_not_called()
    backend.assert_not_called()
    workspace.lock.assert_not_called()


def test_remote_entry_local_endpoint_never_starts_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKYPILOT_API_SERVER_ENDPOINT", "http://127.0.0.1:46580")
    monkeypatch.setenv("SKYPILOT_SERVICE_ACCOUNT_TOKEN", "test-token")
    sky = _sky(monkeypatch, {}, local=True)
    backend = MagicMock()
    monkeypatch.setattr(graph_module, "_SkyCapacityBackend", backend)
    workspace = _workspace()
    with pytest.raises(ExecutionError, match="no local service"):
        graph_module._run_remote(_executor(), RunManifest("run", "snapshot", [], []), workspace)
    sky.api_info.assert_not_called()
    backend.assert_not_called()
    workspace.lock.assert_not_called()


def test_remote_entry_authentication_failure_never_claims_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKYPILOT_API_SERVER_ENDPOINT", "https://sky.example.test")
    monkeypatch.setenv("SKYPILOT_SERVICE_ACCOUNT_TOKEN", "test-token")
    sky = _sky(monkeypatch, {})
    sky.api_info.side_effect = RuntimeError("unauthorized")
    workspace = _workspace()
    backend = MagicMock()
    monkeypatch.setattr(graph_module, "_SkyCapacityBackend", backend)
    with pytest.raises(ExecutionError, match="could not authenticate"):
        graph_module._run_remote(_executor(), RunManifest("run", "snapshot", [], []), workspace)
    workspace.lock.assert_not_called()
    backend.assert_not_called()


def test_valid_remote_entry_checks_health_before_claim_and_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKYPILOT_API_SERVER_ENDPOINT", "https://sky.example.test")
    monkeypatch.setenv("SKYPILOT_SERVICE_ACCOUNT_TOKEN", "test-token")
    sky = _sky(monkeypatch, {"status": "healthy"})
    backend = MagicMock()
    coordinator = MagicMock()
    coordinator.errors = []
    monkeypatch.setattr(graph_module, "_SkyCapacityBackend", backend)
    monkeypatch.setattr(graph_module, "GraphCoordinator", MagicMock(return_value=coordinator))
    monkeypatch.setattr(graph_module, "_read", MagicMock(return_value=None))
    graph_module._run_remote(_executor(), RunManifest("run", "snapshot", [], []), _workspace())
    sky.api_info.assert_called_once_with()
    backend.assert_called_once()
    coordinator.run.assert_called_once_with()
