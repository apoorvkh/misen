"""External SDK validation stays lazy and rejects unsupported installations."""
# ruff: noqa: D103, S101, SLF001

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

import misen.executors.skypilot as skypilot_module
from misen.exceptions import ConfigError


@pytest.mark.parametrize("version", ["0.13", "0.13.0", "0.14.0.dev20260901", "1.0.0.dev20260905", "99.0.0"])
def test_supported_external_sdk_versions(monkeypatch: pytest.MonkeyPatch, version: str) -> None:
    sky = SimpleNamespace(__version__=version)
    monkeypatch.setattr(skypilot_module.importlib, "import_module", MagicMock(return_value=sky))
    assert skypilot_module._load_external_skypilot() is sky


@pytest.mark.parametrize("version", ["0.12.9", "0.13.0rc1", "0.13.0.dev20260801"])
def test_unsupported_external_sdk_versions_fail_actionably(monkeypatch: pytest.MonkeyPatch, version: str) -> None:
    monkeypatch.setattr(
        skypilot_module.importlib, "import_module", MagicMock(return_value=SimpleNamespace(__version__=version))
    )
    with pytest.raises(ConfigError, match=r"SkyPilot >=0.13; upgrade"):
        skypilot_module._load_external_skypilot()


@pytest.mark.parametrize("version", [None, "", 13, "unversioned"])
def test_unknown_external_sdk_versions_fail_actionably(monkeypatch: pytest.MonkeyPatch, version: Any) -> None:
    monkeypatch.setattr(
        skypilot_module.importlib, "import_module", MagicMock(return_value=SimpleNamespace(__version__=version))
    )
    with pytest.raises(ConfigError, match="installed SkyPilot SDK version; install"):
        skypilot_module._load_external_skypilot()


def test_missing_sdk_version_fails_actionably(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(skypilot_module.importlib, "import_module", MagicMock(return_value=SimpleNamespace()))
    with pytest.raises(ConfigError, match="Cannot determine"):
        skypilot_module._load_external_skypilot()
