"""Dedicated worker device masks apply before Dask role initialization."""
# ruff: noqa: D103, S101

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import misen.utils.execute as execute_module


@pytest.mark.parametrize(("count", "expected"), [("0", ""), ("1", "GPU-first-uuid")])
def test_dedicated_gpu_controls_are_consumed_before_dask(
    monkeypatch: pytest.MonkeyPatch, count: str, expected: str
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-first-uuid,GPU-second-uuid")
    monkeypatch.setenv("MISEN_ACCELERATOR_COUNT", count)
    monkeypatch.setenv("MISEN_ACCELERATOR_TYPE", "cuda")
    for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "ZE_AFFINITY_MASK"):
        monkeypatch.setenv(name, "0,1")

    def dask_role() -> bool:
        assert os.environ["CUDA_VISIBLE_DEVICES"] == expected
        assert "MISEN_ACCELERATOR_COUNT" not in os.environ
        assert "MISEN_ACCELERATOR_TYPE" not in os.environ
        return True

    monkeypatch.setattr(execute_module, "run_role_from_env", dask_role)
    execute_module.execute(Path("unused.pkl"))


def test_dedicated_missing_scheduler_gpu_mask_fails_before_dask(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("MISEN_ACCELERATOR_COUNT", "1")
    monkeypatch.setenv("MISEN_ACCELERATOR_TYPE", "cuda")
    role = MagicMock(return_value=True)
    monkeypatch.setattr(execute_module, "run_role_from_env", role)
    with pytest.raises(ValueError, match="inherited scheduler device mask"):
        execute_module.execute(Path("unused.pkl"))
    role.assert_not_called()


def test_agent_consumed_controls_do_not_narrow_again(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    monkeypatch.delenv("MISEN_ACCELERATOR_COUNT", raising=False)
    monkeypatch.delenv("MISEN_ACCELERATOR_TYPE", raising=False)
    helper = MagicMock(side_effect=AssertionError("must not reapply controls"))
    monkeypatch.setattr(execute_module, "narrow_accelerator_environment", helper)
    monkeypatch.setattr(execute_module, "run_role_from_env", lambda: True)
    execute_module.execute(Path("unused.pkl"))
    helper.assert_not_called()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4"
