from misen.executor import Executor
from misen.executors.in_process import InProcessExecutor
from misen.workspace import Workspace
from misen.workspaces.disk import DiskWorkspace


def test_resolve_workspace_returns_explicit_workspace(tmp_path) -> None:
    workspace = DiskWorkspace(directory=str(tmp_path / ".misen"))
    assert Workspace.resolve_auto(workspace) is workspace


def test_resolve_workspace_auto_uses_workspace_auto(monkeypatch, tmp_path) -> None:
    expected = DiskWorkspace(directory=str(tmp_path / ".misen-auto"))
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls: expected))
    assert Workspace.resolve_auto("auto") is expected


def test_resolve_executor_returns_explicit_executor() -> None:
    executor = InProcessExecutor()
    assert Executor.resolve_auto(executor) is executor


def test_resolve_executor_auto_uses_executor_auto(monkeypatch) -> None:
    expected = InProcessExecutor()
    monkeypatch.setattr(Executor, "auto", classmethod(lambda _cls: expected))
    assert Executor.resolve_auto("auto") is expected
