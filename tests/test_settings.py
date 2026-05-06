"""Tests for Settings config layering and Configurable auto-construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from misen.utils.settings import Configurable, ConfigurableMeta, Settings


class TestSettingsNoFiles:
    def test_empty_toml_data_when_no_files_exist(self, tmp_path: Path) -> None:
        settings = Settings(config_file=tmp_path / "nonexistent.toml")
        assert settings.toml_data == {}

    def test_empty_toml_data_with_default_resolution(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISEN_CONFIG", raising=False)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        monkeypatch.chdir(tmp_path)
        settings = Settings()
        assert settings.toml_data == {}


class TestSettingsExplicitOverride:
    def test_config_file_arg_uses_only_that_file(self, tmp_path: Path) -> None:
        override = tmp_path / "override.toml"
        override.write_text('[executor]\ntype = "slurm"\n', encoding="utf-8")
        settings = Settings(config_file=override)
        assert settings.toml_data == {"executor": {"type": "slurm"}}

    def test_misen_config_env_uses_only_that_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        override = tmp_path / "env_override.toml"
        override.write_text('[workspace]\ntype = "disk"\n', encoding="utf-8")
        monkeypatch.setenv("MISEN_CONFIG", str(override))

        xdg = tmp_path / "xdg"
        xdg.mkdir(parents=True)
        (xdg / "misen.toml").write_text('[workspace]\ntype = "should_be_ignored"\n', encoding="utf-8")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))

        settings = Settings()
        assert settings.toml_data == {"workspace": {"type": "disk"}}


class TestSettingsLayering:
    def test_xdg_only(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISEN_CONFIG", raising=False)
        xdg = tmp_path / "xdg"
        xdg.mkdir(parents=True)
        (xdg / "misen.toml").write_text('[executor]\ntype = "local"\nnum_cpus = 2\n', encoding="utf-8")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
        monkeypatch.chdir(tmp_path)

        settings = Settings()
        assert settings.toml_data == {"executor": {"type": "local", "num_cpus": 2}}

    def test_project_only(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISEN_CONFIG", raising=False)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty_xdg"))
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".misen.toml").write_text('[executor]\nnum_cpus = 8\n', encoding="utf-8")

        settings = Settings()
        assert settings.toml_data == {"executor": {"num_cpus": 8}}

    def test_project_section_replaces_xdg_section(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISEN_CONFIG", raising=False)
        xdg = tmp_path / "xdg"
        xdg.mkdir(parents=True)
        (xdg / "misen.toml").write_text('[executor]\ntype = "local"\nnum_cpus = 2\n', encoding="utf-8")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))

        project = tmp_path / "project"
        project.mkdir()
        (project / ".misen.toml").write_text('[executor]\nnum_cpus = 8\n', encoding="utf-8")
        monkeypatch.chdir(project)

        settings = Settings()
        # Shallow merge: project [executor] replaces XDG [executor] entirely
        assert settings.toml_data == {"executor": {"num_cpus": 8}}

    def test_disjoint_sections_from_xdg_and_project(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISEN_CONFIG", raising=False)
        xdg = tmp_path / "xdg"
        xdg.mkdir(parents=True)
        (xdg / "misen.toml").write_text('[executor]\ntype = "local"\n', encoding="utf-8")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))

        project = tmp_path / "project"
        project.mkdir()
        (project / ".misen.toml").write_text('[workspace]\ntype = "disk"\n', encoding="utf-8")
        monkeypatch.chdir(project)

        settings = Settings()
        # Disjoint sections are both preserved
        assert settings.toml_data == {"executor": {"type": "local"}, "workspace": {"type": "disk"}}

    def test_project_overrides_xdg(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISEN_CONFIG", raising=False)
        xdg = tmp_path / "xdg"
        xdg.mkdir(parents=True)
        (xdg / "misen.toml").write_text('[executor]\ntype = "local"\n', encoding="utf-8")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))

        project = tmp_path / "project"
        project.mkdir()
        (project / ".misen.toml").write_text('[executor]\ntype = "slurm"\n', encoding="utf-8")
        monkeypatch.chdir(project)

        settings = Settings()
        assert settings.toml_data == {"executor": {"type": "slurm"}}


class TestSettingsHash:
    def test_same_files_same_hash(self, tmp_path: Path) -> None:
        f = tmp_path / "a.toml"
        f.write_text("x = 1\n", encoding="utf-8")
        s1 = Settings(config_file=f)
        s2 = Settings(config_file=f)
        assert hash(s1) == hash(s2)

    def test_different_files_different_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.toml"
        f2 = tmp_path / "b.toml"
        f1.write_text("x = 1\n", encoding="utf-8")
        f2.write_text("x = 2\n", encoding="utf-8")
        assert hash(Settings(config_file=f1)) != hash(Settings(config_file=f2))

    def test_missing_file_hashable(self, tmp_path: Path) -> None:
        s = Settings(config_file=tmp_path / "missing.toml")
        assert isinstance(hash(s), int)


class TestConfigurable:
    @pytest.fixture(autouse=True)
    def _clear_singleton_cache(self) -> None:
        ConfigurableMeta._instances.clear()

    def test_workspace_auto_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISEN_CONFIG", raising=False)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        monkeypatch.chdir(tmp_path)

        from misen.workspace import Workspace
        from misen.workspaces.disk import DiskWorkspace

        ws = Workspace.auto(settings=Settings(config_file=tmp_path / "empty.toml"))
        assert isinstance(ws, DiskWorkspace)

    def test_executor_auto_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISEN_CONFIG", raising=False)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        monkeypatch.chdir(tmp_path)

        from misen.executor import Executor
        from misen.executors.local import LocalExecutor

        ex = Executor.auto(settings=Settings(config_file=tmp_path / "empty.toml"))
        assert isinstance(ex, LocalExecutor)

    def test_workspace_auto_from_toml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "cfg.toml"
        f.write_text('[workspace]\ntype = "disk"\ndirectory = "custom_dir"\n', encoding="utf-8")

        from misen.workspace import Workspace
        from misen.workspaces.disk import DiskWorkspace

        ws = Workspace.auto(settings=Settings(config_file=f))
        assert isinstance(ws, DiskWorkspace)
        assert ws.directory == str((tmp_path / "custom_dir").resolve())

    def test_executor_auto_from_toml(self, tmp_path: Path) -> None:
        f = tmp_path / "cfg.toml"
        f.write_text('[executor]\ntype = "local"\nnum_cpus = 4\n', encoding="utf-8")

        from misen.executor import Executor
        from misen.executors.local import LocalExecutor

        ex = Executor.auto(settings=Settings(config_file=f))
        assert isinstance(ex, LocalExecutor)
        assert ex.num_cpus == 4

    def test_resolve_type_with_alias(self) -> None:
        from misen.executor import Executor
        from misen.executors.in_process import InProcessExecutor

        assert Executor.resolve_type("in_process") is InProcessExecutor

    def test_resolve_type_with_module_class(self) -> None:
        from misen.executor import Executor
        from misen.executors.local import LocalExecutor

        assert Executor.resolve_type("misen.executors.local:LocalExecutor") is LocalExecutor

    def test_resolve_auto_literal(self, tmp_path: Path) -> None:
        from misen.workspace import Workspace
        from misen.workspaces.disk import DiskWorkspace

        ws = Workspace.resolve_auto("auto")
        assert isinstance(ws, DiskWorkspace)

    def test_resolve_auto_instance(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        from misen.workspaces.disk import DiskWorkspace

        instance = DiskWorkspace(directory=".test")
        assert DiskWorkspace.resolve_auto(instance) is instance

    def test_default_kwargs_without_type(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "cfg.toml"
        f.write_text('[workspace]\ndirectory = "from_defaults"\n', encoding="utf-8")

        from misen.workspace import Workspace
        from misen.workspaces.disk import DiskWorkspace

        ws = Workspace.auto(settings=Settings(config_file=f))
        assert isinstance(ws, DiskWorkspace)
        assert ws.directory == str((tmp_path / "from_defaults").resolve())
