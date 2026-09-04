"""Tests for Settings config layering and Configurable auto-construction."""

from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING

import pytest

import misen.utils.settings as settings_module
from misen.exceptions import ConfigError
from misen.utils.settings import Configurable, ConfigurableMeta, Settings

if TYPE_CHECKING:
    from pathlib import Path


class _BuggyConfigurable(Configurable):
    _config_key = "buggy"
    _config_default_type = f"{__name__}:_BuggyConfigurable"
    _config_aliases = {}

    def __post_init__(self) -> None:
        raise ValueError("plugin initialization bug")


class _ValidatedBase(Configurable):
    _config_key = "validated"
    _config_default_type = f"{__name__}:_ValidatedChild"
    _config_aliases = {}
    _config_validation_errors = (ValueError,)

    def __post_init__(self) -> None:
        raise ValueError("invalid inherited config")


class _ValidatedChild(_ValidatedBase):
    pass


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

    def test_deleted_cwd_raises_config_error_with_cause(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("MISEN_CONFIG", raising=False)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        deleted_cwd = tmp_path / "deleted"
        deleted_cwd.mkdir()
        monkeypatch.chdir(deleted_cwd)
        deleted_cwd.rmdir()
        try:
            with pytest.raises(ConfigError, match="current working directory") as raised:
                _ = Settings().toml_data
        finally:
            monkeypatch.undo()

        assert isinstance(raised.value.__cause__, FileNotFoundError)


class TestSettingsExplicitOverride:
    def test_config_file_arg_uses_only_that_file(self, tmp_path: Path) -> None:
        override = tmp_path / "override.toml"
        override.write_text('[executor]\ntype = "slurm"\n', encoding="utf-8")
        settings = Settings(config_file=override)
        assert settings.toml_data == {"executor": {"type": "slurm"}}

    def test_misen_config_env_uses_only_that_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        override = tmp_path / "env_override.toml"
        override.write_text('[workspace]\ntype = "disk"\n', encoding="utf-8")
        monkeypatch.setenv("MISEN_CONFIG", str(override))

        xdg = tmp_path / "xdg"
        xdg.mkdir(parents=True)
        (xdg / "misen.toml").write_text('[workspace]\ntype = "should_be_ignored"\n', encoding="utf-8")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))

        settings = Settings()
        assert settings.toml_data == {"workspace": {"type": "disk"}}

    def test_invalid_toml_raises_config_error_with_cause(self, tmp_path: Path) -> None:
        config = tmp_path / "invalid.toml"
        config.write_text("[workspace\n", encoding="utf-8")

        with pytest.raises(ConfigError, match="Invalid TOML") as exc_info:
            _ = Settings(config_file=config).toml_data

        assert isinstance(exc_info.value.__cause__, tomllib.TOMLDecodeError)


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
        (tmp_path / ".misen.toml").write_text("[executor]\nnum_cpus = 8\n", encoding="utf-8")

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
        (project / ".misen.toml").write_text("[executor]\nnum_cpus = 8\n", encoding="utf-8")
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
        f.write_text('[executor]\ntype = "local"\nnum_cpus = 1\n', encoding="utf-8")

        from misen.executor import Executor
        from misen.executors.local import LocalExecutor

        ex = Executor.auto(settings=Settings(config_file=f))
        assert isinstance(ex, LocalExecutor)
        assert ex.num_cpus == 1

    def test_skypilot_executor_alias_auto_from_toml(self, tmp_path: Path) -> None:
        config = tmp_path / "skypilot.toml"
        config.write_text(
            (
                "[executor]\n"
                'type = "skypilot"\n'
                'infra = ["aws", "gcp/us-central1"]\n'
                "use_spot = true\n"
                'name_prefix = "research"\n'
                'pool = "misen-dev"\n'
                "[executor.accelerators]\n"
                'cuda = ["A100", "L4"]\n'
                "[executor.accelerator_memory]\n"
                "A100 = 80\n"
                "L4 = 24\n"
            ),
            encoding="utf-8",
        )

        from misen.executor import Executor
        from misen.executors.skypilot import SkyPilotExecutor

        executor = Executor.auto(settings=Settings(config_file=config))

        assert isinstance(executor, SkyPilotExecutor)
        assert executor.infra == ["aws", "gcp/us-central1"]
        assert executor.use_spot is True
        assert executor.name_prefix == "research"
        assert executor.pool == "misen-dev"
        assert executor.accelerators == {"cuda": ["A100", "L4"]}
        assert executor.accelerator_memory == {"A100": 80, "L4": 24}

    def test_resolve_type_with_alias(self) -> None:
        from misen.executor import Executor
        from misen.executors.in_process import InProcessExecutor

        assert Executor.resolve_type("in_process") is InProcessExecutor

    def test_resolve_type_with_module_class(self) -> None:
        from misen.executor import Executor
        from misen.executors.local import LocalExecutor

        assert Executor.resolve_type("misen.executors.local:LocalExecutor") is LocalExecutor

    def test_resolve_type_rejects_invalid_reference_with_cause(self) -> None:
        from misen.executor import Executor

        with pytest.raises(ConfigError, match="expected 'module:Class'") as exc_info:
            Executor.resolve_type("not-a-reference")

        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_resolve_type_wraps_import_failure(self) -> None:
        from misen.executor import Executor

        with pytest.raises(ConfigError, match="Could not resolve") as exc_info:
            Executor.resolve_type("misen.does_not_exist:Executor")

        assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)

    def test_resolve_type_does_not_wrap_dependency_import_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from misen.executor import Executor

        def fail_import(_module_name: str) -> None:
            raise ModuleNotFoundError("No module named 'optional_dependency'", name="optional_dependency")

        monkeypatch.setattr(settings_module, "import_module", fail_import)

        with pytest.raises(ModuleNotFoundError, match="optional_dependency"):
            Executor.resolve_type("configured.executor:Executor")

    def test_resolve_type_wraps_missing_class_but_not_import_time_attribute_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from misen.executor import Executor

        with pytest.raises(ConfigError, match="Could not resolve"):
            Executor.resolve_type("misen.executors.local:MissingExecutor")

        def fail_import(_module_name: str) -> None:
            raise AttributeError("module initialization bug")

        monkeypatch.setattr(settings_module, "import_module", fail_import)
        with pytest.raises(AttributeError, match="module initialization bug"):
            Executor.resolve_type("configured.executor:Executor")

    def test_auto_wraps_invalid_constructor_settings(self, tmp_path: Path) -> None:
        config = tmp_path / "invalid-executor.toml"
        config.write_text('[executor]\ntype = "local"\nnum_cpus = 0\n', encoding="utf-8")

        from misen.executor import Executor

        with pytest.raises(ConfigError, match=r"Invalid \[executor\] settings") as exc_info:
            Executor.auto(settings=Settings(config_file=config))

        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_auto_preserves_unexpected_plugin_value_error(self, tmp_path: Path) -> None:
        settings = Settings(config_file=tmp_path / "missing.toml")

        with pytest.raises(ValueError, match="plugin initialization bug"):
            _BuggyConfigurable.auto(settings=settings)

    def test_auto_inherits_declared_validation_errors(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="invalid inherited config") as exc_info:
            _ValidatedBase.auto(Settings(config_file=tmp_path / "missing.toml"))

        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_auto_rejects_non_table_section(self, tmp_path: Path) -> None:
        config = tmp_path / "invalid-section.toml"
        config.write_text('executor = "local"\n', encoding="utf-8")

        from misen.executor import Executor

        with pytest.raises(ConfigError, match="expected a TOML table"):
            Executor.auto(settings=Settings(config_file=config))

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
