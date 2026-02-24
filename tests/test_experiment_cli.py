from types import SimpleNamespace

import tyro

from misen import Experiment, Task, task
from misen.executor import Executor
from misen.utils.experiment_cli import experiment_cli
from misen.workspace import Workspace


@task(id="cli_task", cache=False)
def cli_task(x: int) -> int:
    return x


class CliExperiment(Experiment):
    value: int = 1

    def tasks(self) -> dict[str, Task[int]]:
        return {"task": Task(cli_task, x=self.value)}


def test_experiment_cli_count_command(monkeypatch, capsys, tmp_path) -> None:
    settings_file = tmp_path / "misen.toml"
    first_args = SimpleNamespace(
        command="count",
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
    )
    second_args = SimpleNamespace(
        command="count",
        settings_file=settings_file,
        executor_type="auto",
        workspace_type="auto",
        experiment=CliExperiment(),
    )

    def fake_cli(*_args: object, **kwargs: object) -> object:
        if kwargs.get("return_unknown_args"):
            return first_args, []
        return second_args

    monkeypatch.setattr(tyro, "cli", fake_cli)
    monkeypatch.setattr(Executor, "auto", classmethod(lambda _cls, settings=None: object()))
    monkeypatch.setattr(Workspace, "auto", classmethod(lambda _cls, settings=None: object()))

    experiment_cli(CliExperiment)

    assert capsys.readouterr().out.strip() == "1"
