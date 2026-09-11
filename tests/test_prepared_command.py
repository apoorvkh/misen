"""Persistent launch commands retain code identity and fresh per-job settings."""

# ruff: noqa: ANN001, ANN201, D103, S101, SLF001, S603
import functools
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import cloudpickle
import pytest

from misen.utils import materialize_env
from misen.utils.bootstrap_transport import worker_bootstrap_script
from misen.utils.snapshot import _MaterializedEnvs
from misen.workspaces.disk import DiskWorkspace


def test_prepared_command_skips_materializer_and_keeps_per_job_environment(monkeypatch, tmp_path):
    uv = tmp_path / "uv"
    uv.write_text(f"""#!{sys.executable}
import json,sys
print('uv 0.12.3' if sys.argv[1:] == ['--version'] else json.dumps(sys.argv[1:]))
""")
    uv.chmod(0o755)
    monkeypatch.delenv("MISEN_UV_BIN", raising=False)
    project = tmp_path / "project"
    project.mkdir()
    payload = tmp_path / "payload with spaces.pkl"
    output = tmp_path / "result.json"
    code = f"import json,os; from pathlib import Path; Path({str(output)!r}).write_text(json.dumps(dict(pid=os.getpid(),value=os.getenv('VALUE'),gpu=os.getenv('CUDA_VISIBLE_DEVICES'),threads=os.getenv('OMP_NUM_THREADS'))))"
    payload.write_bytes(
        cloudpickle.dumps(
            {"workspace": DiskWorkspace(directory=str(tmp_path / "workspace")), "fn": functools.partial(exec, code)}
        )
    )
    env_file = tmp_path / "job.env"
    env_file.write_text("VALUE=first\n")
    options = dict(
        uv_bin=str(uv),
        pixi_bin=None,
        requires_pixi=False,
        transport_script=None,
        misen_requirement="misen==test",
        python_version="3.13",
        store_root=tmp_path / "store",
        project_dir=project,
        snapshot_key=None,
        payload=str(payload),
        env_files=[str(env_file)],
        worker_args=["--job-log-path", str(tmp_path / "job.log")],
        reuse_env=True,
    )

    def run(**changes):
        return subprocess.run(
            ["bash", "-c", worker_bootstrap_script(**(options | changes))], check=True, capture_output=True, text=True
        )

    argv = json.loads(run().stdout)
    launcher = Path(argv[argv.index("--prepared-command") + 1])
    envs = _MaterializedEnvs(
        Path(sys.prefix), Path(sys.prefix), Path(__file__).resolve().parents[1] / "src", None, None
    )
    monkeypatch.setattr(materialize_env, "_materialize_envs", lambda *args, **kwargs: envs)
    monkeypatch.setenv("MISEN_PREPARE_ONLY", "1")
    materialize_env.main(
        project_dir=project, payload=payload, job_log_path=tmp_path / "job.log", prepared_command=launcher
    )
    assert not output.exists()
    monkeypatch.setattr(materialize_env, "_materialize_envs", lambda *a, **k: pytest.fail("warm materialization"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    run()
    first = json.loads(output.read_text())
    env_file.write_text("VALUE=second\n")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    run()
    second = json.loads(output.read_text())
    assert (first["value"], first["gpu"], first["threads"]) == ("first", "1", "2")
    assert (second["value"], second["gpu"], second["threads"]) == ("second", "0", "4")
    assert first["pid"] != second["pid"]
    run(env_files=[])
    assert json.loads(output.read_text())["value"] is None
    # A new snapshot, interpreter selection, or bootstrap version must miss.
    for changes in (
        {"project_dir": tmp_path / "new-snapshot"},
        {"python_version": "3.12"},
        {"misen_requirement": "misen==next"},
    ):
        assert "--prepared-command" in json.loads(run(**changes).stdout)
    monkeypatch.setenv("UV_PYTHON", "3.12")
    assert "--prepared-command" in json.loads(run().stdout)
