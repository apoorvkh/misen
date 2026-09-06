"""Fresh-process import, role dispatch, and canonical serialization boundaries."""
# ruff: noqa: D103, S101, S603

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

_MODULE = "misen.executors.skypilot"
_PROBE_MARKER = "MISEN_IMPORT_PROBE="
_SAFE_IMPORTS = r"""
import json
import os
import sys

observed = []
original_environment = dict(os.environ)

class RejectSkyPilotSDK:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sky" or fullname.startswith("sky."):
            observed.append("SDK import")
            raise AssertionError("SkyPilot SDK must not load during this operation")
        return None

sys.meta_path.insert(0, RejectSkyPilotSDK())

def reject_services(event, arguments):
    if event in {"subprocess.Popen", "os.system", "os.posix_spawn", "os.fork", "os.exec",
                 "socket.bind", "socket.connect"}:
        observed.append(event)
        raise AssertionError("Import/role validation attempted to start a process or contact a service")

sys.addaudithook(reject_services)

def verify_boundaries():
    assert not observed, observed
    assert dict(os.environ) == original_environment
    assert not any(name == "sky" or name.startswith("sky.") for name in sys.modules)
"""


def _probe(
    tmp_path: Path, source: str, *, args: tuple[str, ...] = (), stdin: str | None = None
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["XDG_STATE_HOME"] = str(tmp_path / "state")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-B", "-c", source, *args],
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
        cwd=tmp_path,
        env=env,
    )


def _result(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    lines = [line for line in result.stdout.splitlines() if line.startswith(_PROBE_MARKER)]
    assert lines, result.stderr
    return json.loads(lines[-1].removeprefix(_PROBE_MARKER))


def test_consolidated_module_import_is_lazy_and_side_effect_free(tmp_path: Path) -> None:
    result = _probe(
        tmp_path,
        _SAFE_IMPORTS
        + r"""
import misen.executors.skypilot as module
assert module.SkyPilotExecutor.__module__ == "misen.executors.skypilot"
assert module.SkyPilotCapacity.__module__ == "misen.executors.skypilot"
verify_boundaries()
print("MISEN_IMPORT_PROBE=" + json.dumps({"module": module.__name__, "safe": True}))
""",
    )
    assert result.returncode == 0, result.stderr
    assert _result(result) == {"module": _MODULE, "safe": True}
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize(
    ("arguments", "success"),
    [(("--worker-guard", "--help"), True), ((), False), (("--unknown-role",), False)],
)
def test_role_help_and_invalid_entry_do_not_load_sdk_or_start_services(
    tmp_path: Path, arguments: tuple[str, ...], *, success: bool
) -> None:
    result = _probe(
        tmp_path,
        _SAFE_IMPORTS
        + r"""
import runpy
sys.argv = ["misen.executors.skypilot", *sys.argv[1:]]
try:
    runpy.run_module("misen.executors.skypilot", run_name="__main__", alter_sys=True)
except SystemExit as exc:
    status = 0 if exc.code is None else exc.code
else:
    status = 0
verify_boundaries()
print("MISEN_IMPORT_PROBE=" + json.dumps({"safe": True, "status": status}))
raise SystemExit(status)
""",
        args=arguments,
    )
    assert (result.returncode == 0) is success, result.stderr
    assert _result(result) == {"safe": True, "status": result.returncode}
    if success:
        assert "usage" in (result.stdout + result.stderr).lower()
    assert not (tmp_path / "state").exists()


def test_capacity_and_graph_work_pickle_use_canonical_module_in_another_process(tmp_path: Path) -> None:
    produced = _probe(
        tmp_path,
        _SAFE_IMPORTS
        + r"""
import base64
import cloudpickle
from misen.executors.skypilot import GraphWork, SkyPilotCapacity, run_worker_agent
from misen.task_metadata import Resources, aggregate_resources

capacity = SkyPilotCapacity(cluster="cpu-cluster", cpus=2, memory=8)
work = GraphWork("job", [], "cpu", ["python", "-c", "pass"], {}, "logs/job.log",
                 aggregate_resources([Resources(cpus=1)]))
for value in (type(capacity), type(work), run_worker_agent):
    assert value.__module__ == "misen.executors.skypilot"
payload = cloudpickle.dumps((capacity, work, run_worker_agent))
verify_boundaries()
print("MISEN_IMPORT_PROBE=" + json.dumps({"payload": base64.b64encode(payload).decode("ascii")}))
""",
    )
    assert produced.returncode == 0, produced.stderr
    encoded = _result(produced)["payload"]
    payload = base64.b64decode(encoded)
    for legacy in (
        b"misen.executors.skypilot_capacity",
        b"misen.executors.skypilot_graph",
        b"misen.utils.graph_run",
        b"misen.utils.worker_agent",
    ):
        assert legacy not in payload
    restored = _probe(
        tmp_path,
        _SAFE_IMPORTS
        + r"""
import base64
import cloudpickle
capacity, work, agent = cloudpickle.loads(base64.b64decode(sys.stdin.read()))
assert type(capacity).__module__ == "misen.executors.skypilot"
assert type(work).__module__ == "misen.executors.skypilot"
assert agent.__module__ == "misen.executors.skypilot"
assert capacity.cluster == "cpu-cluster" and capacity.cpus == 2
assert work.job_id == "job" and work.argv == ["python", "-c", "pass"]
verify_boundaries()
print("MISEN_IMPORT_PROBE=" + json.dumps({"restored": True}))
""",
        stdin=encoded,
    )
    assert restored.returncode == 0, restored.stderr
    assert _result(restored) == {"restored": True}


def test_generic_execute_import_does_not_load_skypilot_executor(tmp_path: Path) -> None:
    result = _probe(
        tmp_path,
        _SAFE_IMPORTS
        + r"""
class RejectExecutor:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "misen.executors.skypilot" or fullname.startswith("misen.executors.skypilot."):
            raise AssertionError("Generic execution imported the optional SkyPilot executor")
        return None

sys.meta_path.insert(0, RejectExecutor())
import misen.utils.execute
assert "misen.executors.skypilot" not in sys.modules
verify_boundaries()
print("MISEN_IMPORT_PROBE=" + json.dumps({"generic": True}))
""",
    )
    assert result.returncode == 0, result.stderr
    assert _result(result) == {"generic": True}
