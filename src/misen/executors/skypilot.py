"""Public SkyPilot executor facade.

Implementation details live in :mod:`misen.executors._skypilot`. Importing this
module remains side-effect free and does not import the optional SkyPilot SDK.
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from types import ModuleType
from typing import Any

from ._skypilot import api as _api
from ._skypilot import graph as _graph
from ._skypilot import jobs as _jobs
from ._skypilot import models as _models
from ._skypilot import worker as _worker

__all__ = ("SkyPilotCapacity", "SkyPilotExecutor", "SkyPilotJob", "SkyPilotTaskJob")

SkyPilotCapacity = _models.SkyPilotCapacity
SkyPilotExecutor = _graph.SkyPilotExecutor
SkyPilotJob = _jobs.SkyPilotJob
SkyPilotTaskJob = _graph.SkyPilotTaskJob

_broker_main = _api._broker_main  # noqa: SLF001 - facade owns child-role compatibility
_guard_main = _worker._guard_main  # noqa: SLF001 - facade owns child-role compatibility
_load_isolated_sdk = _api._load_isolated_sdk  # noqa: SLF001 - facade owns child-role compatibility

# Keep the historical consolidated namespace available for durable pickles and
# private integration tests while implementation globals move to focused files.
_INTERNAL_MODULES = (_models, _api, _worker, _jobs, _graph)
_ORIGINS: dict[str, list[ModuleType]] = {}
for _module in _INTERNAL_MODULES:
    for _name, _value in vars(_module).items():
        if _name.startswith("__"):
            continue
        _ORIGINS.setdefault(_name, []).append(_module)
        globals().setdefault(_name, _value)

# These objects have historically been serialized by reference to this module.
# Retaining the canonical path also lets old and new workers exchange payloads.
_CANONICAL_NAMES = (
    "SkyPilotCapacity",
    "GraphWork",
    "AgentWork",
    "RunManifest",
    "LogicalState",
    "RunState",
    "ReadyGraph",
    "read_run_state",
    "ManagedSkyPilotSession",
    "run_worker_agent",
    "SkyPilotJob",
    "SkyPilotTaskJob",
    "GraphCoordinator",
    "GraphSkyPilotExecutor",
    "SkyPilotExecutor",
)
for _name in _CANONICAL_NAMES:
    globals()[_name].__module__ = __name__


class _FacadeModule(types.ModuleType):
    """Mirror compatibility monkeypatches into the defining module."""

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        for module in _ORIGINS.get(name, ()):
            setattr(module, name, value)


sys.modules[__name__].__class__ = _FacadeModule


def _main() -> None:
    """Dispatch explicit child-only roles without loading SkyPilot for guards."""
    if len(sys.argv) > 1 and sys.argv[1] in ("--broker", "--server"):
        role = sys.argv[1]
        os.environ["_MISEN_SKYPILOT_PROCESS_ROLE"] = role
        if role == "--broker":
            del sys.argv[1]
            parser = argparse.ArgumentParser(description="Start an isolated SkyPilot broker.")
            parser.add_argument("directory")
            parser.add_argument("log_path")
            parser.parse_args()
        _broker_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "--worker-guard":
        del sys.argv[1]
        os.environ.pop("_MISEN_SKYPILOT_PROCESS_ROLE", None)
        _guard_main()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("role", choices=("--broker", "--server", "--worker-guard"))
        parser.parse_args()
        parser.error("an explicit child-process role is required")


if __name__ == "__main__":
    _main()
elif __name__ == "__mp_main__" and os.environ.get("_MISEN_SKYPILOT_PROCESS_ROLE") in ("--broker", "--server"):
    # Spawned SDK workers need the same compatibility paths as their parent.
    # Ordinary imports and worker guards must not load the SDK.
    _load_isolated_sdk()
