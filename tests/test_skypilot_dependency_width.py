"""Exact dependency-permitted SkyPilot profile concurrency."""
# ruff: noqa: D103, S101

from __future__ import annotations

import pytest

from misen.executors._skypilot.models import GraphWork, profile_dependency_widths
from misen.task_metadata import Resources


def _work(job_id: str, *dependencies: str, profile: str = "cpu") -> GraphWork:
    return GraphWork(
        job_id=job_id,
        dependencies=list(dependencies),
        profile=profile,
        argv=["python", "worker.py"],
        env={},
        log_path=f"job_logs/{job_id}.log",
        resources=Resources(cpus=1, memory=1),
    )


def test_empty_graph_has_no_profile_widths() -> None:
    assert profile_dependency_widths([]) == {}


def test_long_chain_needs_only_one_worker_for_its_profile() -> None:
    nodes = [_work("node-0")]
    nodes.extend(_work(f"node-{index}", f"node-{index - 1}") for index in range(1, 100))

    assert profile_dependency_widths(reversed(nodes)) == {"cpu": 1}


def test_one_hundred_independent_nodes_have_width_one_hundred() -> None:
    nodes = (_work(f"node-{index}") for index in range(100))

    assert profile_dependency_widths(nodes) == {"cpu": 100}


def test_wide_fanout_has_width_equal_to_its_one_hundred_children() -> None:
    nodes = [_work("root")]
    nodes.extend(_work(f"child-{index}", "root") for index in range(100))

    assert profile_dependency_widths(nodes) == {"cpu": 100}


def test_cross_profile_paths_preserve_same_profile_comparability() -> None:
    nodes = [
        _work("cpu-first"),
        _work("gpu-middle", "cpu-first", profile="gpu"),
        _work("cpu-last", "gpu-middle"),
        _work("cpu-independent"),
    ]

    assert profile_dependency_widths(nodes) == {"cpu": 2, "gpu": 1}


@pytest.mark.parametrize(
    ("nodes", "message"),
    [
        ([_work("child", "missing")], "Unknown logical dependency"),
        ([_work("child", "root", "root"), _work("root")], "Duplicate dependency"),
        ([_work("same"), _work("same")], "Duplicate logical job identity"),
        ([_work("left", "right"), _work("right", "left")], "dependency cycle"),
    ],
)
def test_malformed_graph_uses_ready_graph_validation(nodes: list[GraphWork], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        profile_dependency_widths(nodes)
