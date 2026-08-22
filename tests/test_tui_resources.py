"""TUI resource-summary formatting."""
# ruff: noqa: S101

from misen import Task, meta
from misen.utils.cli.tui import _format_resources


@meta(id="tui_resource_summary")
def _resource_task() -> None:
    return None


def test_multinode_resources_are_labeled_per_node() -> None:
    """Multi-node summaries expose both topology and per-node quantities."""
    resources = (
        Task(_resource_task)
        .with_resources(
            nodes=4,
            cpus=8,
            memory=32,
            accelerators=2,
            accelerator_type="cuda",
            accelerator_memory=80,
        )
        .resources
    )

    assert _format_resources(resources).plain == (
        "Resources: 4 nodes · 8 CPU/node · 32 GiB/node · 2x cuda/node (≥80 GiB/device) · 60m"
    )
