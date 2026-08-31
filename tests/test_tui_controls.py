"""Keyboard controls for the Textual job monitor."""

# ruff: noqa: D103, S101, SLF001

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
from textual.app import App

import misen.utils.cli.tui as tui_module
from misen import Task, meta
from misen.exceptions import ExecutionError
from misen.executor import Job, JobState
from misen.utils.graph import DependencyGraph
from misen.utils.work_unit import WorkUnit
from misen.workspace import Workspace

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence


@meta(id="tui_control_test_task", cache=False)
def _control_task(value: int) -> int:
    return value


class _CancelableJob(Job):
    """Stable-state job that records cancellation requests."""

    def __init__(
        self,
        value: int,
        state: JobState,
        *,
        work_unit: WorkUnit | None = None,
        cancel_error: Exception | None = None,
    ) -> None:
        super().__init__(
            work_unit=work_unit or WorkUnit(root=Task(_control_task, value=value), dependencies=set()),
            job_id=f"job-{value}",
        )
        self.current_state = state
        self.cancel_error = cancel_error
        self.cancel_calls = 0
        self.state_calls = 0

    def state(self) -> JobState:
        self.state_calls += 1
        return self.current_state

    def cancel(self) -> None:
        self.cancel_calls += 1
        if self.cancel_error is not None:
            raise self.cancel_error


def _jobs_with_states(states: Sequence[JobState]) -> tuple[list[_CancelableJob], DependencyGraph[Job]]:
    jobs = [_CancelableJob(value, state) for value, state in enumerate(states)]
    graph: DependencyGraph[Job] = DependencyGraph()
    for job in jobs:
        graph.add_node(job)
    return jobs, graph


def _cancellation_cascade(
    *,
    selected_error: Exception | None = None,
) -> tuple[list[_CancelableJob], DependencyGraph[Job]]:
    selected = _CancelableJob(0, "running", cancel_error=selected_error)
    prerequisite = _CancelableJob(1, "running")
    dependent = _CancelableJob(2, "pending")
    transitive_dependent = _CancelableJob(3, "running")
    unrelated = _CancelableJob(4, "pending")
    jobs = [selected, prerequisite, dependent, transitive_dependent, unrelated]
    graph: DependencyGraph[Job] = DependencyGraph()
    indices = [graph.add_node(job) for job in jobs]
    selected_index, prerequisite_index, dependent_index, transitive_index, unrelated_index = indices
    # Edges point from a dependent to its dependency.
    graph.add_edge(selected_index, prerequisite_index)
    graph.add_edge(dependent_index, selected_index)
    graph.add_edge(transitive_index, dependent_index)
    # Sharing a prerequisite does not make this job a dependent of the
    # selected branch.
    graph.add_edge(unrelated_index, prerequisite_index)
    return jobs, graph


def _run_monitor(
    monkeypatch: pytest.MonkeyPatch,
    jobs: Sequence[_CancelableJob],
    graph: DependencyGraph[Job],
    drive: Callable[[Any, Any], Awaitable[None]],
) -> dict[Job, JobState]:
    """Run the function-local monitor app under Textual's test pilot."""

    async def run_test(app: Any) -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            await drive(app, pilot)

    def run(app: Any, *_args: object, **_kwargs: object) -> None:
        asyncio.run(run_test(app))

    monkeypatch.setattr(App, "run", run)
    workspace = MagicMock(spec=Workspace)
    return tui_module._run_textual_task_monitor(
        named_tasks={f"task-{index}": job.root for index, job in enumerate(jobs)},
        job_graph=graph,
        workspace=workspace,
        # Keep periodic ticks out of these tests. The initial mount poll is
        # enough to populate the state cache that the controls must use.
        poll_interval_s=3600,
        state_poll_interval_s=3600,
    )


def test_c_binding_is_visible_async_and_cancels_only_selected_active_job(monkeypatch: pytest.MonkeyPatch) -> None:
    jobs, graph = _jobs_with_states(["done", "pending", "running"])
    exited_after_cancel: list[bool] = []

    async def drive(app: Any, pilot: Any) -> None:
        binding = next(binding for binding in app.BINDINGS if binding.key == "c")
        assert binding.show is True
        assert inspect.iscoroutinefunction(getattr(app, f"action_{binding.action}"))
        assert app.check_action(binding.action, ()) is False

        # The first job is done. Move the highlight to the pending second job
        # and verify that cancellation becomes available for that job alone.
        await pilot.press("down")
        await pilot.pause()
        assert app.check_action(binding.action, ()) is True
        await pilot.press("c")
        await pilot.pause()
        exited_after_cancel.append(app._exit)
        app.exit()

    _run_monitor(monkeypatch, jobs, graph, drive)

    assert exited_after_cancel == [False]
    assert [job.cancel_calls for job in jobs] == [0, 1, 0]
    # Cancellation must use the latest cached states rather than issuing an
    # extra backend status query from the key handler.
    assert [job.state_calls for job in jobs] == [1, 1, 1]


def test_selected_job_cancellation_cascades_to_transitive_dependents(monkeypatch: pytest.MonkeyPatch) -> None:
    jobs, graph = _cancellation_cascade()

    async def drive(app: Any, pilot: Any) -> None:
        await pilot.press("c")
        await pilot.pause()
        assert app._exit is False
        app.exit()

    _run_monitor(monkeypatch, jobs, graph, drive)

    selected, prerequisite, dependent, transitive_dependent, unrelated = jobs
    assert selected.cancel_calls == dependent.cancel_calls == transitive_dependent.cancel_calls == 1
    assert prerequisite.cancel_calls == unrelated.cancel_calls == 0


def test_cascade_cancellation_attempts_every_target_before_propagating_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs, graph = _cancellation_cascade(selected_error=ExecutionError("selected cancel failed"))

    async def drive(_app: Any, pilot: Any) -> None:
        await pilot.press("c")
        await pilot.pause()

    with pytest.raises(ExecutionError, match="selected cancel failed"):
        _run_monitor(monkeypatch, jobs, graph, drive)

    selected, prerequisite, dependent, transitive_dependent, unrelated = jobs
    assert selected.cancel_calls == dependent.cancel_calls == transitive_dependent.cancel_calls == 1
    assert prerequisite.cancel_calls == unrelated.cancel_calls == 0


def test_job_mode_navigation_cancels_only_highlighted_work_unit(monkeypatch: pytest.MonkeyPatch) -> None:
    jobs, graph = _jobs_with_states(["done", "pending", "running"])

    async def drive(app: Any, pilot: Any) -> None:
        binding = next(binding for binding in app.BINDINGS if binding.key == "c")
        await pilot.press("tab")
        await pilot.pause()
        assert app._mode == "job"
        assert app.check_action(binding.action, ()) is False

        await pilot.press("down")
        await pilot.pause()
        assert app.check_action(binding.action, ()) is True
        await pilot.press("c")
        await pilot.pause()
        assert app._exit is False
        app.exit()

    _run_monitor(monkeypatch, jobs, graph, drive)

    assert [job.cancel_calls for job in jobs] == [0, 1, 0]


def test_inner_task_selection_cancels_containing_work_unit_and_tab_preserves_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inner = Task(_control_task, value=10)
    root = Task(_control_task, value=inner.T)
    work_unit = WorkUnit(root=root, dependencies=set())
    containing_job = _CancelableJob(0, "running", work_unit=work_unit)
    sibling_job = _CancelableJob(1, "running")
    jobs = [containing_job, sibling_job]
    graph: DependencyGraph[Job] = DependencyGraph()
    graph.add_node(containing_job)
    graph.add_node(sibling_job)

    async def drive(app: Any, pilot: Any) -> None:
        # The containing work unit's root is initially highlighted; its inner
        # non-cacheable task is the next task-mode cursor stop.
        await pilot.press("down")
        await pilot.pause()
        assert app._cursor_entry.task is inner
        assert app._cursor_entry.work_unit is work_unit
        await pilot.press("c")
        await pilot.pause()

        # Switching to job mode snaps the cursor to this work unit's canonical
        # row without changing which backend job the selection represents.
        await pilot.press("tab")
        await pilot.pause()
        assert app._mode == "job"
        assert app._cursor_entry.work_unit is work_unit
        assert app._selected_active_job() is containing_job
        app.exit()

    _run_monitor(monkeypatch, jobs, graph, drive)

    assert containing_job.cancel_calls == 1
    assert sibling_job.cancel_calls == 0


@pytest.mark.parametrize("selected_state", ["done", "failed", "unknown"])
def test_c_binding_is_disabled_for_inactive_selected_job(
    monkeypatch: pytest.MonkeyPatch,
    selected_state: JobState,
) -> None:
    jobs, graph = _jobs_with_states([selected_state])

    async def drive(app: Any, pilot: Any) -> None:
        binding = next(binding for binding in app.BINDINGS if binding.key == "c")
        assert app.check_action(binding.action, ()) is False
        await pilot.press("c")
        await pilot.pause()
        assert app._exit is False
        app.exit()

    _run_monitor(monkeypatch, jobs, graph, drive)

    assert jobs[0].cancel_calls == 0
    assert jobs[0].state_calls == 1


def test_selected_job_cancellation_error_is_propagated(monkeypatch: pytest.MonkeyPatch) -> None:
    first = _CancelableJob(0, "running", cancel_error=ExecutionError("cancel failed"))
    sibling = _CancelableJob(1, "pending")
    graph: DependencyGraph[Job] = DependencyGraph()
    graph.add_node(first)
    graph.add_node(sibling)

    async def drive(_app: Any, pilot: Any) -> None:
        await pilot.press("c")
        await pilot.pause()

    with pytest.raises(ExecutionError, match="cancel failed"):
        _run_monitor(monkeypatch, [first, sibling], graph, drive)

    assert first.cancel_calls == 1
    assert sibling.cancel_calls == 0


def test_ctrl_c_propagates_keyboard_interrupt_without_cancelling_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    jobs, graph = _jobs_with_states(["pending", "running", "done", "failed", "unknown"])

    async def drive(_app: Any, pilot: Any) -> None:
        await pilot.press("ctrl+c")
        await pilot.pause()

    with pytest.raises(KeyboardInterrupt):
        _run_monitor(monkeypatch, jobs, graph, drive)

    assert [job.cancel_calls for job in jobs] == [0, 0, 0, 0, 0]
    assert [job.state_calls for job in jobs] == [1, 1, 1, 1, 1]


@pytest.mark.parametrize("key", ["escape", "q"])
def test_quit_key_exits_when_no_job_is_pending_or_running(monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    # Unknown is deliberately included: the quit condition is the absence of
    # known active work, not that every job has reached a terminal state.
    jobs, graph = _jobs_with_states(["done", "failed", "unknown"])
    exited_after_key: list[bool] = []

    async def drive(app: Any, pilot: Any) -> None:
        await pilot.press(key)
        await pilot.pause()
        exited_after_key.append(app._exit)

    _run_monitor(monkeypatch, jobs, graph, drive)

    assert exited_after_key == [True]


@pytest.mark.parametrize("key", ["escape", "q"])
@pytest.mark.parametrize("active_state", ["pending", "running"])
def test_quit_key_is_blocked_while_job_is_active(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    active_state: JobState,
) -> None:
    jobs, graph = _jobs_with_states([active_state, "done", "unknown"])
    exited_after_key: list[bool] = []

    async def drive(app: Any, pilot: Any) -> None:
        await pilot.press(key)
        await pilot.pause()
        exited_after_key.append(app._exit)
        app.exit()

    _run_monitor(monkeypatch, jobs, graph, drive)

    assert exited_after_key == [False]
