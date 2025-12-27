import pytest

from misen.executor import Executor, Job, WorkUnit
from misen.task import Task, resources, task
from misen.workspace import TaskLogs, Workspace, WorkspaceParameters
from misen.workspaces.memory import MemoryWorkspace


class FakeJob(Job):
    def __init__(self, name: str):
        self.name = name


class FakeExecutor(Executor[FakeJob]):
    def __init__(self) -> None:
        self.dispatches: list[dict[str, object]] = []

    def _dispatch(self, function, resources, dependencies):
        job = FakeJob(str(len(self.dispatches)))
        self.dispatches.append(
            {
                "function": function,
                "resources": resources,
                "dependencies": dependencies,
                "job": job,
            }
        )
        return job


class DummyLogs(TaskLogs):
    def __init__(self, content: str) -> None:
        self.content = content


@task(cache=True)
def add(a: int, b: int) -> int:
    return a + b


@task(cache=True)
def double(x: int) -> int:
    return x * 2


@task(cache=True)
def cached_value(x: int) -> int:
    return x + 1


@task(cache=False)
def multiply(x: int) -> int:
    return x * 2


@task(cache=False)
def combine(a: int, b: int) -> int:
    return a + b


def test_task_result_caching_and_dependencies():
    workspace = MemoryWorkspace()

    dependency = Task(double, x=3)
    root = Task(add, a=dependency.T, b=1)

    with pytest.raises(RuntimeError, match="dependencies"):
        root.result(workspace=workspace, compute_if_uncached=True)

    assert root.result(workspace=workspace, compute_if_uncached=True, compute_uncached_deps=True) == 7
    assert root.is_cached(workspace=workspace)
    assert dependency.is_cached(workspace=workspace)
    assert root.result(workspace=workspace) == 7


def test_task_dependency_graph_excludes_cacheable():
    workspace = MemoryWorkspace()

    noncache = Task(multiply, x=2)
    cacheable = Task(cached_value, x=3)
    root = Task(combine, a=noncache.T, b=cacheable.T)

    graph = root._dependency_graph(exclude_cacheable=True, workspace=workspace)
    nodes = set(graph.nodes())

    assert root in nodes
    assert noncache in nodes
    assert cacheable not in nodes


def test_workunit_resource_aggregation():
    @resources(time=5, memory=2, cpus=1, gpus=0)
    @task(cache=False)
    def step_one(x: int) -> int:
        return x + 1

    @resources(time=3, memory=8, cpus=4, gpus=1, gpu_memory=12)
    @task(cache=False)
    def step_two(x: int) -> int:
        return x * 2

    root = Task(step_one, x=Task(step_two, x=1).T)
    work_unit = WorkUnit(task=root, dependencies=set())

    assert work_unit.resources.time == 8
    assert work_unit.resources.memory == 8
    assert work_unit.resources.cpus == 4
    assert work_unit.resources.gpus == 1
    assert work_unit.resources.gpu_memory == 12


def test_executor_submit_dispatches_with_dependencies():
    workspace = MemoryWorkspace()

    cached_task = Task(cached_value, x=1)
    base_task = Task(multiply, x=cached_task.T)
    root = Task(combine, a=base_task.T, b=2)

    executor = FakeExecutor()
    executor.submit(task=root, workspace=workspace)

    assert len(executor.dispatches) == 2

    dispatch_by_dep_count = {len(d["dependencies"]): d for d in executor.dispatches}
    assert set(dispatch_by_dep_count) == {0, 1}

    dependent_dispatch = dispatch_by_dep_count[1]
    independent_dispatch = dispatch_by_dep_count[0]

    dependency_job = next(iter(dependent_dispatch["dependencies"]))
    assert dependency_job is independent_dispatch["job"]


def test_workspace_maps_and_singleton_behavior(tmp_path):
    workspace = MemoryWorkspace()

    task_instance = Task(add, a=1, b=2)

    assert task_instance not in workspace.results
    with pytest.raises(KeyError):
        _ = workspace.results[task_instance]

    assert task_instance.result(workspace=workspace, compute_if_uncached=True) == 3
    assert task_instance in workspace.results
    assert workspace.results[task_instance].value() == 3
    assert "not a task" not in workspace.results

    log_entry = DummyLogs("run-1")
    workspace.logs[task_instance] = log_entry
    assert task_instance in workspace.logs
    assert workspace.logs[task_instance] is log_entry
    del workspace.logs[task_instance]
    assert task_instance not in workspace.logs

    workspace_again = MemoryWorkspace()
    assert workspace is workspace_again

    class NamedWorkspace(Workspace):
        def __init__(self, name: str, base_dir):
            self.name = name
            self.base_dir = base_dir
            super().__init__(
                resolved_hash_cache={},
                result_hash_cache={},
                result_cache={},
                log_store={},
            )

        def to_params(self):
            return WorkspaceParameters(NamedWorkspace, name=self.name, base_dir=self.base_dir)

        def get_work_dir(self, task: Task):
            path = self.base_dir / self.name
            path.mkdir(exist_ok=True)
            return path

    first = NamedWorkspace(name="alpha", base_dir=tmp_path)
    second = NamedWorkspace(name="alpha", base_dir=tmp_path)
    third = NamedWorkspace(name="beta", base_dir=tmp_path)

    assert first is second
    assert first is not third
