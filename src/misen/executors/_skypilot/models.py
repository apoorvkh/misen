"""SkyPilot capacity declarations and durable graph records."""

from __future__ import annotations

import re
from collections import deque
from typing import TYPE_CHECKING, Any, Literal

import msgspec

from misen.exceptions import (
    StorageError,
)
from misen.task_metadata import AcceleratorType, Resources, aggregate_resources

if TYPE_CHECKING:
    from collections.abc import Iterable

    from misen.workspace import Workspace

# Capacity profiles
# ----------------------------------------------------------------------------

_POOL_NAME = re.compile(r"[a-zA-Z](?:[-_.a-zA-Z0-9]*[a-zA-Z0-9])?")


_CLUSTER_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


_MODEL_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*")


_MAX_SOURCE_NAME_LENGTH = 63


_MAX_OPTION_LENGTH = 512


def _positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = f"{name} must be a positive integer."
        raise ValueError(msg)


def _option_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        msg = f"{name} must be a non-empty string."
        raise ValueError(msg)  # noqa: TRY004 - configuration validation uses ValueError
    normalized = value.strip()
    if not normalized or len(normalized) > _MAX_OPTION_LENGTH or not normalized.isprintable():
        msg = (
            f"{name} must be a non-empty string of at most {_MAX_OPTION_LENGTH} characters without control characters."
        )
        raise ValueError(msg)
    return normalized


class SkyPilotCapacity(msgspec.Struct, kw_only=True, forbid_unknown_fields=True):
    """A fixed per-worker reservation from one SkyPilot capacity source.

    Exactly one of ``pool``, ``cluster``, or ``infra`` identifies the source.
    Existing pools and clusters are borrowed: Misen must not reconfigure or
    terminate them. ``infra`` creates run-owned workers. ``cpus``, ``memory``,
    and accelerator quantities declare the reservation per node, not inferred
    hardware capacity; the allocation backend must validate borrowed capacity
    before assigning work. A cluster supplies one worker reservation, whereas
    ``max_workers`` bounds independently acquired pool or run-owned workers.

    A multi-node reservation requires ``dedicated=True``. Dedicated capacity
    runs one work unit at a time and reserves its entire declared shape even
    for a smaller request. Device names are concrete SkyPilot models; the
    programming backend and per-device memory are declared separately.
    """

    pool: str | None = None
    cluster: str | None = None
    infra: str | list[str] | None = None
    cpus: int = 1
    memory: int = 8
    accelerators: dict[str, int] = msgspec.field(default_factory=dict)
    accelerator_type: AcceleratorType = "cuda"
    accelerator_memory: int | None = None
    max_workers: int = 1
    dedicated: bool = False
    nodes: int = 1
    use_spot: bool = False
    instance_type: str | None = None
    image_id: str | None = None
    disk_size: int | None = None

    def __post_init__(self) -> None:
        """Validate resource limits without importing SkyPilot or contacting it."""
        if sum(source is not None for source in (self.pool, self.cluster, self.infra)) != 1:
            msg = "Exactly one capacity source must be set: pool, cluster, or infra."
            raise ValueError(msg)
        for name, pattern in (("pool", _POOL_NAME), ("cluster", _CLUSTER_NAME)):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or len(value) > _MAX_SOURCE_NAME_LENGTH or pattern.fullmatch(value) is None
            ):
                msg = f"{name} must be a valid SkyPilot name of at most {_MAX_SOURCE_NAME_LENGTH} characters."
                raise ValueError(msg)
        if self.infra is not None:
            if isinstance(self.infra, str):
                self.infra = _option_string(self.infra, "infra")
            elif isinstance(self.infra, list) and self.infra:
                alternatives = [_option_string(value, "infra") for value in self.infra]
                if len(set(alternatives)) != len(alternatives):
                    msg = "infra must not contain duplicate alternatives."
                    raise ValueError(msg)
                self.infra = alternatives
            else:
                msg = "infra must be a non-empty SkyPilot infrastructure string or list of strings."
                raise ValueError(msg)
        for name in ("cpus", "memory", "max_workers", "nodes"):
            _positive_integer(getattr(self, name), name)
        for name in ("dedicated", "use_spot"):
            if not isinstance(getattr(self, name), bool):
                msg = f"{name} must be a boolean."
                raise ValueError(msg)  # noqa: TRY004 - configuration validation uses ValueError
        if self.nodes > 1 and not self.dedicated:
            msg = "Multi-node capacity requires dedicated=True."
            raise ValueError(msg)
        if self.cluster is not None and self.max_workers != 1:
            msg = "An existing cluster requires max_workers=1; it is one declared reservation."
            raise ValueError(msg)
        if self.accelerator_type not in ("cuda", "rocm", "xpu", "mps", "tpu"):
            msg = f"Unsupported accelerator type: {self.accelerator_type!r}."
            raise ValueError(msg)
        if not isinstance(self.accelerators, dict) or len(self.accelerators) > 1:
            msg = "accelerators must be a dictionary containing at most one concrete SkyPilot model."
            raise ValueError(msg)
        normalized_accelerators: dict[str, int] = {}
        for raw_model, count in self.accelerators.items():
            model = _option_string(raw_model, "accelerator model")
            if _MODEL_NAME.fullmatch(model) is None:
                msg = "accelerator model must contain only letters, digits, periods, underscores, or hyphens."
                raise ValueError(msg)
            _positive_integer(count, "accelerator count")
            normalized_accelerators[model] = count
        self.accelerators = normalized_accelerators
        if self.accelerator_memory is not None:
            _positive_integer(self.accelerator_memory, "accelerator_memory")
            if not self.accelerators:
                msg = "accelerator_memory requires an accelerator model."
                raise ValueError(msg)
        for name in ("instance_type", "image_id"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, _option_string(value, name))
        if self.disk_size is not None:
            _positive_integer(self.disk_size, "disk_size")
        if self.borrowed and (
            self.use_spot or self.instance_type is not None or self.image_id is not None or self.disk_size is not None
        ):
            msg = (
                "Borrowed pool/cluster capacity cannot set creation options: "
                "use_spot, instance_type, image_id, disk_size."
            )
            raise ValueError(msg)

    @property
    def borrowed(self) -> bool:
        """Whether this source belongs to the user rather than the current run."""
        return self.infra is None

    @property
    def accelerator_count(self) -> int:
        """Return the number of accelerator devices reserved per node."""
        return sum(self.accelerators.values())

    def fits(self, resources: Resources) -> bool:
        """Whether a valid work-unit request fits an otherwise idle reservation.

        This is shape compatibility, not current availability or SkyPilot
        hardware verification. CPU-only work may fit accelerator capacity;
        placement policy should prefer CPU-only workers when suitable.
        Unknown device memory never satisfies an explicit minimum.
        """
        request = aggregate_resources((resources,), sum_time=False)
        if request["cpus"] > self.cpus or request["memory"] > self.memory or request["nodes"] > self.nodes:
            return False
        if request["accelerators"] == 0:
            return True
        if request["accelerators"] > self.accelerator_count or request["accelerator_type"] != self.accelerator_type:
            return False
        required_memory = request["accelerator_memory"]
        return required_memory is None or (
            self.accelerator_memory is not None and required_memory <= self.accelerator_memory
        )

    def as_sky_options(self) -> dict[str, Any]:
        """Build SkyPilot ``Resources`` arguments for the reserved shape.

        Node count belongs on ``sky.Task(num_nodes=...)`` and is not included
        here. Borrowed sources omit infrastructure and creation-only options.
        The caller selects the pool or cluster separately.
        """
        options: dict[str, Any] = {"cpus": f"{self.cpus}+", "memory": f"{self.memory}+"}
        if self.accelerators:
            options["accelerators"] = dict(self.accelerators)
        if not self.borrowed:
            options["infra"] = list(self.infra) if isinstance(self.infra, list) else self.infra
            options["use_spot"] = self.use_spot
            for name in ("instance_type", "image_id", "disk_size"):
                value = getattr(self, name)
                if value is not None:
                    options[name] = value
        return options


# Durable graph records and readiness
# ----------------------------------------------------------------------------


class GraphWork(msgspec.Struct, forbid_unknown_fields=True):
    """One immutable logical work unit, with an already-staged payload."""

    job_id: str
    dependencies: list[str]
    profile: str
    argv: list[str]
    env: dict[str, str]
    log_path: str
    resources: Resources
    uses_dask_client: bool = False
    direct: bool = False
    payload_name: str | None = None


class AgentWork(msgspec.Struct, forbid_unknown_fields=True):
    """A staged agent bootstrap; its capacity may execute many logical jobs."""

    worker_id: str
    profile: str
    job_id: str
    argv: list[str]
    env: dict[str, str]
    log_path: str


class RunManifest(msgspec.Struct, forbid_unknown_fields=True):
    """Durable submission data; no live coordinator or SDK objects."""

    run_id: str
    snapshot_key: str
    nodes: list[GraphWork]
    agents: list[AgentWork]
    control_id: str | None = None
    runtime_key: str | None = None
    version: Literal[1] = 1


class LogicalState(msgspec.Struct, forbid_unknown_fields=True):
    """Persisted state of a logical job, not a SkyPilot allocation."""

    state: Literal["pending", "running", "done", "failed", "unknown"] = "pending"
    reason: str | None = "Waiting for dependencies."
    attempt_id: str | None = None
    worker_id: str | None = None


class RunState(msgspec.Struct, forbid_unknown_fields=True):
    """Coalesced index for batched observation without per-job cloud queries."""

    run_id: str
    jobs: dict[str, LogicalState]
    status: Literal["running", "done", "failed", "interrupted"] = "running"
    cleanup_errors: list[str] = msgspec.field(default_factory=list)
    version: Literal[1] = 1
    heartbeat_at: float = 0.0


class ReadyGraph:
    """Linear-time dependency accounting and ready-only admission.

    Each node has one attempt in this implementation. Uncertain execution is
    never replayed automatically. Independent branches survive another branch's
    failure, and blocked descendants never consume a capacity reservation.
    """

    def __init__(self, nodes: Iterable[GraphWork]) -> None:
        """Validate an immutable DAG and initialize its ready queue."""
        ordered = list(nodes)
        self.nodes = {node.job_id: node for node in ordered}
        if len(self.nodes) != len(ordered):
            msg = "Duplicate logical job identity in run manifest."
            raise ValueError(msg)
        self.states = {key: LogicalState() for key in self.nodes}
        self.remaining: dict[str, int] = {}
        self.dependents: dict[str, list[str]] = {key: [] for key in self.nodes}
        for node in ordered:
            if len(set(node.dependencies)) != len(node.dependencies):
                msg = "Duplicate dependency in run manifest."
                raise ValueError(msg)
            self.remaining[node.job_id] = len(node.dependencies)
            for dependency in node.dependencies:
                if dependency not in self.nodes:
                    msg = "Unknown logical dependency in run manifest."
                    raise ValueError(msg)
                self.dependents[dependency].append(node.job_id)
        counts = dict(self.remaining)
        queue = deque(key for key, count in counts.items() if count == 0)
        visited = 0
        while queue:
            key = queue.popleft()
            visited += 1
            for child in self.dependents[key]:
                counts[child] -= 1
                if counts[child] == 0:
                    queue.append(child)
        if visited != len(self.nodes):
            msg = "Run manifest contains a dependency cycle."
            raise ValueError(msg)
        self.ready = deque(key for key, count in self.remaining.items() if count == 0)
        for key in self.ready:
            self.states[key].reason = "Waiting for compatible capacity."
        self.active: set[str] = set()
        self.finished: set[str] = set()
        self.revision = 0

    @property
    def complete(self) -> bool:
        """Whether every logical node has a terminal outcome."""
        return len(self.finished) == len(self.nodes)

    def assign(self, job_id: str, attempt_id: str, worker_id: str) -> None:
        """Claim only a ready, unassigned logical job."""
        if self.remaining[job_id] or job_id in self.active or job_id in self.finished:
            msg = "Only a ready, unassigned work unit can be assigned."
            raise ValueError(msg)
        state = self.states[job_id]
        state.attempt_id = attempt_id
        state.worker_id = worker_id
        state.reason = "Preparing execution environment."
        self.active.add(job_id)
        self.revision += 1

    def running(self, job_id: str, attempt_id: str) -> bool:
        """Apply a matching started event; ignore stale or duplicate events."""
        state = self.states[job_id]
        if job_id not in self.active or state.attempt_id != attempt_id:
            return False
        if state.state == "running":
            return False
        state.state = "running"
        state.reason = None
        self.revision += 1
        return True

    def finish(self, job_id: str, *, success: bool, reason: str | None = None) -> None:
        """Commit a terminal logical outcome and advance affected descendants."""
        if job_id in self.finished:
            return
        queue = deque([(job_id, success, reason)])
        while queue:
            key, succeeded, failure = queue.popleft()
            if key in self.finished:
                continue
            self.finished.add(key)
            self.revision += 1
            self.active.discard(key)
            self.states[key].state = "done" if succeeded else "failed"
            self.states[key].reason = failure
            for child in self.dependents[key]:
                if child in self.finished:
                    continue
                if not succeeded:
                    queue.append((child, False, f"Dependency {key} did not succeed."))
                else:
                    self.remaining[child] -= 1
                    if self.remaining[child] == 0:
                        self.states[child].reason = "Waiting for compatible capacity."
                        self.ready.append(child)

    def apply_result(self, job_id: str, attempt_id: str, *, success: bool, reason: str | None = None) -> bool:
        """Accept an outcome only for the currently assigned attempt."""
        if job_id not in self.active or self.states[job_id].attempt_id != attempt_id:
            return False
        self.finish(job_id, success=success, reason=reason)
        return True


def _maximum_bipartite_matching(adjacency: list[list[int]]) -> int:
    """Return a maximum matching cardinality with Hopcroft-Karp."""
    size = len(adjacency)
    left_matches = [-1] * size
    right_matches = [-1] * size
    distance = [0] * size
    unreachable = size + 1
    cardinality = 0

    while True:
        queue: deque[int] = deque()
        shortest = unreachable
        for left, right in enumerate(left_matches):
            distance[left] = 0 if right < 0 else unreachable
            if right < 0:
                queue.append(left)
        while queue:
            left = queue.popleft()
            if distance[left] >= shortest:
                continue
            for right in adjacency[left]:
                partner = right_matches[right]
                if partner < 0:
                    shortest = distance[left] + 1
                elif distance[partner] == unreachable:
                    distance[partner] = distance[left] + 1
                    queue.append(partner)
        if shortest == unreachable:
            return cardinality

        for start, match in enumerate(left_matches):
            if match >= 0 or distance[start] == unreachable:
                continue
            # Each frame retains its next edge. Distances strictly increase,
            # so a successful stack is an alternating augmenting path.
            stack = [(start, 0)]
            while stack:
                left, offset = stack[-1]
                if offset == len(adjacency[left]):
                    distance[left] = unreachable
                    stack.pop()
                    continue
                right = adjacency[left][offset]
                stack[-1] = (left, offset + 1)
                partner = right_matches[right]
                if partner < 0:
                    if distance[left] + 1 != shortest:
                        continue
                    current_right = right
                    for path_left, _ in reversed(stack):
                        previous_right = left_matches[path_left]
                        left_matches[path_left] = current_right
                        right_matches[current_right] = path_left
                        current_right = previous_right
                    cardinality += 1
                    break
                if distance[partner] == distance[left] + 1:
                    stack.append((partner, 0))


def profile_dependency_widths(nodes: Iterable[GraphWork]) -> dict[str, int]:
    """Return exact dependency-permitted maximum concurrency by profile.

    Each profile's work units form a poset under reachability in the complete
    DAG. Paths through other profiles therefore still order their endpoints.
    By Dilworth's theorem, the maximum antichain size is the node count minus
    a maximum matching in the poset's bipartite reachability graph.

    Capacity limits, resource shapes, and task durations are intentionally not
    considered here. ``ReadyGraph`` supplies the manifest validation contract.
    """
    graph = ReadyGraph(nodes)
    ordered = list(graph.nodes.values())
    if not ordered:
        return {}

    positions = {node.job_id: index for index, node in enumerate(ordered)}
    remaining = dict(graph.remaining)
    queue = deque(graph.ready)
    topological: list[str] = []
    while queue:
        job_id = queue.popleft()
        topological.append(job_id)
        for child in graph.dependents[job_id]:
            remaining[child] -= 1
            if remaining[child] == 0:
                queue.append(child)

    descendants = [0] * len(ordered)
    for job_id in reversed(topological):
        index = positions[job_id]
        for child in graph.dependents[job_id]:
            child_index = positions[child]
            descendants[index] |= descendants[child_index] | (1 << child_index)

    profiles: dict[str, list[int]] = {}
    for index, node in enumerate(ordered):
        profiles.setdefault(node.profile, []).append(index)

    widths: dict[str, int] = {}
    for profile, members in profiles.items():
        local_positions = {global_index: local_index for local_index, global_index in enumerate(members)}
        member_mask = sum(1 << global_index for global_index in members)
        adjacency: list[list[int]] = []
        for global_index in members:
            reachable = descendants[global_index] & member_mask
            neighbors: list[int] = []
            while reachable:
                bit = reachable & -reachable
                neighbors.append(local_positions[bit.bit_length() - 1])
                reachable ^= bit
            adjacency.append(neighbors)
        widths[profile] = len(members) - _maximum_bipartite_matching(adjacency)
    return widths


def read_run_state(workspace: Workspace, run_id: str) -> RunState:
    """Read the batched durable state index, rejecting wrong-run records.

    Raises:
        StorageError: If the durable record is malformed or belongs to another run.
    """
    try:
        state = msgspec.json.decode(workspace.read_job_file(run_id, "run-state.json"), type=RunState)
    except msgspec.DecodeError as exc:
        msg = "Invalid graph run state."
        raise StorageError(msg) from exc
    if state.run_id != run_id:
        msg = "Graph state belongs to a different run."
        raise StorageError(msg)
    return state
