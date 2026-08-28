"""Workspace abstraction for caching, locking, and runtime artifacts.

``Workspace`` isolates storage concerns from execution concerns:

- Executors schedule and run work.
- Tasks describe computation and identity.
- Workspace persists hashes/results and coordinates cross-process locks.
- Runtime lock contract: for cacheable tasks, a workspace lock keyed by
  resolved task identity enforces at most one active execution at a time.
  Non-cacheable tasks are not serialized by this runtime lock.

This separation keeps execution backends modular while preserving one
consistent cache/locking contract.
"""

from __future__ import annotations

import logging
import shutil
from abc import abstractmethod
from collections.abc import Iterator, MutableMapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TextIO, TypeAlias, TypeVar, cast

from misen.exceptions import CacheError, LockUnavailableError, SerializationError, StorageError
from misen.tasks import Task
from misen.utils import serde
from misen.utils.fsync import atomic_write_bytes as _atomic_write_bytes
from misen.utils.settings import Configurable

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from misen.utils.hashing import ResolvedTaskHash, ResultHash, TaskHash
    from misen.utils.locks import LockLike
    from misen.utils.work_unit import WorkUnit

__all__ = ["Workspace"]


WorkspaceType: TypeAlias = Literal["disk", "cloud", "memory"]
TRACE_LEVEL = logging.DEBUG - 5
logger = logging.getLogger(__name__)
_HashMappingT = TypeVar("_HashMappingT")


def _hash_mapping_type(cls: type[_HashMappingT], item: tuple[type[Any], type[Any]]) -> type[_HashMappingT]:
    """Build a hash mapping specialized with concrete key/value types."""
    key_type, value_type = item
    return cast(
        "type[_HashMappingT]",
        type(
            f"{cls.__name__}[{key_type.__name__},{value_type.__name__}]",
            (cls,),
            {"_key_type": key_type, "_value_type": value_type, "__module__": cls.__module__},
        ),
    )


@contextmanager
def _storage_errors(
    operation: str,
    *errors: type[BaseException],
    passthrough: tuple[type[BaseException], ...] = (),
) -> Iterator[None]:
    """Translate backend I/O failures at one storage boundary."""
    caught = errors or (OSError,)
    try:
        yield
    except caught as exc:
        if isinstance(exc, passthrough):
            raise
        msg = f"{operation}: {exc}"
        raise StorageError(msg) from exc


def _require_result_lock(lock: LockLike, task: Task[Any]) -> None:
    """Fail immediately before publish when a result lock was lost."""
    if not lock.is_locked():
        msg = f"Lost the result lock for task {task} before its payload could be committed."
        raise LockUnavailableError(msg)


class Workspace(Configurable):
    """Base class for workspace storage backends.

    Concrete implementations provide persistence (for hashes/results), lock
    implementations, and task/job log storage.
    """

    _config_key: ClassVar[str] = "workspace"
    _config_default_type: ClassVar[str] = "misen.workspaces.disk:DiskWorkspace"
    _config_aliases: ClassVar[dict[WorkspaceType, str]] = {
        "disk": "misen.workspaces.disk:DiskWorkspace",
        "cloud": "misen.workspaces.cloud:CloudWorkspace",
        "memory": "misen.workspaces.memory:InMemoryWorkspace",
    }

    def _post_init(
        self,
        resolved_hash_cache: MutableMapping[TaskHash, ResolvedTaskHash],
        result_hash_cache: MutableMapping[ResolvedTaskHash, ResultHash],
        result_store: MutableMapping[ResultHash, Path],
    ) -> None:
        """Initialize workspace caches and storage backends.

        Args:
            resolved_hash_cache: Persistent cache for task resolved hashes.
            result_hash_cache: Persistent cache for task result hashes.
            result_store: Store mapping result hashes to on-disk directories.
        """
        # Session-local hot caches reduce repeated backend lookups.
        self._resolved_hashes: dict[TaskHash, ResolvedTaskHash] = {}
        self._result_hashes: dict[TaskHash, ResultHash] = {}

        # Persistent caches/stores are backend-specific (e.g., sharded files on disk, cloud objects).
        self._resolved_hash_cache: MutableMapping[TaskHash, ResolvedTaskHash] = resolved_hash_cache
        self._result_hash_cache: MutableMapping[ResolvedTaskHash, ResultHash] = result_hash_cache
        self._result_map = ResultMap(result_store=result_store, workspace=self)

    def get_resolved_hash(self, task: Task) -> ResolvedTaskHash | None:
        """Return cached resolved hash for a task, if available.

        Args:
            task: Task to query.

        Returns:
            Resolved hash if present, otherwise ``None``.

        Raises:
            StorageError: If the persistent hash index cannot be read.
        """
        task_hash = task.task_hash()
        # Fast path: in-memory session cache.
        resolved_hash = self._resolved_hashes.get(task_hash)
        if resolved_hash is not None:
            logger.log(TRACE_LEVEL, "Resolved-hash memory cache hit for task %s.", task)
            return resolved_hash
        # Slow path: persistent workspace cache.
        with _storage_errors(f"Could not read the resolved hash for task {task}"):
            resolved_hash = self._resolved_hash_cache.get(task_hash)
        # Promote to session cache after a persistent hit.
        if resolved_hash is not None:
            self._resolved_hashes[task_hash] = resolved_hash
            logger.log(TRACE_LEVEL, "Resolved-hash persistent cache hit for task %s.", task)
        else:
            logger.log(TRACE_LEVEL, "Resolved-hash cache miss for task %s.", task)
        return resolved_hash

    def set_resolved_hash(self, task: Task, resolved_hash: ResolvedTaskHash) -> None:
        """Persist resolved hash for a task.

        Args:
            task: Task to update.
            resolved_hash: Resolved task hash value.

        Raises:
            StorageError: If the persistent hash index cannot be written.
        """
        task_hash = task.task_hash()
        with _storage_errors(f"Could not persist the resolved hash for task {task}"):
            self._resolved_hash_cache[task_hash] = resolved_hash
        self._resolved_hashes[task_hash] = resolved_hash
        logger.debug("Stored resolved hash for task %s.", task)

    def get_result_hash(self, task: Task) -> ResultHash:
        """Return the result hash for a completed task.

        Raises:
            CacheError: If the task has not been computed yet.
            StorageError: If the persistent hash index cannot be read.
        """
        # Fast path: in-memory session cache.
        task_hash = task.task_hash()
        result_hash = self._result_hashes.get(task_hash)
        if result_hash is not None:
            logger.log(TRACE_LEVEL, "Result-hash memory cache hit for task %s.", task)
            return result_hash
        # Slow path: persistent workspace cache by resolved task identity.
        resolved_hash = task.resolved_hash(workspace=self)
        with _storage_errors(f"Could not read the result hash for task {task}"):
            result_hash = self._result_hash_cache.get(resolved_hash)
        if result_hash is None:
            logger.log(TRACE_LEVEL, "Result-hash cache miss for task %s.", task)
            msg = f"Task {task} must be computed first."
            raise CacheError(msg)
        # Promote to session cache after a persistent hit.
        self._result_hashes[task_hash] = result_hash
        logger.log(TRACE_LEVEL, "Result-hash persistent cache hit for task %s.", task)
        return result_hash

    def set_result_hash(
        self,
        task: Task,
        result_hash: ResultHash,
        *,
        before_commit: Callable[[], None] | None = None,
    ) -> None:
        """Persist result hash for a task.

        Args:
            task: Task to update.
            result_hash: Result hash value.
            before_commit: Optional ownership check run at the durable pointer
                commit point.

        Raises:
            StorageError: If the persistent hash index cannot be written.
        """
        resolved_hash = task.resolved_hash(workspace=self)
        with _storage_errors(f"Could not persist the result hash for task {task}"):
            if before_commit is None:
                self._result_hash_cache[resolved_hash] = result_hash
            elif commit := getattr(self._result_hash_cache, "commit", None):
                commit(resolved_hash, result_hash, before_commit=before_commit)
            else:
                before_commit()
                self._result_hash_cache[resolved_hash] = result_hash
        self._result_hashes[task.task_hash()] = result_hash
        logger.debug("Stored result hash for task %s.", task)

    def clear_result_hash(self, task: Task) -> None:
        """Remove persisted result-hash mapping for a task.

        Args:
            task: Task whose mapping should be removed.

        Raises:
            StorageError: If the persistent hash index cannot be updated.
        """
        resolved_hash = task.resolved_hash(workspace=self)
        try:
            del self._result_hash_cache[resolved_hash]
        except KeyError:
            pass
        except OSError as exc:
            msg = f"Could not clear the result hash for task {task}: {exc}"
            raise StorageError(msg) from exc
        self._result_hashes.pop(task.task_hash(), None)
        logger.debug("Cleared result hash for task %s.", task)

    @property
    def results(self) -> ResultMap:
        """Return mapping-like interface for cached task results."""
        return self._result_map

    @abstractmethod
    def lock(self, namespace: Literal["task", "result", "job"], key: str) -> LockLike:
        """Return a lock object for task/result/job namespaces.

        Args:
            namespace: Lock namespace (task runtime, result materialization, or job submission).
            key: Lock key unique within the namespace.

        Returns:
            Lock-like object with acquire/release/context APIs.

        Notes:
            ``namespace="task"`` is used for cacheable task runtime exclusion
            (single active execution per workspace/key). ``namespace="result"``
            is used for serialized result materialization.
        """

    @abstractmethod
    def get_temp_dir(self) -> Path:
        """Return temporary directory used for workspace operations."""

    @abstractmethod
    def publish_snapshot(self, key: str, staged_dir: Path) -> None:
        """Publish a staged project tree under its immutable content key.

        Publication must be idempotent. Implementations may consume
        ``staged_dir``; callers treat it as gone after this call.

        Args:
            key: Content key of the staged tree.
            staged_dir: Directory containing the fully-written tree.
        """

    @abstractmethod
    def fetch_snapshot(self, key: str) -> Path:
        """Return a local read-only directory holding snapshot ``key``.

        Remote backends materialize into a backend-owned local cache.

        Raises:
            FileNotFoundError: If no snapshot is published under ``key``.
        """

    @abstractmethod
    def put_job_file(self, submission_id: str, name: str, data: bytes) -> str:
        """Store submission-scoped bytes and return an opaque ref.

        Job files may contain secrets such as ``.env.local``. Path-backed
        implementations must use owner-only permissions.

        Args:
            submission_id: Grouping key for one executor submission.
            name: File name within the submission (no path separators).
            data: File contents.

        Returns:
            A path when :meth:`bootstrap_transport` returns ``None``;
            otherwise an opaque ref consumed by that transport.
        """

    def read_job_file(self, submission_id: str, name: str) -> bytes:
        """Read submission-scoped bytes previously written by this workspace.

        Executors may use small job files for durable coordination between
        workers. Missing files must raise :class:`FileNotFoundError` so a
        consumer can distinguish "not published yet" from a storage failure.

        Backends that do not support coordination reads may retain this
        default; executors requiring the capability must reject them during
        submission preflight.
        """
        raise NotImplementedError

    def supports_job_file_reads(self) -> bool:
        """Return whether this backend implements submission-file reads."""
        return type(self).read_job_file is not Workspace.read_job_file

    @staticmethod
    def _validate_job_file_name(name: str) -> None:
        """Reject names that could escape a submission's job-file prefix."""
        if not name or "/" in name or "\\" in name or name in {".", ".."}:
            msg = f"Invalid job-file name: {name!r}"
            raise ValueError(msg)

    @abstractmethod
    def bootstrap_transport(self) -> str | None:
        """Return Bash that fetches snapshots/job files, or ``None`` for paths.

        Misen invokes it separately for the snapshot and every job file with
        these environment variables:

        - ``MISEN_TRANSPORT_OPERATION``: ``snapshot`` or ``job-file``;
        - ``MISEN_TRANSPORT_REF``: the content key or opaque job-file ref;
        - ``MISEN_TRANSPORT_DEST``: the local path the script must create;
        - ``MISEN_UV_BIN``: worker-resolved uv;
        - optional ``MISEN_PIXI_BIN`` when the project itself uses Pixi.

        The script must materialize a directory for ``snapshot`` and a file
        for ``job-file``. It is responsible for resolving any extra tools it
        needs, using ``MISEN_UV_BIN`` or its own shell commands. Misen owns
        temporary destinations, validation, publication, and per-host reuse.
        Its source may be visible in executor or scheduler command lines, so
        it must not contain credentials. Refs must be unique for a given
        transport source; transports with workspace-local refs should include
        a stable, non-secret workspace identity in their source.

        Python-backed transports can use
        :func:`misen.utils.bootstrap_transport.render_python_transport` to
        extract a self-contained function and declare its worker dependencies
        while preserving this Bash interface.

        ``None`` means no transport is needed because snapshot and job-file
        refs are directly worker-visible paths.
        """
        raise NotImplementedError

    def get_scratch_dir(self, task: Task) -> Path:
        """Return a per-task scratch directory for cacheable tasks.

        Args:
            task: Task requesting its scratch directory.

        Returns:
            Filesystem path for runtime intermediate artifacts.

        Raises:
            RuntimeError: If the task is non-cacheable.
            StorageError: If the workspace cannot prepare the scratch directory.
        """
        if not task.meta.cache:
            msg = f"{task} cannot use workspace scratch_dir unless Task.meta.cache == True."
            raise RuntimeError(msg)
        with _storage_errors(f"Could not prepare a scratch directory for task {task}"):
            return self._get_scratch_dir(task)

    @abstractmethod
    def _get_scratch_dir(self, task: Task) -> Path: ...

    def start_scratch_dir_sync(self, task: Task) -> None:
        """Begin syncing a cacheable task's scratch_dir with durable storage.

        Workspaces with off-machine durable storage (e.g.
        :class:`misen.workspaces.cloud.CloudWorkspace`) should override
        this to download any existing snapshot from durable storage into
        the local scratch_dir and start a background uploader that
        periodically pushes local writes back. The uploader gives
        cacheable tasks a checkpoint location: writes that reach
        durable storage survive a worker crash, so a future invocation
        with the same resolved hash can resume from the latest synced
        state.

        Implementations must be idempotent: subsequent calls while sync
        is already active are no-ops. Implementations must also be safe
        under abnormal exit (worker killed mid-execution): on-exit
        :meth:`finalize_scratch_dir` should leave durable storage in a
        consistent state if it runs, but if it does not run the next
        invocation must still produce correct behavior.
        """
        del task

    def finalize_scratch_dir(self, task: Task) -> None:
        """Stop the background sync and perform a final upload sweep.

        Idempotent. Called by the runtime after the task function
        returns (success or failure). For cacheable tasks the scratch_dir
        contents are preserved in durable storage so a future
        resumption can start from the latest checkpoint. Path-backed
        workspaces may implement this as a no-op.
        """
        del task

    def remove_scratch_dir(self, task: Task) -> None:
        """Remove durable + local copies of a cacheable task's scratch_dir.

        Called by :meth:`misen.tasks.Task.result` after a successful run.
        The default implementation removes only the local directory;
        backends with off-machine durable storage override to also delete
        remote objects.

        Args:
            task: Cacheable task whose scratch_dir should be removed.

        Raises:
            RuntimeError: If the task is non-cacheable.
        """
        if not task.meta.cache:
            msg = f"{task} cannot use workspace scratch_dir unless Task.meta.cache == True."
            raise RuntimeError(msg)
        path = self._get_scratch_dir(task)
        with _storage_errors(f"Could not remove scratch directory {path}"):
            if path.exists():
                shutil.rmtree(path)

    def _task_log_dir(self, task: Task) -> tuple[Path, str]:
        """Return the (created) log directory and per-task key for ``task``.

        Hook behind the default path-backed :meth:`get_task_log` /
        :meth:`read_task_log`. Logs are keyed by resolved hash so two runs
        of the same task with different dependency results land in
        distinct files; resolving requires every dependency's result hash
        to be cached, so early callers should expect ``CacheError``.
        """
        raise NotImplementedError

    def get_task_log(self, task: Task, job_id: str | None = None) -> Path:
        """Return path where ``task``'s log for ``job_id`` should be written.

        Logs are keyed by ``(task_hash, job_id)``; each task execution
        produces one log file per job. ``job_id=None`` selects a default
        identifier so callers without a backend-assigned job id still get
        a stable path.

        Workspaces that publish to remote storage (e.g.
        :class:`misen.workspaces.cloud.CloudWorkspace`) override to start
        streaming the local file on this call; the matching
        :meth:`finalize_task_log` call stops it.

        Raises:
            CacheError: If the task's dependencies have not been resolved.
            StorageError: If the log path cannot be prepared.
        """
        with _storage_errors(f"Could not prepare a task log for {task}"):
            log_dir, key_str = self._task_log_dir(task)
        return log_dir / f"{key_str}_{job_id or '0'}.log"

    def finalize_task_log(self, task: Task, job_id: str | None = None) -> None:
        """Hook called when a task log is no longer being written.

        Workspaces that publish locally-written logs to a shared store
        should override this to flush the final state. Implementations
        must be idempotent and tolerant of a missing local file.
        Path-backed workspaces may implement this as a no-op.
        """
        del task, job_id

    def read_task_log(self, task: Task, job_id: str | None = None) -> TextIO:
        """Open a previously-written task log for reading.

        If ``job_id`` is provided, opens that specific log. If ``job_id``
        is ``None``, opens the most recent log for the task. Recency is
        implementation-defined (e.g., filesystem mtime, object-store
        upload timestamp).

        Raises:
            CacheError: If the task's dependencies have not been resolved.
            FileNotFoundError: If no matching log exists.
            StorageError: If the log path cannot be listed or opened.
        """
        with _storage_errors(f"Could not read a task log for {task}", passthrough=(FileNotFoundError,)):
            log_dir, key_str = self._task_log_dir(task)
            if job_id is None:
                matches = sorted(log_dir.glob(f"{key_str}_*.log"), key=lambda p: p.stat().st_mtime)
                log_path = matches[-1] if matches else None
            else:
                log_path = log_dir / f"{key_str}_{job_id}.log"
            if log_path is not None:
                return log_path.open("r", buffering=1)
        msg = f"No logs found for {key_str} in {log_dir}"
        raise FileNotFoundError(msg)

    def task_log_iter(self, task: Task) -> Iterator[tuple[str, Path]]:
        """Return ``(job_id, path)`` pairs for every available task log.

        Raises:
            CacheError: If the task's dependencies have not been resolved.
            StorageError: If the logs cannot be listed.
        """
        with _storage_errors(f"Could not locate task logs for {task}"):
            log_dir, key_str = self._task_log_dir(task)

        def iter_paths() -> Iterator[tuple[str, Path]]:
            with _storage_errors(f"Could not list task logs in {log_dir}"):
                prefix = f"{key_str}_"
                for path in log_dir.glob(f"{prefix}*.log"):
                    yield path.stem.removeprefix(prefix), path

        return iter_paths()

    def _job_logs_dir(self) -> Path:
        """Return the local directory where job-log files live.

        Subclasses may override to relocate logs (e.g. out of an
        ephemeral temp dir). The default is alongside the workspace's
        temporary directory.
        """
        return self.get_temp_dir().parent / "job_logs"

    def get_job_log(self, job_id: str, work_unit: WorkUnit) -> Path:
        """Return job-log path for a work unit.

        Args:
            job_id: Backend job identifier.
            work_unit: Work unit associated with the job.

        Returns:
            Path where the backend should write combined job logs.

        Raises:
            StorageError: If the job-log directory cannot be created.
        """
        log_dir = self._job_logs_dir()
        with _storage_errors(f"Could not prepare job-log directory {log_dir}"):
            log_dir.mkdir(parents=True, exist_ok=True)
        work_unit_prefix = work_unit.root.task_hash().b32()
        path = log_dir / f"{work_unit_prefix}_{job_id}.log"
        logger.debug("Resolved job log path for work unit %s: %s.", work_unit, path)
        return path

    def streaming_job_log(self, local_path: Path) -> AbstractContextManager[None]:
        """Return a context manager that publishes ``local_path`` while it is open.

        The worker process that writes ``local_path`` is expected to wrap
        its entire lifecycle in ``with workspace.streaming_job_log(...):``.
        Workspaces that publish to a remote shared store (e.g.
        :class:`misen.workspaces.cloud.CloudWorkspace`) start a background
        uploader on enter and finalize on exit. Path-backed workspaces may
        return a no-op context manager when ``local_path`` is already on
        durable shared storage.

        Implementations must be safe under abnormal exit (e.g. the worker
        being killed mid-execution): the context's ``__exit__`` should
        still leave the bucket in a consistent state if it runs.
        """
        del local_path
        return nullcontext()

    def finalize_job_log(self, local_path: Path) -> None:
        """One-shot publish of ``local_path``'s current contents.

        Intended to be called by the parent (executor) after the job has
        reached a terminal state, to capture anything written to the file
        *after* the worker's :meth:`streaming_job_log` context closed --
        most importantly, a SLURM epilogue, which the controller writes
        to ``--output`` once the wrapped command has exited.

        Implementations must be idempotent and tolerant of a missing
        local file. Workspaces where ``local_path`` is already on durable
        shared storage may implement this as a no-op.
        """
        del local_path

    def job_log_iter(self, work_unit: WorkUnit | None = None) -> Iterator[Path]:
        """Return iterator over job-log files.

        Args:
            work_unit: Optional filter for a specific work unit.

        Returns:
            Iterator of log-file paths.

        Raises:
            StorageError: If the job-log directory cannot be listed.
        """
        log_dir = self._job_logs_dir()
        if work_unit is None:
            logger.debug("Iterating all job logs in %s.", log_dir)
            paths = log_dir.iterdir()
        else:
            work_unit_prefix = work_unit.root.task_hash().b32()
            logger.debug("Iterating job logs in %s for work unit %s.", log_dir, work_unit)
            paths = log_dir.glob(f"{work_unit_prefix}_*.log")

        def iter_paths() -> Iterator[Path]:
            with _storage_errors(f"Could not list job logs in {log_dir}"):
                yield from paths

        return iter_paths()


class _PathWorkspace(Workspace):
    """Workspace whose job-file references are directly visible paths."""

    def put_job_file(self, submission_id: str, name: str, data: bytes) -> str:
        """Store an owner-only submission file and return its path."""
        self._validate_job_file_name(name)
        path = self.get_temp_dir().parent / "job_files" / submission_id / name
        with _storage_errors(f"Could not persist job file {name!r} for submission {submission_id!r}"):
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_bytes(path, data)
            path.chmod(0o600)
        return str(path)

    def read_job_file(self, submission_id: str, name: str) -> bytes:
        """Read one submission file without hiding a not-yet-published file."""
        self._validate_job_file_name(name)
        path = self.get_temp_dir().parent / "job_files" / submission_id / name
        with _storage_errors(f"Could not read job file {name!r}", passthrough=(FileNotFoundError,)):
            return path.read_bytes()

    def bootstrap_transport(self) -> None:
        """Use directly worker-visible snapshot and job-file paths."""


R = TypeVar("R")


class ResultMap(MutableMapping[Task[Any], Any]):
    """Mapping-like interface over workspace result storage.

    Keys are :class:`misen.tasks.Task` objects and values are deserialized
    result payloads.
    """

    __slots__ = ("result_store", "workspace")

    def __init__(self, result_store: MutableMapping[ResultHash, Path], workspace: Workspace) -> None:
        """Initialize result-map wrapper.

        Args:
            result_store: Mapping from result hash to payload directory.
            workspace: Owning workspace.
        """
        self.result_store = result_store
        self.workspace = workspace

    def _result_hash(self, task: Task[Any]) -> ResultHash:
        try:
            return task.result_hash(workspace=self.workspace)
        except CacheError as exc:
            msg = f"Result for task {task} not found in cache."
            raise KeyError(msg) from exc
        except OSError as exc:
            msg = f"Could not resolve the cached result for task {task}: {exc}"
            raise StorageError(msg) from exc

    def __getitem__(self, key: Task[R], /) -> R:
        """Return the cached result for a task.

        Raises:
            KeyError: If the result is not present in the cache.
        """
        result_hash = self._result_hash(key)

        try:
            directory = self.result_store[result_hash]
        except KeyError as e:
            msg = f"Result for task {key} not found in cache."
            raise KeyError(msg) from e
        except OSError as e:
            msg = f"Could not read the cached result for task {key}: {e}"
            raise StorageError(msg) from e
        logger.debug("Loading cached result for task %s from %s.", key, directory)
        try:
            return serde.load(directory, ser_cls=key.meta.serializer)
        except KeyError as exc:
            msg = f"Could not deserialize the cached result for task {key}: {exc}"
            raise SerializationError(msg) from exc

    def __setitem__(self, key: Task[R], value: R, /) -> None:
        """Persist result for the given task.

        Derives the content-addressed ``result_hash`` from the task's stored
        pointer. Prefer :meth:`store` on the write path where the hash is
        already known, so the payload can be committed before the pointer.

        Args:
            key: Task key.
            value: Computed result value.
        """
        self.store(key, value, key.result_hash(workspace=self.workspace))

    def store(
        self,
        task: Task[R],
        value: R,
        result_hash: ResultHash,
        *,
        before_commit: Callable[[], None] | None = None,
    ) -> None:
        """Persist ``value`` at the content-addressed location for ``result_hash``.

        ``result_hash`` is supplied explicitly rather than read back from the
        workspace pointer so the payload can be committed *before* the durable
        ``resolved_hash -> result_hash`` pointer exists. This is what lets
        :func:`misen.utils.task_utils.save_task_result` order the payload ahead
        of the pointer and so never strand a pointer without its payload.

        Args:
            task: Task whose serializer materializes the payload.
            value: Computed result value.
            result_hash: Content-addressed identity the payload is stored under.
            before_commit: Optional ownership check run at the payload's
                durable commit point.
        """
        with _storage_errors(f"Could not persist the cached result for task {task}"):
            result_lock = self.workspace.lock(namespace="result", key=result_hash.b32())
            with result_lock.context(blocking=True, timeout=None):
                if result_hash in self.result_store:
                    logger.debug("Result store already has payload for task %s.", task)
                    return
                tmp_dir = self.workspace.get_temp_dir() / "results" / result_hash.b32()
                tmp_dir.mkdir(parents=True, exist_ok=True)
                try:
                    serde.save(value, tmp_dir, ser_cls=task.meta.serializer)

                    def require_ownership() -> None:
                        _require_result_lock(result_lock, task)
                        if before_commit is not None:
                            before_commit()

                    # ``result_store[...] = tmp_dir`` moves the directory into the
                    # store; tmp_dir is consumed on success.
                    if commit := getattr(self.result_store, "commit", None):
                        commit(result_hash, tmp_dir, before_commit=require_ownership)
                    else:
                        require_ownership()
                        self.result_store[result_hash] = tmp_dir
                    logger.debug("Stored cached result for task %s at %s.", task, tmp_dir)
                finally:
                    # Cleanup is best-effort and must not replace the primary
                    # serialization/storage failure.
                    shutil.rmtree(tmp_dir, ignore_errors=True)

    def __delitem__(self, key: Task[R], /) -> None:
        """Remove a cached result for a task.

        Raises:
            KeyError: If the result is not present in the cache.
        """
        result_hash = self._result_hash(key)

        try:
            del self.result_store[result_hash]
            logger.debug("Deleted cached result payload for task %s.", key)
        except KeyError as e:
            msg = f"Result for task {key} not found in cache."
            raise KeyError(msg) from e
        except OSError as e:
            msg = f"Could not delete the cached result for task {key}: {e}"
            raise StorageError(msg) from e

    def __iter__(self) -> Iterator[Task]:
        """Iterate over task keys.

        Notes:
            This is not implemented because the persistent store is keyed by
            result hashes rather than by task identity.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Return number of cached results."""
        return len(self.result_store)

    def __contains__(self, key: object, /) -> bool:
        """Return True if the task has a cached result."""
        if not isinstance(key, Task):
            return False
        try:
            result_hash = key.result_hash(workspace=self.workspace)
        except CacheError:
            return False
        return result_hash in self.result_store
