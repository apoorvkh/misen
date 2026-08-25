# ruff: noqa: D102, D105, D107
"""Object-store-backed workspace for S3, GCS, and Azure Blob.

This backend stores hash indices, result payloads, task logs, and job logs in a
cloud object store through ``obstore``. Scratch directories and actively-written
logs stay on local cache storage; live logs are uploaded as offset chunks and
compacted into a single ``.log`` object on close.
"""

from __future__ import annotations

import binascii
import contextlib
import io
import logging
import shutil
import tarfile
import tempfile
import threading
import uuid
from collections.abc import Iterator, MutableMapping
from importlib.metadata import version as package_version
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, Self, TextIO, TypeAlias, TypeVar, cast

import msgspec
import obstore as obs
from obstore.exceptions import AlreadyExistsError
from obstore.exceptions import BaseError as ObstoreError
from obstore.store import AzureStore, GCSStore, S3Store
from xxhash import xxh3_64_hexdigest

from misen.exceptions import ConfigError, LockUnavailableError, StorageError
from misen.utils.bootstrap_transport import render_python_transport
from misen.utils.hashing import Hash, ResolvedTaskHash, ResultHash, TaskHash
from misen.utils.locks import ObjectStoreLock, _cleanup_on_exit
from misen.utils.serde import MANIFEST_FILENAME
from misen.workspace import Workspace, _hash_mapping_type, _storage_errors

if TYPE_CHECKING:
    from collections.abc import Callable

    from misen.tasks import Task
    from misen.utils.locks import LockLike
    from misen.utils.work_unit import WorkUnit

__all__ = ("CloudBackend", "CloudWorkspace", "ObstoreMapping", "ObstoreResultStore")


KT = TypeVar("KT", bound=Hash)
VT = TypeVar("VT", bound=Hash)
CloudBackend: TypeAlias = Literal["s3", "gcs", "azure"]
_CHUNKS = ".chunks"
_STATE = ".state.json"
_LOG_CHUNK_SIZE = 8 * 1024 * 1024
_RESULT_POINTER_PREFIX = b"misen-result-v2:"
logger = logging.getLogger(__name__)


def _result_payload_prefix(remote_prefix: str, pointer: bytes) -> str:
    """Resolve a committed generation pointer, preserving legacy layouts."""
    if not pointer.startswith(_RESULT_POINTER_PREFIX):
        return remote_prefix
    generation = pointer.removeprefix(_RESULT_POINTER_PREFIX).decode("ascii")
    if uuid.UUID(hex=generation).hex != generation:
        msg = f"Invalid result generation {generation!r}."
        raise ValueError(msg)
    return f"{remote_prefix}/.builds/{generation}"


def _require_result_manifest(directory: Path, key: ResultHash) -> None:
    if not (directory / MANIFEST_FILENAME).is_file():
        msg = f"Committed result {key.b32()} has no {MANIFEST_FILENAME}."
        raise StorageError(msg)


@contextlib.contextmanager
def _cloud_errors(
    operation: str,
    *extra: type[BaseException],
    passthrough: tuple[type[BaseException], ...] = (),
) -> Iterator[None]:
    """Translate local/object-store failures at a cloud boundary."""
    with _storage_errors(operation, OSError, ObstoreError, *extra, passthrough=passthrough):
        yield


# --------------------------------------------------------------------------
# Obstore data-plane helpers shared by workspace operations.
# --------------------------------------------------------------------------


def _join_prefix(prefix: str, *parts: str) -> str:
    """Join object-key parts under an optional prefix (no leading/trailing slashes)."""
    return "/".join(part for part in (prefix, *(p.strip("/") for p in parts if p)) if part)


def _obstore_store_config(
    backend: str,
    *,
    endpoint: str | None = None,
    s3_region: str | None = None,
    config: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Merge dedicated endpoint/region fields into obstore configuration.

    Raises:
        ValueError: If a dedicated field also appears in ``config``.
    """
    cfg = cast("dict[str, Any]", dict(config or {}))
    explicit: dict[str, Any] = {}
    if backend == "s3":
        if s3_region is not None:
            explicit["region"] = s3_region
        if endpoint is not None:
            explicit["endpoint"] = endpoint
    elif backend == "azure":
        if endpoint is not None:
            explicit["endpoint"] = endpoint
    for key, value in explicit.items():
        if key in cfg:
            msg = f"{key!r} cannot appear in both config and the dedicated field."
            raise ValueError(msg)
        cfg[key] = value
    return cfg


def _build_obstore_store(
    backend: str,
    bucket: str,
    *,
    endpoint: str | None = None,
    s3_region: str | None = None,
    config: dict[str, str] | None = None,
) -> Any:
    """Construct the obstore client for a backend/bucket/config triple."""
    cfg = _obstore_store_config(backend, endpoint=endpoint, s3_region=s3_region, config=config)

    if backend == "s3":
        return S3Store(bucket=bucket, **cfg)
    if backend == "gcs":
        return GCSStore(bucket=bucket, **cfg)
    if backend == "azure":
        return AzureStore(container_name=bucket, **cfg)
    msg = f"Unsupported cloud backend: {backend!r}"
    raise ValueError(msg)


def _snapshot_object_key(prefix: str, key: str) -> str:
    """Object key of a published snapshot tarball."""
    return _join_prefix(prefix, "snapshots", f"{key}.tar.gz")


def _download_snapshot(store: Any, prefix: str, key: str, target_dir: Path) -> Path:
    """Materialize a published snapshot tarball into ``target_dir``.

    Extracts into a sibling temp dir and renames into place, tolerating a
    concurrent fetch of the same key (first rename wins; losers discard).

    Raises:
        FileNotFoundError: If no snapshot is published under ``key``.
    """
    if target_dir.exists():
        return target_dir
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    object_key = _snapshot_object_key(prefix, key)
    try:
        payload = obs.get(store, object_key).bytes()
    except FileNotFoundError as e:
        msg = f"No snapshot published under key {key!r} at {object_key}."
        raise FileNotFoundError(msg) from e
    tmp = Path(tempfile.mkdtemp(dir=target_dir.parent, prefix=f".{key}.", suffix=".tmp"))
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
            tar.extractall(tmp, filter="data")
        tmp.rename(target_dir)
    except OSError:
        # Renaming onto a directory a concurrent fetch just created fails
        # (EEXIST/ENOTEMPTY); identical content is already in place.
        shutil.rmtree(tmp, ignore_errors=True)
        if not target_dir.exists():
            raise
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return target_dir


def _delete_prefix(store: Any, prefix: str) -> None:
    """Bulk-delete every object under ``prefix``."""
    keys = [entry["path"] for batch in obs.list(store, prefix=prefix) for entry in batch]
    if keys:
        obs.delete(store, keys)


class ObstoreMapping(MutableMapping[KT, VT], Generic[KT, VT]):
    """Typed hash->hash mapping stored as one object per key."""

    _key_type: type[KT]
    _value_type: type[VT]
    __slots__ = ("_prefix", "_store")

    def __class_getitem__(cls, item: tuple[type[KT], type[VT]]) -> type[Self]:
        return _hash_mapping_type(cls, item)

    def __init__(self, store: Any, prefix: str) -> None:
        if not hasattr(self, "_key_type") or not hasattr(self, "_value_type"):
            msg = "Construct as ObstoreMapping[KeyType, ValueType](...)"
            raise TypeError(msg)
        self._store = store
        self._prefix = prefix.rstrip("/")

    def __getitem__(self, key: KT) -> VT:
        try:
            with _cloud_errors(
                f"Could not read stored value for {key!r}",
                passthrough=(FileNotFoundError,),
            ):
                return self._value_type.decode(bytes(obs.get(self._store, f"{self._prefix}/{key.b32()}").bytes()))
        except FileNotFoundError as e:
            raise KeyError(key) from e
        except ValueError as exc:
            msg = f"Stored value for {key!r} is corrupt or incompatible."
            raise StorageError(msg) from exc

    def __setitem__(self, key: KT, value: VT) -> None:
        with _cloud_errors(f"Could not persist stored value for {key!r}"):
            obs.put(self._store, f"{self._prefix}/{key.b32()}", value.encode(), mode="overwrite")

    def commit(self, key: KT, value: VT, *, before_commit: Callable[[], None]) -> None:
        """Create a fenced, write-once index entry without stale overwrites."""
        try:
            with _cloud_errors(
                f"Could not persist stored value for {key!r}",
                passthrough=(AlreadyExistsError,),
            ):
                before_commit()
                obs.put(self._store, f"{self._prefix}/{key.b32()}", value.encode(), mode="create")
        except AlreadyExistsError:
            if self[key] != value:
                msg = f"Another writer committed a different value for {key!r}."
                raise LockUnavailableError(msg) from None

    def __delitem__(self, key: KT) -> None:
        path = f"{self._prefix}/{key.b32()}"
        try:
            with _cloud_errors(f"Could not delete stored value for {key!r}", passthrough=(FileNotFoundError,)):
                obs.head(self._store, path)
                obs.delete(self._store, path)
        except FileNotFoundError as e:
            raise KeyError(key) from e

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, self._key_type):
            return False
        try:
            with _cloud_errors(f"Could not inspect stored value for {key!r}", passthrough=(FileNotFoundError,)):
                obs.head(self._store, f"{self._prefix}/{key.b32()}")
        except FileNotFoundError:
            return False
        return True

    def __iter__(self) -> Iterator[KT]:
        prefix = f"{self._prefix}/"
        with _cloud_errors(f"Could not list stored values under {self._prefix}"):
            for batch in obs.list(self._store, prefix=prefix):
                for entry in batch:
                    rel = entry["path"][len(prefix) :]
                    if not rel:
                        continue
                    try:
                        yield self._key_type.from_b32(rel)
                    except (binascii.Error, ValueError, TypeError):
                        continue

    def __len__(self) -> int:
        return sum(1 for _ in self)


class ObstoreResultStore(MutableMapping[ResultHash, Path]):
    """Result payload store backed by cloud objects and a local materialization cache."""

    __slots__ = ("_cache_dir", "_prefix", "_store")

    def __init__(self, store: Any, prefix: str, cache_dir: Path) -> None:
        self._store = store
        self._prefix = prefix.rstrip("/")
        self._cache_dir = cache_dir
        with _storage_errors(f"Could not initialize result cache at {self._cache_dir}"):
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, ResultHash):
            return False
        with _storage_errors(f"Could not inspect the local cache for result {key.b32()}"):
            cached = (self._cache_dir / key.b32()).exists()
        if cached:
            return True
        try:
            with _cloud_errors(f"Could not inspect result {key.b32()}", passthrough=(FileNotFoundError,)):
                obs.head(self._store, f"{self._prefix}/{key.b32()}/{MANIFEST_FILENAME}")
        except FileNotFoundError:
            return False
        return True

    def __getitem__(self, key: ResultHash) -> Path:
        local = self._cache_dir / key.b32()
        with _storage_errors(f"Could not inspect the local cache for result {key.b32()}"):
            cached = local.exists()
        if cached:
            return local

        remote_prefix = f"{self._prefix}/{key.b32()}"
        try:
            with _cloud_errors(
                f"Could not inspect result {key.b32()}",
                UnicodeError,
                ValueError,
                passthrough=(FileNotFoundError,),
            ):
                pointer = bytes(obs.get(self._store, f"{remote_prefix}/{MANIFEST_FILENAME}").bytes())
                payload_prefix = _result_payload_prefix(remote_prefix, pointer)
        except FileNotFoundError as e:
            raise KeyError(key) from e

        with _storage_errors(f"Could not create a local cache directory for result {key.b32()}"):
            tmp = Path(tempfile.mkdtemp(dir=self._cache_dir, prefix=f".{key.b32()}.", suffix=".tmp"))
        try:
            with _cloud_errors(f"Could not materialize result {key.b32()}", passthrough=(FileExistsError,)):
                for batch in obs.list(self._store, prefix=f"{payload_prefix}/"):
                    for entry in batch:
                        rel = entry["path"][len(payload_prefix) + 1 :]
                        if not rel or (payload_prefix == remote_prefix and rel.startswith(".builds/")):
                            continue
                        target = tmp / rel
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(bytes(obs.get(self._store, entry["path"]).bytes()))
                _require_result_manifest(tmp, key)
                tmp.rename(local)
        except FileExistsError:
            shutil.rmtree(tmp, ignore_errors=True)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return local

    def __setitem__(self, key: ResultHash, value: Path) -> None:
        self.commit(key, value, before_commit=lambda: None)

    def commit(self, key: ResultHash, value: Path, *, before_commit: Callable[[], None]) -> None:
        """Upload to an immutable generation, then atomically publish its pointer."""
        remote_prefix = f"{self._prefix}/{key.b32()}"
        try:
            with _cloud_errors(f"Could not inspect result {key.b32()}", passthrough=(FileNotFoundError,)):
                obs.head(self._store, f"{remote_prefix}/{MANIFEST_FILENAME}")
        except FileNotFoundError:
            pass
        else:
            return

        manifest = value / MANIFEST_FILENAME
        if not manifest.is_file():
            raise FileNotFoundError(manifest)

        generation = uuid.uuid4().hex
        build_prefix = f"{remote_prefix}/.builds/{generation}"
        uploaded: list[str] = []

        def discard_generation() -> None:
            if not uploaded:
                return
            try:
                obs.delete(self._store, uploaded)
            except (OSError, ObstoreError):
                logger.warning("Could not remove unpublished result generation %s.", build_prefix, exc_info=True)

        try:
            with _cloud_errors(f"Could not upload result {key.b32()}"):
                for path in sorted(p for p in value.rglob("*") if p.is_file()):
                    remote_path = f"{build_prefix}/{path.relative_to(value).as_posix()}"
                    with path.open("rb") as f:
                        obs.put(self._store, remote_path, f, mode="overwrite")
                    uploaded.append(remote_path)
            before_commit()
        except BaseException:
            discard_generation()
            raise

        try:
            with _cloud_errors(
                f"Could not publish result {key.b32()}",
                passthrough=(AlreadyExistsError,),
            ):
                obs.put(
                    self._store,
                    f"{remote_prefix}/{MANIFEST_FILENAME}",
                    _RESULT_POINTER_PREFIX + generation.encode(),
                    mode="create",
                )
        except AlreadyExistsError:
            discard_generation()

    def __delitem__(self, key: ResultHash) -> None:
        prefix = f"{self._prefix}/{key.b32()}/"
        with _cloud_errors(f"Could not delete result {key.b32()}"):
            keys = [entry["path"] for batch in obs.list(self._store, prefix=prefix) for entry in batch]
            local = self._cache_dir / key.b32()
            commit_key = f"{prefix}{MANIFEST_FILENAME}"
            if commit_key not in keys and not local.exists():
                raise KeyError(key)
            if commit_key in keys:
                obs.delete(self._store, [commit_key])
                keys.remove(commit_key)
            if keys:
                obs.delete(self._store, keys)
            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(local)

    def __iter__(self) -> Iterator[ResultHash]:
        prefix = f"{self._prefix}/"
        b32_len = len(ResultHash(0).b32())
        with _cloud_errors(f"Could not list results under {self._prefix}"):
            for batch in obs.list(self._store, prefix=prefix):
                for entry in batch:
                    rel = entry["path"][len(prefix) :]
                    head, sep, tail = rel.partition("/")
                    if sep != "/" or tail != MANIFEST_FILENAME or len(head) != b32_len:
                        continue
                    try:
                        yield ResultHash.from_b32(head)
                    except (binascii.Error, ValueError, TypeError):
                        continue

    def __len__(self) -> int:
        return sum(1 for _ in self)


class _PeriodicPublisher:
    """Shared lifecycle for best-effort periodic cloud publication."""

    __slots__ = (
        "_finalized",
        "_interval_s",
        "_lifecycle_lock",
        "_local_path",
        "_remote_key",
        "_stop",
        "_store",
        "_thread",
    )

    _failure_message: ClassVar[str]
    _finalized_on_skip: ClassVar[bool] = False
    _stop_prefix: ClassVar[str]
    _thread_prefix: ClassVar[str]

    def __init__(self, store: Any, local_path: Path, remote_key: str, interval_s: float) -> None:
        self._store = store
        self._local_path = local_path
        self._remote_key = remote_key
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lifecycle_lock = threading.Lock()
        self._finalized = False

    def _before_start(self) -> None:
        pass

    def _publish_once(self) -> None: ...

    def _finalize(self) -> None: ...

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._before_start()
            self._finalized = False
            self._thread = threading.Thread(
                target=self._run,
                daemon=True,
                name=f"{self._thread_prefix}[{self._remote_key}]",
            )
            self._thread.start()

    @property
    def active(self) -> bool:
        """Return whether periodic publication is accepting more work."""
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                self._publish_once()
            except (OSError, ObstoreError):
                logger.exception(self._failure_message, self._local_path, self._remote_key)

    def stop(self, *, final_upload: bool = True) -> None:
        with self._lifecycle_lock:
            self._stop.set()
            thread = self._thread
            if thread is not None and thread.is_alive():
                timeout_s = max(self._interval_s * 2, 2.0)
                thread.join(timeout=timeout_s)
                if thread.is_alive():
                    msg = f"{self._stop_prefix} {self._local_path} did not stop within {timeout_s:g}s."
                    raise StorageError(msg)
            self._thread = None
            if not final_upload:
                if self._finalized_on_skip:
                    self._finalized = True
                return
            if not self._finalized:
                self._finalize()
                self._finalized = True


class _ScratchDirSync(_PeriodicPublisher):
    """Sync a local scratch_dir to/from cloud object storage.

    The runtime lock for cacheable tasks already guarantees a single
    active execution per resolved hash, so this class assumes the local
    directory has exactly one writer (the task itself) for the lifetime
    of one start/stop cycle.

    On :meth:`restore` the latest snapshot under ``remote_prefix`` is
    downloaded into ``local_dir`` (overwriting any existing local
    files at matching paths), seeding the per-file ``(size, mtime)``
    map so the uploader does not immediately re-upload restored bytes.
    On :meth:`start` a background thread wakes every ``interval_s``
    seconds, walks the local tree, and pushes any new or modified file
    plus deletes any remote object whose local counterpart was removed.
    On :meth:`stop` the thread is joined and (by default) one final
    sweep runs so the bucket reflects the directory's terminal state.
    """

    __slots__ = ("_known",)

    _failure_message = "Scratch dir sync failed for %s -> %s."
    _finalized_on_skip = True
    _stop_prefix = "Scratch sync for"
    _thread_prefix = "misen-scratch-sync"

    def __init__(self, store: Any, local_dir: Path, remote_prefix: str, interval_s: float) -> None:
        super().__init__(store, local_dir, remote_prefix.rstrip("/"), interval_s)
        self._known: dict[str, tuple[int, float]] = {}

    def restore(self) -> None:
        """Download the remote snapshot into the local dir."""
        prefix = f"{self._remote_key}/"
        for batch in obs.list(self._store, prefix=prefix):
            for entry in batch:
                rel = entry["path"][len(prefix) :]
                if not rel:
                    continue
                local_path = self._local_path / rel
                local_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    data = bytes(obs.get(self._store, entry["path"]).bytes())
                except FileNotFoundError:
                    continue
                local_path.write_bytes(data)
                stat = local_path.stat()
                self._known[rel] = (stat.st_size, stat.st_mtime)

    def _publish_once(self) -> None:
        seen: set[str] = set()
        if self._local_path.exists():
            for path in self._local_path.rglob("*"):
                if not path.is_file():
                    continue
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    continue
                rel = path.relative_to(self._local_path).as_posix()
                seen.add(rel)
                current = (stat.st_size, stat.st_mtime)
                if self._known.get(rel) == current:
                    continue
                try:
                    with path.open("rb") as f:
                        obs.put(self._store, f"{self._remote_key}/{rel}", f, mode="overwrite")
                except FileNotFoundError:
                    continue
                self._known[rel] = current
        # Mirror local deletions to the remote, but only for keys we
        # previously uploaded -- never touch objects this sync did not
        # produce, so a stale or unrelated entry under the prefix is
        # left alone.
        for rel in [r for r in self._known if r not in seen]:
            with contextlib.suppress(FileNotFoundError):
                obs.delete(self._store, f"{self._remote_key}/{rel}")
            del self._known[rel]

    def _finalize(self) -> None:
        with _cloud_errors(f"Could not finalize scratch directory {self._local_path} to {self._remote_key}"):
            self._publish_once()

    def finalize_current_tree(self) -> None:
        """Mirror the current local tree after an earlier owner was closed.

        This is used when ``CloudWorkspace.close()`` raced with the task
        finalizer.  Seeding every remote key with an impossible local stat
        makes the final sweep both upload every extant local file and delete
        remote files that the task removed after ``close()`` returned.
        """
        with self._lifecycle_lock:
            prefix = f"{self._remote_key}/"
            with _cloud_errors(f"Could not finalize scratch directory {self._local_path} to {self._remote_key}"):
                for batch in obs.list(self._store, prefix=prefix):
                    for entry in batch:
                        rel = entry["path"][len(prefix) :]
                        if rel:
                            self._known[rel] = (-1, -1.0)
                self._publish_once()
            self._finalized = True


class _LiveLogUploader(_PeriodicPublisher):
    """Upload appended log chunks in the background, then compact on close."""

    __slots__ = ("_uploaded_offset",)

    _failure_message = "Live chunk upload failed for %s -> %s."
    _stop_prefix = "Log uploader for"
    _thread_prefix = "misen-log-upload"

    def __init__(self, store: Any, local_path: Path, remote_key: str, interval_s: float) -> None:
        super().__init__(store, local_path, remote_key, interval_s)
        self._uploaded_offset = 0

    def _before_start(self) -> None:
        self._delete_live_objects()

    def _publish_once(self) -> None:
        try:
            size = self._local_path.stat().st_size
        except FileNotFoundError:
            return
        if size < self._uploaded_offset:
            self._delete_live_objects()
            self._uploaded_offset = 0
        if size <= self._uploaded_offset:
            return

        offset = self._uploaded_offset
        with self._local_path.open("rb") as f:
            f.seek(offset)
            while offset < size and (payload := f.read(min(_LOG_CHUNK_SIZE, size - offset))):
                obs.put(self._store, f"{self._remote_key}{_CHUNKS}/{offset:020d}.chunk", payload, mode="overwrite")
                offset += len(payload)
        if offset == self._uploaded_offset:
            return
        obs.put(
            self._store,
            f"{self._remote_key}{_STATE}",
            msgspec.json.encode({"offset": offset, "closed": False}),
            mode="overwrite",
        )
        self._uploaded_offset = offset

    def _delete_live_objects(self) -> None:
        keys = [
            entry["path"] for batch in obs.list(self._store, prefix=f"{self._remote_key}{_CHUNKS}/") for entry in batch
        ]
        with contextlib.suppress(FileNotFoundError):
            obs.head(self._store, f"{self._remote_key}{_STATE}")
            keys.append(f"{self._remote_key}{_STATE}")
        if keys:
            obs.delete(self._store, keys)

    def compact(self) -> None:
        """Publish the canonical final log and remove superseded live chunks.

        Raises:
            StorageError: If the local log or backing object store cannot be
                accessed while publishing the final log.
        """
        try:
            with _cloud_errors(
                f"Could not finalize log {self._local_path} to {self._remote_key}",
                passthrough=(FileNotFoundError,),
            ):
                with self._local_path.open("rb") as local_file:
                    obs.put(self._store, self._remote_key, local_file, mode="overwrite")
                self._uploaded_offset = self._local_path.stat().st_size
        except FileNotFoundError:
            return
        try:
            self._delete_live_objects()
        except (OSError, ObstoreError):
            logger.exception("Compacted %s, but live-log cleanup failed for %s.", self._local_path, self._remote_key)

    def _finalize(self) -> None:
        self.compact()


class CloudWorkspace(Workspace):
    """Workspace backed by S3, GCS, or Azure Blob via obstore."""

    backend: CloudBackend
    bucket: str
    prefix: str = ""
    endpoint: str | None = None
    s3_region: str | None = None
    config: dict[str, str] = msgspec.field(default_factory=dict)
    cache_dir: str = ".cache/misen"
    log_flush_interval_s: float = 1.0
    scratch_dir_sync_interval_s: float = 30.0
    _config_validation_errors: ClassVar[tuple[type[Exception], ...]] = (ValueError,)

    def __post_init__(self) -> None:
        if self.log_flush_interval_s <= 0:
            msg = "log_flush_interval_s must be positive"
            raise ValueError(msg)
        if self.scratch_dir_sync_interval_s <= 0:
            msg = "scratch_dir_sync_interval_s must be positive"
            raise ValueError(msg)
        if self.s3_region is not None and self.backend != "s3":
            msg = f"s3_region is only supported for backend='s3', got backend={self.backend!r}."
            raise ValueError(msg)
        if self.endpoint is not None and self.backend == "gcs":
            msg = "endpoint is not supported for backend='gcs'."
            raise ValueError(msg)

        with _cloud_errors(f"Could not initialize {self.backend} object store for bucket {self.bucket!r}"):
            self._store = self._build_store()
        self._cloud_prefix = self.prefix.strip("/")
        # Append a deterministic id so distinct workspaces never share cache.
        # Two workspaces with identical identity-affecting fields collapse to
        # the same subdir, which is exactly when sharing is safe.
        self._cache = Path(self.cache_dir) / self.workspace_id
        with _storage_errors(f"Could not initialize the cloud workspace cache at {self._cache}"):
            for subdir in (
                "tmp",
                "scratch",
                "task_logs",
                "task_log_cache",
                "job_logs",
                "job_log_cache",
                "results_cache",
            ):
                (self._cache / subdir).mkdir(parents=True, exist_ok=True)
        self._live_log_uploaders: dict[Path, _LiveLogUploader] = {}
        self._live_log_lock = threading.Lock()
        self._scratch_dir_syncs: dict[str, _ScratchDirSync] = {}
        self._scratch_dir_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._closed = False

        super()._post_init(
            resolved_hash_cache=ObstoreMapping[TaskHash, ResolvedTaskHash](
                self._store,
                self._under("resolved_hash_cache"),
            ),
            result_hash_cache=ObstoreMapping[ResolvedTaskHash, ResultHash](
                self._store,
                self._under("result_hash_cache"),
            ),
            result_store=ObstoreResultStore(self._store, self._under("results"), self._cache / "results_cache"),
        )
        logger.info(
            "Initialized CloudWorkspace id=%s backend=%s bucket=%s cache=%s.",
            self.workspace_id,
            self.backend,
            self.bucket,
            self._cache,
        )

    @property
    def workspace_id(self) -> str:
        """Short deterministic id derived from identity-affecting fields.

        Two workspaces with the same ``(backend, bucket, prefix, endpoint,
        s3_region)`` produce the same id and may safely share local cache;
        any other pair produces distinct ids.
        """
        payload = msgspec.json.encode((self.backend, self.bucket, self.prefix, self.endpoint, self.s3_region))
        return xxh3_64_hexdigest(payload)

    def _build_store(self) -> Any:
        return _build_obstore_store(
            self.backend, self.bucket, endpoint=self.endpoint, s3_region=self.s3_region, config=self.config
        )

    @staticmethod
    def _bootstrap_transport(context: dict[str, str | None], operation: str, ref: str, destination: Path) -> None:
        """Fetch one snapshot or job file using only the declared worker dependency."""
        import io
        import tarfile

        import obstore as obs
        from obstore.store import AzureStore, GCSStore, S3Store

        backend = context["backend"]
        bucket = context["bucket"]
        prefix = context["prefix"]
        if backend is None or bucket is None or prefix is None:
            msg = "Cloud transport requires backend, bucket, and prefix."
            raise ValueError(msg)

        config: dict[str, str] = {}
        if backend == "s3":
            if context["s3_region"] is not None:
                config["region"] = context["s3_region"]
            if context["endpoint"] is not None:
                config["endpoint"] = context["endpoint"]
            store = S3Store(bucket=bucket, **config)
        elif backend == "gcs":
            store = GCSStore(bucket=bucket, **config)
        elif backend == "azure":
            if context["endpoint"] is not None:
                config["endpoint"] = context["endpoint"]
            store = AzureStore(container_name=bucket, **config)
        else:
            msg = f"Unsupported cloud backend: {backend!r}"
            raise ValueError(msg)

        if operation == "snapshot":
            object_key = "/".join(part for part in (prefix, "snapshots", f"{ref}.tar.gz") if part)
            payload = obs.get(store, object_key).bytes()
            destination.mkdir(parents=True)
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
                archive.extractall(destination, filter="data")
        elif operation == "job-file":
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(obs.get(store, ref).bytes())
        else:
            msg = f"Unsupported transport operation: {operation!r}"
            raise ValueError(msg)

    def bootstrap_transport(self) -> str:
        """Return a self-contained obstore transport run through uv.

        The transport depends only on ``obstore`` plus the Python standard
        library; it neither installs nor imports Misen or this workspace.
        Credentials come from the worker's ambient environment or workload
        identity and are never embedded in scheduler-visible shell text.
        """
        if self.config:
            msg = (
                "CloudWorkspace.config cannot be embedded in the worker bootstrap because its shell text may be "
                "visible through the executor or scheduler. Configure worker authentication and obstore options "
                "through the ambient worker environment, or use a custom workspace transport."
            )
            raise ConfigError(msg)
        return render_python_transport(
            self._bootstrap_transport,
            requirements=(f"obstore=={package_version('obstore')}",),
            context={
                "backend": self.backend,
                "bucket": self.bucket,
                "prefix": self._cloud_prefix,
                "endpoint": self.endpoint,
                "s3_region": self.s3_region,
            },
        )

    def _under(self, *parts: str) -> str:
        return _join_prefix(self._cloud_prefix, *parts)

    def lock(self, namespace: Literal["task", "result"], key: str) -> LockLike:
        return ObjectStoreLock(
            store=self._store,
            key=self._under("locks", namespace, key),
            lifetime=30,
            refresh_interval=20,
        )

    def get_temp_dir(self) -> Path:
        return self._cache / "tmp"

    # -- snapshot store -------------------------------------------------
    # One ``snapshots/<key>.tar.gz`` object per content key (a tarball
    # avoids per-file object churn: a staged tree is many small files),
    # unpacked into the per-host cache on fetch. ``publish_snapshot``
    # seeds the local cache from the staged tree so the submitting host
    # never re-downloads its own upload.

    def _snapshot_cache_dir(self, key: str) -> Path:
        return self._cache / "snapshots" / key

    def publish_snapshot(self, key: str, staged_dir: Path) -> None:
        with _cloud_errors(f"Could not publish snapshot {key}", tarfile.TarError):
            try:
                obs.head(self._store, _snapshot_object_key(self._cloud_prefix, key))
            except FileNotFoundError:
                buffer = io.BytesIO()
                with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
                    tar.add(staged_dir, arcname=".")
                obs.put(
                    self._store,
                    _snapshot_object_key(self._cloud_prefix, key),
                    buffer.getbuffer(),
                    mode="overwrite",
                )
            local = self._snapshot_cache_dir(key)
            if not local.exists():
                local.parent.mkdir(parents=True, exist_ok=True)
                with contextlib.suppress(OSError):
                    staged_dir.rename(local)

    def fetch_snapshot(self, key: str) -> Path:
        with _cloud_errors(f"Could not fetch snapshot {key}", tarfile.TarError, passthrough=(FileNotFoundError,)):
            return _download_snapshot(self._store, self._cloud_prefix, key, self._snapshot_cache_dir(key))

    # -- job files ------------------------------------------------------

    def put_job_file(self, submission_id: str, name: str, data: bytes) -> str:
        self._validate_job_file_name(name)
        ref = self._under("job_files", submission_id, name)
        with _cloud_errors(f"Could not publish job file {name!r}"):
            obs.put(self._store, ref, data, mode="overwrite")
        return ref

    def read_job_file(self, submission_id: str, name: str) -> bytes:
        """Read one submission file without hiding a not-yet-published object."""
        self._validate_job_file_name(name)
        ref = self._under("job_files", submission_id, name)
        with _cloud_errors(f"Could not read job file {name!r}", passthrough=(FileNotFoundError,)):
            return bytes(obs.get(self._store, ref).bytes())

    def _get_scratch_dir(self, task: Task) -> Path:
        path = self._cache / "scratch" / task.resolved_hash(workspace=self).b32()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _scratch_dir_remote_prefix(self, task: Task) -> str:
        return self._under("scratch_dirs", task.resolved_hash(workspace=self).b32())

    def start_scratch_dir_sync(self, task: Task) -> None:
        if not task.meta.cache:
            return
        local_path = self.get_scratch_dir(task)
        remote_prefix = self._scratch_dir_remote_prefix(task)
        key = task.resolved_hash(workspace=self).b32()
        with self._lifecycle_lock, self._scratch_dir_lock:
            if self._closed:
                msg = "Cannot start scratch sync on a closed CloudWorkspace."
                raise StorageError(msg)
            existing = self._scratch_dir_syncs.get(key)
            if existing is not None:
                if existing.active:
                    return
                existing.stop(final_upload=True)
                del self._scratch_dir_syncs[key]
            sync = _ScratchDirSync(self._store, local_path, remote_prefix, self.scratch_dir_sync_interval_s)
            with _cloud_errors(f"Could not start scratch sync for {local_path}"):
                sync.restore()
                sync.start()
            self._scratch_dir_syncs[key] = sync

    def finalize_scratch_dir(self, task: Task) -> None:
        if not task.meta.cache:
            return
        key = task.resolved_hash(workspace=self).b32()
        with self._lifecycle_lock, self._scratch_dir_lock:
            sync = self._scratch_dir_syncs.get(key)
            if sync is not None:
                sync.stop(final_upload=True)
                del self._scratch_dir_syncs[key]
            else:
                # A producer may keep writing after close; its own finalizer
                # remains the true commit point and performs an exact mirror.
                _ScratchDirSync(
                    self._store,
                    self.get_scratch_dir(task),
                    self._scratch_dir_remote_prefix(task),
                    self.scratch_dir_sync_interval_s,
                ).finalize_current_tree()

    def _delete_scratch_dir_remote(self, remote_prefix: str) -> None:
        with _cloud_errors(f"Could not remove remote scratch directory {remote_prefix}"):
            _delete_prefix(self._store, f"{remote_prefix}/")

    def remove_scratch_dir(self, task: Task) -> None:
        if not task.meta.cache:
            msg = f"{task} cannot use workspace scratch_dir unless Task.meta.cache == True."
            raise RuntimeError(msg)
        key = task.resolved_hash(workspace=self).b32()
        with self._scratch_dir_lock:
            sync = self._scratch_dir_syncs.get(key)
            if sync is not None:
                sync.stop(final_upload=False)
                del self._scratch_dir_syncs[key]
        self._delete_scratch_dir_remote(self._scratch_dir_remote_prefix(task))
        local_path = self._cache / "scratch" / key
        with _storage_errors(f"Could not remove local scratch directory {local_path}"):
            if local_path.exists():
                shutil.rmtree(local_path)

    def _task_log_paths(self, task: Task, job_id: str) -> tuple[Path, Path, str]:
        """Return ``(writer_path, cache_path, remote_key)`` for a task log.

        ``writer_path`` is the file the executor appends to and the live
        uploader reads chunks from. The read path never writes to it.

        ``cache_path`` is a separate location where reads materialize a
        downloaded copy of the cloud blob. Keeping it distinct prevents the
        cache refresh (which uses an atomic rename) from orphaning the inode
        that an active writer is appending to -- a real risk on shared
        scratch (LocalExecutor and SLURM both share the orchestrator's
        ``.cache/misen`` via cwd).

        Logs are keyed by :meth:`Task.resolved_hash`, so two runs of the
        same task with different dependency results land in distinct log
        directories. Resolving the hash requires every dependency's result
        hash to be cached -- callers that may invoke this before the deps
        complete should expect :class:`CacheError`.
        """
        key = task.resolved_hash(workspace=self).b32()
        writer_dir = self._cache / "task_logs" / key
        cache_dir = self._cache / "task_log_cache" / key
        with _storage_errors(f"Could not prepare task-log directories for {key}"):
            writer_dir.mkdir(parents=True, exist_ok=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
        return (
            writer_dir / f"{job_id}.log",
            cache_dir / f"{job_id}.log",
            self._under("task_logs", key, f"{job_id}.log"),
        )

    def _start_live_upload(self, local_path: Path, remote_key: str) -> None:
        with self._lifecycle_lock, self._live_log_lock:
            if self._closed:
                msg = "Cannot start a log uploader on a closed CloudWorkspace."
                raise StorageError(msg)
            existing = self._live_log_uploaders.get(local_path)
            if existing is not None:
                if existing.active:
                    return
                existing.stop(final_upload=True)
                del self._live_log_uploaders[local_path]
            uploader = _LiveLogUploader(self._store, local_path, remote_key, self.log_flush_interval_s)
            with _cloud_errors(f"Could not start log uploader for {local_path}"):
                uploader.start()
            self._live_log_uploaders[local_path] = uploader

    def _stop_live_upload(self, local_path: Path, remote_key: str | None = None) -> None:
        with self._lifecycle_lock, self._live_log_lock:
            uploader = self._live_log_uploaders.get(local_path)
            if uploader is not None:
                uploader.stop(final_upload=True)
                del self._live_log_uploaders[local_path]
            else:
                # The producer may outlive close; publish its now-final bytes.
                _LiveLogUploader(
                    self._store,
                    local_path,
                    remote_key or self._job_log_remote_key(local_path),
                    self.log_flush_interval_s,
                ).compact()

    def _ensure_local_copy(self, remote_key: str, local_path: Path) -> None:
        tmp_path: Path | None = None
        try:
            with _cloud_errors(f"Could not download log object {remote_key}", passthrough=(FileNotFoundError,)):
                response = obs.get(self._store, remote_key)
                local_path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    "wb", dir=local_path.parent, prefix=f".{local_path.name}.", suffix=".tmp", delete=False
                ) as tmp:
                    tmp_path = Path(tmp.name)
                    for chunk in response.stream():
                        tmp.write(chunk)
                tmp_path.replace(local_path)
                tmp_path = None
        except FileNotFoundError as e:
            msg = f"Object not found: {remote_key}"
            raise FileNotFoundError(msg) from e
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    def _remote_final_info(self, remote_key: str) -> tuple[int, float] | None:
        try:
            with _cloud_errors(f"Could not inspect log object {remote_key}", passthrough=(FileNotFoundError,)):
                meta = obs.head(self._store, remote_key)
        except FileNotFoundError:
            return None
        last_modified = meta["last_modified"]
        return int(meta["size"]), 0.0 if last_modified is None else last_modified.timestamp()

    def _remote_state_info(self, remote_key: str) -> tuple[int, float] | None:
        try:
            with _cloud_errors(f"Could not inspect live-log state for {remote_key}", passthrough=(FileNotFoundError,)):
                resp = obs.get(self._store, f"{remote_key}{_STATE}")
        except FileNotFoundError:
            return None
        try:
            state = msgspec.json.decode(bytes(resp.bytes()), type=dict[str, Any])
            offset = int(state.get("offset", 0))
        except (AttributeError, TypeError, ValueError, msgspec.DecodeError):
            offset = 0
        last_modified = resp.meta["last_modified"]
        return offset, 0.0 if last_modified is None else last_modified.timestamp()

    def _download_log_chunks(self, remote_key: str, local_path: Path, expected_size: int | None) -> bool:
        with _cloud_errors(f"Could not download live-log chunks for {remote_key}"):
            prefix = f"{remote_key}{_CHUNKS}/"
            chunks: list[tuple[int, str]] = []
            for batch in obs.list(self._store, prefix=prefix):
                for entry in batch:
                    name = entry["path"][len(prefix) :]
                    if "/" not in name and name.endswith(".chunk"):
                        with contextlib.suppress(ValueError):
                            chunks.append((int(name.removesuffix(".chunk")), entry["path"]))
            if not chunks:
                return False

            local_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    "wb", dir=local_path.parent, prefix=f".{local_path.name}.", suffix=".tmp", delete=False
                ) as tmp:
                    tmp_path = Path(tmp.name)
                    expected_offset = 0
                    for offset, path in sorted(chunks):
                        if expected_size is not None and offset >= expected_size:
                            continue
                        if offset != expected_offset:
                            return False
                        try:
                            data = bytes(obs.get(self._store, path).bytes())
                        except FileNotFoundError:
                            return False
                        if expected_size is not None and expected_offset + len(data) > expected_size:
                            return False
                        tmp.write(data)
                        expected_offset += len(data)
                    if expected_size is not None and expected_offset != expected_size:
                        return False
                tmp_path.replace(local_path)
            finally:
                if tmp_path is not None:
                    tmp_path.unlink(missing_ok=True)
            return True

    def _ensure_log_local(self, remote_key: str, local_path: Path) -> None:
        final_info = self._remote_final_info(remote_key)
        state_info = self._remote_state_info(remote_key)

        if final_info is not None and (state_info is None or final_info[1] >= state_info[1]):
            self._ensure_local_copy(remote_key, local_path)
            return
        if state_info is not None and self._download_log_chunks(remote_key, local_path, expected_size=state_info[0]):
            return
        if final_info is not None:
            self._ensure_local_copy(remote_key, local_path)
            return
        msg = f"Object not found: {remote_key}"
        raise FileNotFoundError(msg)

    def _refresh_log_local(self, remote_key: str, local_path: Path) -> None:
        with self._live_log_lock:
            if local_path in self._live_log_uploaders:
                return

        if not local_path.exists():
            self._ensure_log_local(remote_key, local_path)
            return

        local_size = local_path.stat().st_size
        final_info = self._remote_final_info(remote_key)
        state_info = self._remote_state_info(remote_key)

        if final_info is not None and (state_info is None or final_info[1] >= state_info[1]):
            if state_info is not None or final_info[0] != local_size:
                self._ensure_local_copy(remote_key, local_path)
            return
        if state_info is not None and (final_info is None or state_info[1] > final_info[1]):
            if state_info[0] != local_size:
                self._ensure_log_local(remote_key, local_path)
            return
        if final_info is not None and final_info[0] != local_size:
            self._ensure_local_copy(remote_key, local_path)

    def _remote_logs(self, prefix: str) -> dict[str, float]:
        with _cloud_errors(f"Could not list remote logs under {prefix}"):
            logs: dict[str, float] = {}
            for batch in obs.list(self._store, prefix=prefix):
                for entry in batch:
                    path = entry["path"]
                    if path.endswith(".log"):
                        remote_key = path
                    elif path.endswith(_STATE):
                        remote_key = path[: -len(_STATE)]
                    else:
                        continue
                    ts = 0.0 if entry["last_modified"] is None else entry["last_modified"].timestamp()
                    logs[remote_key] = max(logs.get(remote_key, 0.0), ts)
            return logs

    def _collect_logs(self, remote_prefix: str, writer_dir: Path, cache_dir: Path) -> dict[str, Path]:
        """Materialize remote logs without replacing fresher local writers."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        for remote_key in self._remote_logs(remote_prefix):
            filename = remote_key.rsplit("/", 1)[-1]
            writer_path = writer_dir / filename
            if writer_path.exists():
                paths[filename] = writer_path
                continue
            cache_path = cache_dir / filename
            self._refresh_log_local(remote_key, cache_path)
            paths[filename] = cache_path
        if writer_dir.exists():
            paths.update((path.name, path) for path in writer_dir.iterdir() if path.is_file())
        return paths

    def get_task_log(self, task: Task, job_id: str | None = None) -> Path:
        writer_path, _, remote_key = self._task_log_paths(task, job_id or "0")
        self._start_live_upload(writer_path, remote_key)
        return writer_path

    def finalize_task_log(self, task: Task, job_id: str | None = None) -> None:
        writer_path, _, remote_key = self._task_log_paths(task, job_id or "0")
        self._stop_live_upload(writer_path, remote_key)

    def read_task_log(self, task: Task, job_id: str | None = None) -> TextIO:
        with _cloud_errors(
            f"Could not read a task log from cloud workspace {self.workspace_id}",
            passthrough=(FileNotFoundError,),
        ):
            if job_id is not None:
                return self._open_task_log_for_read(task, job_id)

            key = task.resolved_hash(workspace=self).b32()
            writer_dir = self._cache / "task_logs" / key
            candidates = [(p.stat().st_mtime, p.stem) for p in writer_dir.glob("*.log")]
            for remote_key, ts in self._remote_logs(self._under("task_logs", key) + "/").items():
                filename = remote_key.rsplit("/", 1)[-1]
                if filename.endswith(".log"):
                    candidates.append((ts, filename.removesuffix(".log")))

            if not candidates:
                msg = f"No logs found for task {task} in workspace {self.backend}://{self.bucket!r}."
                raise FileNotFoundError(msg)
            return self._open_task_log_for_read(task, max(candidates, key=lambda item: item[0])[1])

    def task_log_iter(self, task: Task) -> Iterator[tuple[str, Path]]:
        with _cloud_errors(
            f"Could not list task logs from cloud workspace {self.workspace_id}",
            passthrough=(FileNotFoundError,),
        ):
            return iter(self._task_log_entries(task).items())

    def _task_log_entries(self, task: Task) -> dict[str, Path]:
        key = task.resolved_hash(workspace=self).b32()
        writer_dir = self._cache / "task_logs" / key
        paths = self._collect_logs(
            self._under("task_logs", key) + "/",
            writer_dir,
            self._cache / "task_log_cache" / key,
        )
        return {filename.removesuffix(".log"): path for filename, path in paths.items() if filename.endswith(".log")}

    def _open_task_log_for_read(self, task: Task, job_id: str) -> TextIO:
        writer_path, cache_path, remote_key = self._task_log_paths(task, job_id)
        # Writer files visible on local FS (this process or a same-FS sibling
        # like a LocalExecutor subprocess or a SLURM worker on shared scratch)
        # always have the freshest bytes. Read-only opens never disturb the
        # writer, so prefer the writer file when present.
        if writer_path.exists():
            return writer_path.open("r", encoding="utf-8")
        self._refresh_log_local(remote_key, cache_path)
        return cache_path.open("r", encoding="utf-8")

    def _job_log_remote_key(self, local_path: Path) -> str:
        return self._under("job_logs", local_path.name)

    def finalize_job_log(self, local_path: Path) -> None:
        with _storage_errors(f"Could not inspect job log {local_path}"):
            exists = local_path.exists()
        if exists:
            _LiveLogUploader(
                self._store,
                local_path,
                self._job_log_remote_key(local_path),
                self.log_flush_interval_s,
            ).compact()

    @contextlib.contextmanager
    def streaming_job_log(self, local_path: Path) -> Iterator[None]:
        self._start_live_upload(local_path, self._job_log_remote_key(local_path))
        with _cleanup_on_exit(lambda: self._stop_live_upload(local_path), f"finalizing job log {local_path}"):
            yield

    def job_log_iter(self, work_unit: WorkUnit | None = None) -> Iterator[Path]:
        with _cloud_errors(
            f"Could not read job logs from cloud workspace {self.workspace_id}",
            passthrough=(FileNotFoundError,),
        ):
            return self._job_log_iter(work_unit)

    def _job_log_iter(self, work_unit: WorkUnit | None) -> Iterator[Path]:
        writer_dir = self._cache / "job_logs"
        paths = self._collect_logs(
            self._under("job_logs") + "/",
            writer_dir,
            self._cache / "job_log_cache",
        )

        if work_unit is None:
            return iter(paths.values())
        prefix = f"{work_unit.root.task_hash().b32()}_"
        return iter(p for filename, p in paths.items() if filename.startswith(prefix))

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return

            errors: list[tuple[str, BaseException]] = []

            def stop_all(resources: dict[Any, Any], label: str) -> None:
                for key, resource in list(resources.items()):
                    try:
                        resource.stop(final_upload=True)
                    except BaseException as exc:  # noqa: BLE001 -- stop every resource
                        errors.append((f"{label} {key}", exc))
                    else:
                        del resources[key]

            with self._live_log_lock:
                stop_all(self._live_log_uploaders, "log uploader for")

            with self._scratch_dir_lock:
                stop_all(self._scratch_dir_syncs, "scratch sync")

            if errors:
                primary_label, primary_error = errors[0]
                primary_error.add_note(f"CloudWorkspace.close failed while stopping {primary_label}.")
                for label, error in errors[1:]:
                    primary_error.add_note(f"Additionally, stopping {label} failed: {type(error).__name__}: {error}")
                raise primary_error

            self._closed = True
