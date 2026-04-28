# ruff: noqa: D102, D105, D107
"""Object-store-backed workspace for S3, GCS, and Azure Blob.

This backend stores hash indices, result payloads, task logs, and job logs in a
cloud object store through ``obstore``. Work directories and actively-written
logs stay on local scratch storage; live logs are uploaded as offset chunks and
compacted into a single ``.log`` object on close.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import logging
import shutil
import tempfile
import threading
from collections.abc import Iterator, MutableMapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, Self, TextIO, TypeAlias, TypeVar, cast

import msgspec
import obstore as obs
from obstore.store import AzureStore, GCSStore, S3Store
from xxhash import xxh3_64_hexdigest

from misen.utils.hashing import Hash, ResolvedTaskHash, ResultHash, TaskHash
from misen.utils.locks import ObjectStoreLock
from misen.utils.serde import MANIFEST_FILENAME
from misen.workspace import Workspace

if TYPE_CHECKING:
    from misen.tasks import Task
    from misen.utils.locks import LockLike
    from misen.utils.work_unit import WorkUnit

__all__ = ("CloudBackend", "CloudWorkspace", "ObstoreMapping", "ObstoreResultStore")


KT = TypeVar("KT", bound=Hash)
VT = TypeVar("VT", bound=Hash)
CloudBackend: TypeAlias = Literal["s3", "gcs", "azure"]
_CHUNKS = ".chunks"
_STATE = ".state.json"
logger = logging.getLogger(__name__)


class ObstoreMapping(MutableMapping[KT, VT], Generic[KT, VT]):
    """Typed hash->hash mapping stored as one object per key."""

    _key_type: type[KT]
    _value_type: type[VT]
    __slots__ = ("_prefix", "_store")

    def __class_getitem__(cls, item: tuple[type[KT], type[VT]]) -> type[Self]:
        key_t, val_t = item
        return cast(
            "type[Self]",
            type(
                f"{cls.__name__}[{key_t.__name__},{val_t.__name__}]",
                (cls,),
                {"_key_type": key_t, "_value_type": val_t, "__module__": cls.__module__},
            ),
        )

    def __init__(self, store: Any, prefix: str) -> None:
        if not hasattr(self, "_key_type") or not hasattr(self, "_value_type"):
            msg = "Construct as ObstoreMapping[KeyType, ValueType](...)"
            raise TypeError(msg)
        self._store = store
        self._prefix = prefix.rstrip("/")

    def __getitem__(self, key: KT) -> VT:
        try:
            return self._value_type.decode(bytes(obs.get(self._store, f"{self._prefix}/{key.b32()}").bytes()))
        except FileNotFoundError as e:
            raise KeyError(key) from e

    def __setitem__(self, key: KT, value: VT) -> None:
        obs.put(self._store, f"{self._prefix}/{key.b32()}", value.encode(), mode="overwrite")

    def __delitem__(self, key: KT) -> None:
        path = f"{self._prefix}/{key.b32()}"
        try:
            obs.head(self._store, path)
        except FileNotFoundError as e:
            raise KeyError(key) from e
        obs.delete(self._store, path)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, self._key_type):
            return False
        try:
            obs.head(self._store, f"{self._prefix}/{key.b32()}")
        except FileNotFoundError:
            return False
        return True

    def __iter__(self) -> Iterator[KT]:
        prefix = f"{self._prefix}/"
        for batch in obs.list(self._store, prefix=prefix):
            for entry in batch:
                rel = entry["path"][len(prefix) :]
                if not rel:
                    continue
                try:
                    yield self._key_type(int.from_bytes(base64.b32decode(rel + "=" * (-len(rel) % 8)), "big"))
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
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, ResultHash):
            return False
        if (self._cache_dir / key.b32()).exists():
            return True
        try:
            obs.head(self._store, f"{self._prefix}/{key.b32()}/{MANIFEST_FILENAME}")
        except FileNotFoundError:
            return False
        return True

    def __getitem__(self, key: ResultHash) -> Path:
        local = self._cache_dir / key.b32()
        if local.exists():
            return local

        remote_prefix = f"{self._prefix}/{key.b32()}"
        try:
            obs.head(self._store, f"{remote_prefix}/{MANIFEST_FILENAME}")
        except FileNotFoundError as e:
            raise KeyError(key) from e

        tmp = Path(tempfile.mkdtemp(dir=self._cache_dir, prefix=f".{key.b32()}.", suffix=".tmp"))
        try:
            for batch in obs.list(self._store, prefix=f"{remote_prefix}/"):
                for entry in batch:
                    rel = entry["path"][len(remote_prefix) + 1 :]
                    if not rel:
                        continue
                    target = tmp / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(bytes(obs.get(self._store, entry["path"]).bytes()))
            tmp.rename(local)
        except FileExistsError:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return local

    def __setitem__(self, key: ResultHash, value: Path) -> None:
        remote_prefix = f"{self._prefix}/{key.b32()}"
        try:
            obs.head(self._store, f"{remote_prefix}/{MANIFEST_FILENAME}")
        except FileNotFoundError:
            pass
        else:
            return

        manifest = value / MANIFEST_FILENAME
        if not manifest.is_file():
            raise FileNotFoundError(manifest)

        for path in sorted(p for p in value.rglob("*") if p.is_file() and p != manifest):
            with path.open("rb") as f:
                obs.put(self._store, f"{remote_prefix}/{path.relative_to(value).as_posix()}", f, mode="overwrite")
        with manifest.open("rb") as f:
            obs.put(self._store, f"{remote_prefix}/{MANIFEST_FILENAME}", f, mode="overwrite")

    def __delitem__(self, key: ResultHash) -> None:
        prefix = f"{self._prefix}/{key.b32()}/"
        keys = [entry["path"] for batch in obs.list(self._store, prefix=prefix) for entry in batch]
        local = self._cache_dir / key.b32()
        if not keys and not local.exists():
            raise KeyError(key)
        if keys:
            obs.delete(self._store, keys)
        shutil.rmtree(local, ignore_errors=True)

    def __iter__(self) -> Iterator[ResultHash]:
        prefix = f"{self._prefix}/"
        b32_len = len(ResultHash(0).b32())
        for batch in obs.list(self._store, prefix=prefix):
            for entry in batch:
                rel = entry["path"][len(prefix) :]
                head, sep, tail = rel.partition("/")
                if sep != "/" or tail != MANIFEST_FILENAME or len(head) != b32_len:
                    continue
                try:
                    yield ResultHash(int.from_bytes(base64.b32decode(head + "=" * (-len(head) % 8)), "big"))
                except (binascii.Error, ValueError, TypeError):
                    continue

    def __len__(self) -> int:
        return sum(1 for _ in self)


class _LiveLogUploader:
    """Upload appended log chunks in the background, then compact on close."""

    __slots__ = ("_interval_s", "_local_path", "_remote_key", "_stop", "_store", "_thread", "_uploaded_offset")

    def __init__(self, store: Any, local_path: Path, remote_key: str, interval_s: float) -> None:
        self._store = store
        self._local_path = local_path
        self._remote_key = remote_key
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._uploaded_offset = 0

    def start(self) -> None:
        self._stop.clear()
        self._delete_live_objects()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"misen-log-upload[{self._remote_key}]")
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                size = self._local_path.stat().st_size
            except FileNotFoundError:
                continue
            if size < self._uploaded_offset:
                self._delete_live_objects()
                self._uploaded_offset = 0
            if size <= self._uploaded_offset:
                continue

            offset = self._uploaded_offset
            try:
                with self._local_path.open("rb") as f:
                    f.seek(offset)
                    payload = f.read()
                if not payload:
                    continue
                new_offset = offset + len(payload)
                obs.put(self._store, f"{self._remote_key}{_CHUNKS}/{offset:020d}.chunk", payload, mode="overwrite")
                obs.put(
                    self._store,
                    f"{self._remote_key}{_STATE}",
                    msgspec.json.encode({"offset": new_offset, "closed": False}),
                    mode="overwrite",
                )
                self._uploaded_offset = new_offset
            except Exception:
                logger.exception("Live chunk upload failed for %s -> %s.", self._local_path, self._remote_key)

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
        if not self._local_path.exists():
            return
        try:
            with self._local_path.open("rb") as f:
                obs.put(self._store, self._remote_key, f, mode="overwrite")
            self._uploaded_offset = self._local_path.stat().st_size
        except Exception:
            logger.exception("Final log compaction failed for %s -> %s.", self._local_path, self._remote_key)
            return
        try:
            self._delete_live_objects()
        except Exception:
            logger.exception("Compacted %s, but live-log cleanup failed for %s.", self._local_path, self._remote_key)

    def stop(self, *, final_upload: bool = True) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=max(self._interval_s * 2, 2.0))
        self._thread = None
        if final_upload:
            self.compact()


class CloudWorkspace(Workspace):
    """Workspace backed by S3, GCS, or Azure Blob via obstore."""

    backend: CloudBackend
    bucket: str
    prefix: str = ""
    endpoint: str | None = None
    s3_region: str | None = None
    config: dict[str, str] = msgspec.field(default_factory=dict)
    scratch_dir: str = ".cache/misen"
    log_flush_interval_s: float = 1.0

    def __post_init__(self) -> None:
        if self.log_flush_interval_s <= 0:
            msg = "log_flush_interval_s must be positive"
            raise ValueError(msg)
        if self.s3_region is not None and self.backend != "s3":
            msg = f"s3_region is only supported for backend='s3', got backend={self.backend!r}."
            raise ValueError(msg)
        if self.endpoint is not None and self.backend == "gcs":
            msg = "endpoint is not supported for backend='gcs'."
            raise ValueError(msg)

        self._store = self._build_store()
        self._cloud_prefix = self.prefix.strip("/")
        # Append a deterministic id so distinct workspaces never share scratch.
        # Two workspaces with identical identity-affecting fields collapse to
        # the same subdir, which is exactly when sharing is safe.
        self._scratch = Path(self.scratch_dir) / self.workspace_id
        for subdir in ("tmp", "work", "task_logs", "task_log_cache", "job_logs", "job_log_cache", "results_cache"):
            (self._scratch / subdir).mkdir(parents=True, exist_ok=True)
        self._live_log_uploaders: dict[Path, _LiveLogUploader] = {}
        self._live_log_lock = threading.Lock()

        super()._post_init(
            resolved_hash_cache=ObstoreMapping[TaskHash, ResolvedTaskHash](
                self._store,
                self._under("resolved_hash_cache"),
            ),
            result_hash_cache=ObstoreMapping[ResolvedTaskHash, ResultHash](
                self._store,
                self._under("result_hash_cache"),
            ),
            result_store=ObstoreResultStore(self._store, self._under("results"), self._scratch / "results_cache"),
        )
        logger.info(
            "Initialized CloudWorkspace id=%s backend=%s bucket=%s scratch=%s.",
            self.workspace_id,
            self.backend,
            self.bucket,
            self._scratch,
        )

    @property
    def workspace_id(self) -> str:
        """Short deterministic id derived from identity-affecting fields.

        Two workspaces with the same ``(backend, bucket, prefix, endpoint,
        s3_region)`` produce the same id and may safely share local scratch;
        any other pair produces distinct ids.
        """
        payload = msgspec.json.encode(
            (self.backend, self.bucket, self.prefix, self.endpoint, self.s3_region)
        )
        return xxh3_64_hexdigest(payload)

    def _build_store(self) -> Any:
        cfg = cast("dict[str, Any]", dict(self.config))
        explicit: dict[str, Any] = {}
        if self.backend == "s3":
            if self.s3_region is not None:
                explicit["region"] = self.s3_region
            if self.endpoint is not None:
                explicit["endpoint"] = self.endpoint
        elif self.backend == "azure":
            if self.endpoint is not None:
                explicit["endpoint"] = self.endpoint
        for key, value in explicit.items():
            if key in cfg:
                msg = f"{key!r} cannot appear in both config and the dedicated field."
                raise ValueError(msg)
            cfg[key] = value

        if self.backend == "s3":
            return S3Store(bucket=self.bucket, **cfg)
        if self.backend == "gcs":
            return GCSStore(bucket=self.bucket, **cfg)
        if self.backend == "azure":
            return AzureStore(container_name=self.bucket, **cfg)
        msg = f"Unsupported cloud backend: {self.backend!r}"
        raise ValueError(msg)

    def _under(self, *parts: str) -> str:
        return "/".join(part for part in (self._cloud_prefix, *(p.strip("/") for p in parts if p)) if part)

    def lock(self, namespace: Literal["task", "result"], key: str) -> LockLike:
        return ObjectStoreLock(
            store=self._store,
            key=self._under("locks", namespace, key),
            lifetime=30,
            refresh_interval=20,
        )

    def get_temp_dir(self) -> Path:
        return self._scratch / "tmp"

    def _get_work_dir(self, task: Task) -> Path:
        path = self._scratch / "work" / task.resolved_hash(workspace=self).b32()
        path.mkdir(parents=True, exist_ok=True)
        return path

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
        writer_dir = self._scratch / "task_logs" / key
        cache_dir = self._scratch / "task_log_cache" / key
        writer_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return (
            writer_dir / f"{job_id}.log",
            cache_dir / f"{job_id}.log",
            self._under("task_logs", key, f"{job_id}.log"),
        )

    def _start_live_upload(self, local_path: Path, remote_key: str) -> None:
        with self._live_log_lock:
            if local_path in self._live_log_uploaders:
                return
            uploader = _LiveLogUploader(self._store, local_path, remote_key, self.log_flush_interval_s)
            uploader.start()
            self._live_log_uploaders[local_path] = uploader

    def _stop_live_upload(self, local_path: Path) -> None:
        with self._live_log_lock:
            uploader = self._live_log_uploaders.pop(local_path, None)
        if uploader is not None:
            uploader.stop(final_upload=True)

    def _ensure_local_copy(self, remote_key: str, local_path: Path) -> None:
        try:
            data = bytes(obs.get(self._store, remote_key).bytes())
        except FileNotFoundError as e:
            msg = f"Object not found: {remote_key}"
            raise FileNotFoundError(msg) from e
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)

    def _remote_final_info(self, remote_key: str) -> tuple[int, float] | None:
        try:
            meta = obs.head(self._store, remote_key)
        except FileNotFoundError:
            return None
        last_modified = meta["last_modified"]
        return int(meta["size"]), 0.0 if last_modified is None else last_modified.timestamp()

    def _remote_state_info(self, remote_key: str) -> tuple[int, float] | None:
        try:
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
        prefix = f"{remote_key}{_CHUNKS}/"
        chunks: list[tuple[int, str]] = []
        for batch in obs.list(self._store, prefix=prefix):
            for entry in batch:
                name = entry["path"][len(prefix) :]
                if "/" in name or not name.endswith(".chunk"):
                    continue
                with contextlib.suppress(ValueError):
                    chunks.append((int(name.removesuffix(".chunk")), entry["path"]))
        if not chunks:
            return False

        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "wb",
                dir=local_path.parent,
                prefix=f".{local_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
                expected_offset = 0
                for offset, path in sorted(chunks):
                    if offset != expected_offset:
                        return False
                    try:
                        data = bytes(obs.get(self._store, path).bytes())
                    except FileNotFoundError:
                        return False
                    tmp.write(data)
                    expected_offset += len(data)
                if expected_size is not None and expected_offset != expected_size:
                    return False
            tmp_path.replace(local_path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()
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

    def get_task_log(self, task: Task, job_id: str | None = None) -> Path:
        writer_path, _, remote_key = self._task_log_paths(task, job_id or "0")
        self._start_live_upload(writer_path, remote_key)
        return writer_path

    def finalize_task_log(self, task: Task, job_id: str | None = None) -> None:
        writer_path, _, _ = self._task_log_paths(task, job_id or "0")
        self._stop_live_upload(writer_path)

    def read_task_log(self, task: Task, job_id: str | None = None) -> TextIO:
        if job_id is not None:
            return self._open_task_log_for_read(task, job_id)

        key = task.resolved_hash(workspace=self).b32()
        writer_dir = self._scratch / "task_logs" / key
        remote_logs = self._remote_logs(self._under("task_logs", key) + "/")
        candidates: list[tuple[float, str]] = []
        if writer_dir.exists():
            candidates.extend((p.stat().st_mtime, p.stem) for p in writer_dir.glob("*.log"))
        for remote_key, ts in remote_logs.items():
            filename = remote_key.rsplit("/", 1)[-1]
            if filename.endswith(".log"):
                candidates.append((ts, filename[: -len(".log")]))

        if not candidates:
            msg = f"No logs found for task {task} in workspace {self.backend}://{self.bucket!r}."
            raise FileNotFoundError(msg)

        _, latest_job_id = max(candidates, key=lambda item: item[0])
        return self._open_task_log_for_read(task, latest_job_id)

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
        if local_path.exists():
            _LiveLogUploader(
                self._store,
                local_path,
                self._job_log_remote_key(local_path),
                self.log_flush_interval_s,
            ).compact()

    @contextlib.contextmanager
    def streaming_job_log(self, local_path: Path) -> Iterator[None]:
        self._start_live_upload(local_path, self._job_log_remote_key(local_path))
        try:
            yield
        finally:
            self._stop_live_upload(local_path)

    def job_log_iter(self, work_unit: WorkUnit | None = None) -> Iterator[Path]:
        writer_dir = self._scratch / "job_logs"
        cache_dir = self._scratch / "job_log_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Refresh remote logs into the read cache. The cache is distinct from
        # the writer dir, so a refresh cannot clobber a writer that's still
        # appending to ``writer_dir/<filename>``.
        paths: dict[str, Path] = {}
        for remote_key in self._remote_logs(self._under("job_logs") + "/"):
            filename = remote_key.rsplit("/", 1)[-1]
            cache_path = cache_dir / filename
            self._refresh_log_local(remote_key, cache_path)
            paths[filename] = cache_path

        # Local writer files (this process or a same-FS sibling) override the
        # cache for the same filename: they always have the freshest bytes,
        # and read-only opens by callers don't disturb them.
        if writer_dir.exists():
            for p in writer_dir.iterdir():
                if p.is_file():
                    paths[p.name] = p

        if work_unit is None:
            return iter(paths.values())
        prefix = f"{work_unit.root.task_hash().b32()}_"
        return iter(p for filename, p in paths.items() if filename.startswith(prefix))

    def close(self) -> None:
        with self._live_log_lock:
            uploaders = list(self._live_log_uploaders.values())
            self._live_log_uploaders.clear()
        for uploader in uploaders:
            uploader.stop(final_upload=True)
        with contextlib.suppress(AttributeError):
            del self._store
