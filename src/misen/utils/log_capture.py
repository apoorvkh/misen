"""Capture process stdout/stderr (including C-level writes) into a stream."""

from __future__ import annotations

import codecs
import contextlib
import ctypes
import io
import os
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, TextIO, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

T = TypeVar("T")


def _try(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T | None:
    """Call function and return ``None`` on any exception."""
    try:
        return fn(*args, **kwargs)
    except Exception:  # noqa: BLE001
        return None


def _fflush_all() -> None:
    """Flush all C stdio buffers for current process."""
    ctypes.CDLL(None).fflush(None)


def _write(text: str, lock: threading.Lock, target: TextIO) -> None:
    """Write text to target stream under lock and flush."""
    if not text:
        return
    with lock:
        target.write(text)
        _try(target.flush)


def _try_fsync(target: TextIO) -> None:
    """Call ``os.fsync`` on ``target`` if it's a real file; no-op otherwise."""
    if callable(fileno := _try(getattr, target, "fileno")) and (fd := _try(fileno)) is not None:
        _try(os.fsync, fd)


def _make_decoder(enc: str) -> codecs.IncrementalDecoder:
    """Create incremental decoder for the given encoding."""
    return codecs.getincrementaldecoder(enc)(errors="replace")


# Bound NFS tail-reader staleness without paying for an fsync on every chunk.
_LOG_FSYNC_INTERVAL_S = 0.1


def _drain_and_write(
    pipe_fd: int,
    decoder: codecs.IncrementalDecoder,
    lock: threading.Lock,
    targets: Sequence[TextIO],
    *,
    deadline: float | None = None,
    fsync_interval_s: float = _LOG_FSYNC_INTERVAL_S,
) -> None:
    """Drain decoded pipe data to all targets, periodically fsyncing them."""
    last_fsync_at = time.monotonic()
    while deadline is None or time.monotonic() < deadline:
        try:
            chunk = os.read(pipe_fd, 8192)
        except (BlockingIOError, OSError):
            break
        if not chunk:
            break
        text = decoder.decode(chunk)
        for target in targets:
            _write(text, lock=lock, target=target)
        now = time.monotonic()
        if now - last_fsync_at >= fsync_interval_s:
            for target in targets:
                _try_fsync(target)
            last_fsync_at = now

    tail = decoder.decode(b"", final=True)
    for target in targets:
        _write(tail, lock=lock, target=target)
        _try_fsync(target)


def _wrap_fd(fd: int, enc: str, *, closefd: bool = False) -> TextIO:
    """Wrap file descriptor in a line-buffered text writer."""
    raw = io.FileIO(fd, mode="w", closefd=closefd)
    buf = io.BufferedWriter(raw)
    return io.TextIOWrapper(buf, encoding=enc, errors="replace", line_buffering=True, write_through=True)


def _validate_capture_target(target: TextIO, old_stdout: TextIO, old_stderr: TextIO) -> None:
    """Validate capture target to avoid recursive stdout/stderr loops."""
    if target is old_stdout or target is old_stderr:
        msg = "capture_all_output: `target` must not be sys.stdout/sys.stderr (would recurse)"
        raise ValueError(msg)

    fileno = _try(getattr, target, "fileno")
    if callable(fileno):
        fd = _try(fileno)
        if fd in (1, 2):
            msg = "capture_all_output: `target` must not write to fd 1/2 (would recurse)"
            raise ValueError(msg)


@contextlib.contextmanager
def capture_all_output(
    target: TextIO,
    timeout: float = 10.0,
    *,
    tee_to_stdout: bool = False,
) -> Iterator[None]:
    """Capture fd 1/2 writes into ``target``, optionally teeing to stdout.

    Exit waits up to ``timeout`` to drain, then performs a nonblocking
    best-effort drain. ``target`` cannot itself point to fd 1 or 2.
    """
    encoding: str = getattr(sys.stdout, "encoding", None) or "utf-8"

    old_stdout, old_stderr = sys.stdout, sys.stderr
    lock = threading.Lock()

    # Guard against recursion/feedback loops.
    _validate_capture_target(target, old_stdout, old_stderr)

    for stream in (old_stdout, old_stderr):
        _try(stream.flush)
    _try(_fflush_all)

    # Save original inheritability of fds 1/2 and make them non-inheritable (best-effort)
    stdio_fds = (1, 2)
    inheritability = tuple(_try(os.get_inheritable, fd) for fd in stdio_fds)
    for fd in stdio_fds:
        _try(os.set_inheritable, fd, False)  # noqa: FBT003

    saved_fds = tuple(os.dup(fd) for fd in stdio_fds)
    tee_stdout: TextIO | None = _wrap_fd(os.dup(saved_fds[0]), encoding, closefd=True) if tee_to_stdout else None

    rfd, wfd = os.pipe()
    os.set_inheritable(wfd, False)  # noqa: FBT003
    os.set_inheritable(rfd, False)  # noqa: FBT003

    targets: tuple[TextIO, ...] = (target,) if tee_stdout is None else (target, tee_stdout)

    def reader() -> None:
        """Read redirected pipe bytes and write decoded text to targets."""
        dec = _make_decoder(encoding)
        try:
            _drain_and_write(rfd, dec, lock, targets)
        except (OSError, ValueError) as exc:
            for t in targets:
                _try(t.write, f"[misen] log capture reader stopped early: {exc}\n")
        finally:
            _try(os.close, rfd)

    t = threading.Thread(target=reader, name="capture_all_output", daemon=True)
    t.start()

    # Redirect fd 1/2 -> pipe
    try:
        for fd in stdio_fds:
            os.dup2(wfd, fd)
        # Re-apply non-inheritable on 1/2 after dup2 (belt & suspenders)
        for fd in stdio_fds:
            _try(os.set_inheritable, fd, False)  # noqa: FBT003
    finally:
        _try(os.close, wfd)

    new_stdout = _wrap_fd(1, encoding)
    new_stderr = _wrap_fd(2, encoding)
    sys.stdout, sys.stderr = new_stdout, new_stderr

    try:
        yield
    finally:
        # Flush while still redirected
        for stream in (sys.stdout, sys.stderr):
            _try(stream.flush)
        _try(_fflush_all)

        # Close wrappers so their buffers go into the pipe
        for stream in (new_stdout, new_stderr):
            _try(stream.close)

        # Restore original fds 1/2 (closing this proc's pipe write ends)
        try:
            for fd, saved_fd in zip(stdio_fds, saved_fds, strict=True):
                os.dup2(saved_fd, fd)
        finally:
            for saved_fd in saved_fds:
                _try(os.close, saved_fd)

        sys.stdout, sys.stderr = old_stdout, old_stderr

        # Restore original inheritability for fds 1/2 (best-effort)
        for fd, inherited in zip(stdio_fds, inheritability, strict=True):
            if inherited is not None:
                _try(os.set_inheritable, fd, bool(inherited))

        deadline = time.monotonic() + max(0.0, float(timeout))
        t.join(timeout=max(0.0, deadline - time.monotonic()))

        if t.is_alive():
            # Best-effort: drain what is currently available, then stop.
            _try(os.set_blocking, rfd, blocking=False)
            _try(_drain_and_write, rfd, _make_decoder(encoding), lock, targets, deadline=deadline)

            # Force reader to exit; since it's daemon, we still won't hang regardless.
            _try(os.close, rfd)
            t.join(timeout=0.2)

        _try(target.flush)
        if tee_stdout is not None:
            _try(tee_stdout.flush)
            _try(tee_stdout.close)
