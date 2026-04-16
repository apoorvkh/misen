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


def _make_decoder(enc: str) -> codecs.IncrementalDecoder:
    """Create incremental decoder for the given encoding."""
    return codecs.getincrementaldecoder(enc)(errors="replace")


def _drain_and_write(
    pipe_fd: int,
    decoder: codecs.IncrementalDecoder,
    lock: threading.Lock,
    targets: Sequence[TextIO],
    *,
    deadline: float | None = None,
) -> None:
    """Read from ``pipe_fd`` until EOF/error and write decoded text to targets.

    Stops when the pipe returns EOF, raises ``OSError``/``BlockingIOError``, or
    ``deadline`` (a ``time.monotonic`` timestamp) is reached. Always flushes the
    decoder's trailing bytes to every target before returning.
    """
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

    tail = decoder.decode(b"", final=True)
    for target in targets:
        _write(tail, lock=lock, target=target)


def _wrap_fd(fd: int, enc: str, *, closefd: bool = False) -> TextIO:
    """Wrap file descriptor in a line-buffered text writer."""
    raw = io.FileIO(fd, mode="w", closefd=closefd)
    buf = io.BufferedWriter(raw)
    return io.TextIOWrapper(
        buf,
        encoding=enc,
        errors="replace",
        line_buffering=True,
        write_through=True,
    )


def _join_until(th: threading.Thread, deadline: float) -> None:
    """Join thread until deadline timestamp."""
    th.join(timeout=max(0.0, deadline - time.monotonic()))


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
    """Capture writes to fds 1/2 and tee them into ``target``.

    Exit behavior:
      - waits up to `timeout` seconds to drain cleanly
      - if not drained, best-effort drains what's currently available and then stops
        (never hangs, may lose trailing output from stray inheritors).

    Args:
        target: Destination stream for captured output.
        timeout: Drain timeout in seconds on context exit.
        tee_to_stdout: Whether to mirror captured output to original stdout.

    Yields:
        ``None`` while output redirection is active.

    Raises:
        ValueError: If ``target`` points to stdout/stderr and would recurse.
    """
    encoding: str = getattr(sys.stdout, "encoding", None) or "utf-8"

    old_stdout, old_stderr = sys.stdout, sys.stderr
    lock = threading.Lock()

    # Guard against recursion/feedback loops.
    _validate_capture_target(target, old_stdout, old_stderr)

    # Flush before redirecting
    _try(old_stdout.flush)
    _try(old_stderr.flush)
    _try(_fflush_all)

    # Save original inheritability of fds 1/2 and make them non-inheritable (best-effort)
    inh1 = _try(os.get_inheritable, 1)
    inh2 = _try(os.get_inheritable, 2)
    _try(os.set_inheritable, 1, False)  # noqa: FBT003
    _try(os.set_inheritable, 2, False)  # noqa: FBT003

    saved_fd1 = os.dup(1)
    saved_fd2 = os.dup(2)
    tee_stdout: TextIO | None = _wrap_fd(os.dup(saved_fd1), encoding, closefd=True) if tee_to_stdout else None

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
        os.dup2(wfd, 1)
        os.dup2(wfd, 2)
        # Re-apply non-inheritable on 1/2 after dup2 (belt & suspenders)
        _try(os.set_inheritable, 1, False)  # noqa: FBT003
        _try(os.set_inheritable, 2, False)  # noqa: FBT003
    finally:
        _try(os.close, wfd)

    new_stdout = _wrap_fd(1, encoding)
    new_stderr = _wrap_fd(2, encoding)
    sys.stdout, sys.stderr = new_stdout, new_stderr

    try:
        yield
    finally:
        # Flush while still redirected
        _try(sys.stdout.flush)
        _try(sys.stderr.flush)
        _try(_fflush_all)

        # Close wrappers so their buffers go into the pipe
        _try(new_stdout.close)
        _try(new_stderr.close)

        # Restore original fds 1/2 (closing this proc's pipe write ends)
        try:
            os.dup2(saved_fd1, 1)
            os.dup2(saved_fd2, 2)
        finally:
            _try(os.close, saved_fd1)
            _try(os.close, saved_fd2)

        sys.stdout, sys.stderr = old_stdout, old_stderr

        # Restore original inheritability for fds 1/2 (best-effort)
        if inh1 is not None:
            _try(os.set_inheritable, 1, bool(inh1))
        if inh2 is not None:
            _try(os.set_inheritable, 2, bool(inh2))

        deadline = time.monotonic() + max(0.0, float(timeout))
        _join_until(t, deadline)

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
