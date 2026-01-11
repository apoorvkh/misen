from __future__ import annotations

import codecs
import contextlib
import ctypes
import io
import os
import sys
import threading
import time
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from collections.abc import Callable


def _try(fn: Callable, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:  # noqa: BLE001
        return None


def _fflush_all() -> None:
    ctypes.CDLL(None).fflush(None)


def _write(text: str, lock: threading.Lock, target: TextIO) -> None:
    if not text:
        return
    with lock:
        target.write(text)
        _try(target.flush)


def _make_decoder(enc: str) -> codecs.IncrementalDecoder:
    return codecs.getincrementaldecoder(enc)(errors="replace")


def _wrap_fd(fd: int, enc: str) -> TextIO:
    raw = io.FileIO(fd, mode="w", closefd=False)
    buf = io.BufferedWriter(raw)
    return io.TextIOWrapper(
        buf,
        encoding=enc,
        errors="replace",
        line_buffering=True,
        write_through=True,
    )


def _join_until(th: threading.Thread, deadline: float) -> None:
    th.join(timeout=max(0.0, deadline - time.monotonic()))


@contextlib.contextmanager
def capture_all_output(target: TextIO, timeout: float = 10.0):
    """
    Capture everything written to OS fds 1/2 (stdout/stderr), including C/C++ writes,
    and tee into `target`.

    Exit behavior:
      - waits up to `timeout` seconds to drain cleanly
      - if not drained, best-effort drains what's currently available and then stops
        (never hangs, may lose trailing output from stray inheritors).
    """
    encoding: str = getattr(sys.stdout, "encoding", None) or "utf-8"

    old_stdout, old_stderr = sys.stdout, sys.stderr
    lock = threading.Lock()

    # --- guard against recursion / feedback loops ---
    # Case 1: explicitly passing current stdout/stderr objects
    if target is old_stdout or target is old_stderr:
        msg = "capture_all_output: `target` must not be sys.stdout/sys.stderr (would recurse)"
        raise ValueError(msg)

    # Case 2: target ultimately writes to fd 1 or 2 (e.g. TextIOWrapper over stdout)
    # Best-effort: if it quacks like a file descriptor and is 1/2, refuse.
    fileno = _try(getattr, target, "fileno")
    if callable(fileno):
        fd = _try(fileno)
        if fd in (1, 2):
            msg = "capture_all_output: `target` must not write to fd 1/2 (would recurse)"
            raise ValueError(msg)
    # --- end guard ---

    # Flush before redirecting
    _try(old_stdout.flush)
    _try(old_stderr.flush)
    _try(_fflush_all)

    # Save original inheritability of fds 1/2 and make them non-inheritable (best-effort)
    inh1 = _try(os.get_inheritable, 1)
    inh2 = _try(os.get_inheritable, 2)
    _try(os.set_inheritable, 1, False)
    _try(os.set_inheritable, 2, False)

    saved_fd1 = os.dup(1)
    saved_fd2 = os.dup(2)

    rfd, wfd = os.pipe()
    os.set_inheritable(rfd, False)
    os.set_inheritable(wfd, False)

    def reader() -> None:
        dec = _make_decoder(encoding)
        try:
            while True:
                try:
                    chunk = os.read(rfd, 8192)
                except OSError:
                    break
                if not chunk:
                    break
                _write(dec.decode(chunk), lock=lock, target=target)
            _write(dec.decode(b"", final=True), lock=lock, target=target)
        except Exception:
            pass
        finally:
            _try(os.close, rfd)

    t = threading.Thread(target=reader, name="capture_all_output", daemon=True)
    t.start()

    # Redirect fd 1/2 -> pipe
    try:
        os.dup2(wfd, 1)
        os.dup2(wfd, 2)
        # Re-apply non-inheritable on 1/2 after dup2 (belt & suspenders)
        _try(os.set_inheritable, 1, False)
        _try(os.set_inheritable, 2, False)
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
            # Best-effort: drain what is available, then stop.
            _try(os.set_blocking, rfd, False)

            dec = _make_decoder(encoding)
            while time.monotonic() < deadline:
                try:
                    chunk = os.read(rfd, 8192)
                except BlockingIOError:
                    break
                except OSError:
                    break
                if not chunk:
                    break
                _write(dec.decode(chunk), lock=lock, target=target)

            _try(_write, dec.decode(b"", final=True), lock, target)

            # Force reader to exit; since it's daemon, we still won't hang regardless.
            _try(os.close, rfd)
            t.join(timeout=0.2)

        _try(target.flush)
