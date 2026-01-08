from __future__ import annotations

import codecs
import contextlib
import io
import os
import sys
import threading
from typing import Optional, TextIO


@contextlib.contextmanager
def capture_all_output(target: TextIO, *, encoding: Optional[str] = None, flush_target: bool = True):
    """
    Redirect *OS fds* 1/2 (stdout/stderr) into `target`, capturing:
      - Python prints
      - `os.write(1/2, ...)`
      - C/C++ writes to stdout/stderr (fd 1/2)

    Notes:
      - Best-effort `fflush(NULL)` on enter/exit to flush libc stdio buffers.
      - Merges stdout+stderr into one stream (ordering is "as observed" by the pipe).
    """
    if encoding is None:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"

    # Flush libc buffers if possible (helps under sbatch / non-TTY buffering).
    def _fflush_all() -> None:
        try:
            import ctypes  # local import to keep it lightweight

            ctypes.CDLL(None).fflush(None)
        except Exception:
            pass

    # Save Python-level streams
    old_stdout, old_stderr = sys.stdout, sys.stderr

    # Flush Python + libc before switching
    try:
        old_stdout.flush()
    except Exception:
        pass
    try:
        old_stderr.flush()
    except Exception:
        pass
    _fflush_all()

    # Save OS fds 1/2
    saved_fd1 = os.dup(1)
    saved_fd2 = os.dup(2)

    # One pipe for both stdout/stderr
    rfd, wfd = os.pipe()
    os.set_inheritable(rfd, False)
    os.set_inheritable(wfd, False)

    lock = threading.Lock()
    stop_evt = threading.Event()

    def reader() -> None:
        decoder = codecs.getincrementaldecoder(encoding)(errors="replace")
        try:
            while True:
                try:
                    chunk = os.read(rfd, 8192)
                except OSError:
                    break
                if not chunk:
                    break
                text = decoder.decode(chunk)
                if text:
                    with lock:
                        target.write(text)
                        if flush_target:
                            target.flush()
            # Flush any remaining decoder state
            tail = decoder.decode(b"", final=True)
            if tail:
                with lock:
                    target.write(tail)
                    if flush_target:
                        target.flush()
        finally:
            stop_evt.set()

    t = threading.Thread(target=reader, name="capture_all_output", daemon=True)
    t.start()

    # Redirect fds 1/2 to the pipe write-end
    try:
        os.dup2(wfd, 1)
        os.dup2(wfd, 2)
    finally:
        os.close(wfd)  # fds 1/2 now reference the pipe

    # Replace sys.stdout/sys.stderr with line-buffered, write-through wrappers over fds 1/2
    def _wrap_fd(fd: int) -> TextIO:
        raw = io.FileIO(fd, mode="w", closefd=False)
        buf = io.BufferedWriter(raw)
        return io.TextIOWrapper(buf, encoding=encoding, errors="replace", line_buffering=True, write_through=True)

    new_stdout = _wrap_fd(1)
    new_stderr = _wrap_fd(2)
    sys.stdout, sys.stderr = new_stdout, new_stderr

    try:
        yield
    finally:
        # Flush Python + libc while still redirected
        try:
            sys.stdout.flush()
        except Exception:
            pass
        try:
            sys.stderr.flush()
        except Exception:
            pass
        _fflush_all()

        # Close wrappers *before* restoring fds, so buffered text goes into the pipe
        try:
            new_stdout.close()
        except Exception:
            pass
        try:
            new_stderr.close()
        except Exception:
            pass

        # Restore original fds 1/2
        try:
            os.dup2(saved_fd1, 1)
            os.dup2(saved_fd2, 2)
        finally:
            os.close(saved_fd1)
            os.close(saved_fd2)

        # Restore Python-level streams
        sys.stdout, sys.stderr = old_stdout, old_stderr

        # Stop reader: close read end to avoid hangs if some child inherited the write end
        try:
            os.close(rfd)
        except Exception:
            pass

        # Join briefly; closing rfd above prevents indefinite hangs
        t.join(timeout=1.0)
        stop_evt.set()
