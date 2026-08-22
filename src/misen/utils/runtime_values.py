"""Work-unit-scoped values injected through runtime sentinels."""

from __future__ import annotations

import os
from contextlib import ExitStack, contextmanager, suppress
from typing import TYPE_CHECKING, Any

from misen.sentinels import DASK_CLIENT
from misen.utils.dask_runtime import (
    DASK_EXPECTED_WORKERS_ENV,
    DASK_SCHEDULER_ADDRESS_ENV,
    DASK_STARTUP_TIMEOUT_ENV,
    DEFAULT_DASK_STARTUP_TIMEOUT,
    MIN_DASK_WORKERS,
    positive_int_env,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


class RuntimeValues(ExitStack):
    """Lazily open, share, and close runtime-injected values for one WorkUnit."""

    __slots__ = ("_dask_client",)

    def __init__(self) -> None:
        """Create an empty lazy runtime-value manager."""
        super().__init__()
        self._dask_client: Any | None = None

    def resolve(self, sentinel: object) -> Any:
        """Return a sentinel's shared runtime value, opening it on first use."""
        if sentinel is not DASK_CLIENT:
            msg = f"No runtime value provider is registered for {sentinel!r}."
            raise RuntimeError(msg)
        if self._dask_client is None:
            self._dask_client = self.enter_context(_open_dask_client())
        return self._dask_client


@contextmanager
def _open_dask_client() -> Iterator[Any]:
    """Open one allocation-scoped Client and enforce fixed membership."""
    try:
        from distributed import Client
    except ImportError as exc:
        msg = "DASK_CLIENT requires the `distributed` package in the task environment."
        raise RuntimeError(msg) from exc

    address = os.environ.get(DASK_SCHEDULER_ADDRESS_ENV)
    if not address:
        msg = "DASK_CLIENT was requested, but this executor did not provide a Dask scheduler."
        raise RuntimeError(msg)
    expected_workers = positive_int_env(DASK_EXPECTED_WORKERS_ENV)
    if expected_workers < MIN_DASK_WORKERS:
        msg = f"{DASK_EXPECTED_WORKERS_ENV} must be at least {MIN_DASK_WORKERS} for DASK_CLIENT."
        raise RuntimeError(msg)
    startup_timeout = positive_int_env(DASK_STARTUP_TIMEOUT_ENV, default=DEFAULT_DASK_STARTUP_TIMEOUT)
    client = Client(address, set_as_default=False, timeout=startup_timeout)

    try:
        client.wait_for_workers(expected_workers, timeout=startup_timeout)
        initial_workers = _dask_worker_topology(client)
        if len(initial_workers) != expected_workers:
            msg = f"Dask runtime expected exactly {expected_workers} worker(s), found {len(initial_workers)}."
            raise RuntimeError(msg)
        yield client
        current_workers = _dask_worker_topology(client)
        if current_workers != initial_workers:
            msg = (
                "Dask worker membership changed during WorkUnit execution: "
                f"initial={sorted(initial_workers)!r}, current={sorted(current_workers)!r}."
            )
            raise RuntimeError(msg)
    finally:
        # Client.close() releases the borrowed connection. Client.shutdown()
        # would also stop the executor-owned scheduler and must not be used.
        with suppress(Exception):
            client.close()


def _dask_worker_topology(client: Any) -> frozenset[str]:
    """Return the current workers' stable scheduler addresses."""
    return frozenset(map(str, client.scheduler_info()["workers"]))
