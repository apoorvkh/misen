"""Canonical object hashing with extensible type handlers.

Design notes:

- Hashes are type-aware: runtime type name is included in the top-level digest.
- Built-in handlers cover common primitives/collections deterministically.
- Optional library handlers are matched dynamically when installed.
- Unknown types fall back to dill serialization.
"""

import warnings
from functools import cache, partial
from importlib.metadata import PackageNotFoundError, packages_distributions
from importlib.metadata import version as distribution_version
from typing import Any

from misen_hash.handler_base import PrimitiveHandler, qualified_type_name
from misen_hash.handlers.fallback import DillHandler
from misen_hash.hash import hash_msgspec
from misen_hash.registry import lookup_handler

__all__ = ["stable_hash"]

@cache
def _pin_recommendation_for_module(module_name: str) -> str | None:
    top_level_package = module_name.split(".", maxsplit=1)[0]
    distributions = packages_distributions().get(top_level_package)
    if not distributions:
        return None

    distribution_name = sorted(distributions)[0]
    try:
        current_version = distribution_version(distribution_name)
    except PackageNotFoundError:
        return None

    return (
        f"We recommend pinning {distribution_name}=={current_version} in your pyproject `dev` dependencies "
        f'(for example: `uv add --dev "{distribution_name}=={current_version}"`).'
    )


@cache
def _on_dill_fallback(obj_cls: type[Any]) -> None:
    obj_type = qualified_type_name(obj_cls)
    module_name = obj_cls.__module__

    message = (
        f"stable_hash fell back to dill serialization for {obj_type}. "
        "This may be unstable across Python/library versions."
    )
    if pin_recommendation := _pin_recommendation_for_module(module_name):
        message = f"{message} {pin_recommendation}"

    warnings.warn(message, UserWarning, stacklevel=3)


def stable_hash(obj: Any, *, _active_ids: set[int] | None = None) -> int:
    """Return a stable 64-bit hash for an arbitrary Python object."""
    if _active_ids is None:
        _active_ids = set()

    obj_type = qualified_type_name(type(obj))
    handler_cls = lookup_handler(obj)
    if handler_cls is DillHandler:
        _on_dill_fallback(type(obj))

    if issubclass(handler_cls, PrimitiveHandler):
        obj_hash = handler_cls.digest(obj)
    else:
        obj_id = id(obj)
        if obj_id in _active_ids:
            # Use a stable marker for back-edges to avoid infinite recursion.
            return hash_msgspec((obj_type, "__recursive_reference__"))

        _active_ids.add(obj_id)
        try:
            obj_hash = handler_cls.digest(obj, element_hash=partial(stable_hash, _active_ids=_active_ids))
        finally:
            _active_ids.remove(obj_id)

    return hash_msgspec((obj_type, obj_hash))
