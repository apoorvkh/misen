"""Helpers for frozen/immutable object behavior."""

from __future__ import annotations

from typing import Any


class FrozenMixin:
    """Mixin that disallows mutation after `freeze()`.

    This mixin provides a lightweight "freezable" immutability mechanism for
    slot-based classes.

    Key behaviors:
    - Before freezing, attribute assignment behaves normally.
    - After calling `freeze()`, any attempt to set an attribute via normal
      assignment raises `AttributeError`.
    - `unfreeze()` re-enables mutation.
    - Pickling support (`__getstate__` / `__setstate__`) captures and restores
      all slot-backed attributes across the entire MRO (including `_frozen`),
      avoiding the common pitfall where `self.__slots__` only reflects the most
      derived class.
    """

    __slots__ = ("_frozen",)

    def freeze(self) -> None:
        """Mark the object as immutable."""
        object.__setattr__(self, "_frozen", True)

    def unfreeze(self) -> None:
        """Mark the object as mutable."""
        object.__setattr__(self, "_frozen", False)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute, enforcing immutability when frozen.

        Raises:
            AttributeError: If the object is frozen and a mutation is attempted.
        """
        if getattr(self, "_frozen", False):
            msg = f"{type(self).__name__} is immutable. Tried to set attribute {name!r} to {value!r}."
            raise AttributeError(msg)
        object.__setattr__(self, name, value)

    def __getstate__(self) -> dict[str, Any]:
        """Return pickle state for slot-based objects.

        The state includes all slot-backed attributes across the MRO (including
        `_frozen`). Slot attributes that have not been set are omitted.
        """
        return {
            name: getattr(self, name)
            for base in type(self).__mro__
            for s in (getattr(base, "__slots__", ()),)
            for name in ((s,) if isinstance(s, str) else s)
            if name not in ("__dict__", "__weakref__") and hasattr(self, name)
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore pickle state while avoiding immutability checks.

        This method sets attributes via `object.__setattr__` to ensure unpickling
        does not trigger `__setattr__`'s frozen guard.
        """
        for name, value in state.items():
            object.__setattr__(self, name, value)
