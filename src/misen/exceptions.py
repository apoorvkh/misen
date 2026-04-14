"""Exception hierarchy for the ``misen`` package.

All errors deliberately raised by ``misen`` derive from :class:`MisenError`, so
user code can catch the whole family with a single ``except`` clause while
still discriminating by subclass when needed.

Subclasses:

- :class:`CacheError`: cache miss or missing prerequisite cache entry.
- :class:`ConfigError`: invalid or unresolvable TOML configuration.
- :class:`HashError`: a value cannot be hashed by ``stable_hash``.
- :class:`SerializationError`: serializer load/save failure.
"""

from __future__ import annotations

__all__ = [
    "CacheError",
    "ConfigError",
    "HashError",
    "MisenError",
    "SerializationError",
]


class MisenError(Exception):
    """Base class for all ``misen``-raised exceptions."""


class CacheError(MisenError):
    """Raised on cache misses or missing prerequisite cache entries."""


class ConfigError(MisenError):
    """Raised when TOML settings cannot be resolved to a concrete instance."""


class HashError(MisenError):
    """Raised when ``stable_hash`` cannot hash a value."""


class SerializationError(MisenError):
    """Raised on serializer save/load failures."""
