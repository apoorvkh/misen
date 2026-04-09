"""Handler collections grouped by domain."""

from misen.utils.hashing.handlers.optional import optional_handlers, optional_handlers_by_type
from misen.utils.hashing.handlers.stdlib import stdlib_handlers, stdlib_handlers_by_type

__all__ = [
    "optional_handlers",
    "optional_handlers_by_type",
    "stdlib_handlers",
    "stdlib_handlers_by_type",
]
