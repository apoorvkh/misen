"""Handler collections grouped by domain."""

from misen_hash.handlers.builtin import builtin_handlers, builtin_handlers_by_type
from misen_hash.handlers.fallback import DillHandler
from misen_hash.handlers.optional import optional_handlers, optional_handlers_by_type

__all__ = [
    "DillHandler",
    "builtin_handlers",
    "builtin_handlers_by_type",
    "optional_handlers",
    "optional_handlers_by_type",
]
