from __future__ import annotations
from collections import defaultdict
import functools
import types
import inspect
import sys
import importlib

from ..core import Task


# TODO
# Consider sys.addaudithook
# unittest.mock


class TaskGraphBuilder:
    def __init__(self, globals) -> None:
        self.globals = globals

    @staticmethod
    def task_wrapper(func, *args, **kwargs):
        # TODO: deprecated: use signature.bind instead
        # parse kwargs properly
        callargs = inspect.getcallargs(func, *args, **kwargs)
        return Task(func, func.__task__, callargs)

    def __enter__(self):
        self.globals_tasks = {
            name: fn
            for name, fn in self.globals.items()
            if isinstance(fn, types.FunctionType) and hasattr(fn, "__task__")
        }

        self.globals.update(
            {
                name: functools.partial(self.task_wrapper, fn)
                for name, fn in self.globals_tasks.items()
            }
        )

        self.module_tasks = defaultdict(dict)

        for module_name, m in sys.modules.items():
            if not (
                module_name in sys.builtin_module_names
                or importlib._imp.is_frozen(module_name)
                or isinstance(m.__dict__, types.MappingProxyType)
                # could add more checks here
            ):
                for function_name in list(m.__dict__.keys()):
                    fn = m.__dict__[function_name]
                    if isinstance(fn, types.FunctionType) and hasattr(fn, "__task__"):
                        self.module_tasks[m][function_name] = fn
                        m.__dict__[function_name] = functools.partial(
                            self.task_wrapper, fn
                        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.globals.update(self.globals_tasks)
        for m in self.module_tasks.keys():
            m.__dict__.update(self.module_tasks[m])
