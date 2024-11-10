from __future__ import annotations

import importlib
import sys
import types
from collections import defaultdict

from ..task import Task

# TODO
# Consider sys.addaudithook
# unittest.mock

# or switch to multiple dispatch?
# add dispatch where first object is Workspace or TaskInfo
# returns Task object
# or dispatch with argument passthrough


# TODO: is it possible to encode (normal) functions with Task arguments as Tasks (cache=False)?


class TaskGraphBuilder:
    def __init__(self, globals) -> None:
        self.globals = globals

    def __enter__(self):
        # get @task functions from global namespace
        self.globals_tasks = {
            name: fn
            for name, fn in self.globals.items()
            if isinstance(fn, types.FunctionType) and hasattr(fn, "__task__")
        }

        # when func(*args, **kwargs) is called: we get Task(func, *args, **kwargs)
        self.globals.update(
            {name: Task._get_factory(fn) for name, fn in self.globals_tasks.items()}
        )

        self.module_tasks = defaultdict(dict)

        for module_name, m in sys.modules.items():
            if not (
                module_name in sys.builtin_module_names
                or importlib._imp.is_frozen(module_name)  # pyright: ignore [reportAttributeAccessIssue]
                or isinstance(m.__dict__, types.MappingProxyType)
                # could add more checks here
            ):
                for function_name in list(m.__dict__.keys()):
                    fn = m.__dict__[function_name]
                    if isinstance(fn, types.FunctionType) and hasattr(fn, "__task__"):
                        self.module_tasks[m][function_name] = fn
                        m.__dict__[function_name] = Task._get_factory(fn)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.globals.update(self.globals_tasks)
        for m in self.module_tasks.keys():
            m.__dict__.update(self.module_tasks[m])
