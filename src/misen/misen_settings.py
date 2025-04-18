from functools import cached_property
from .executor import Executor, LocalExecutor
from .workspace import Workspace, TestWorkSpace


class MisenSettings:
    def __init__(self):
        pass

    @cached_property
    def workspace(self) -> Workspace:
        return TestWorkSpace()

    @cached_property
    def executor(self) -> Executor:
        return LocalExecutor()
