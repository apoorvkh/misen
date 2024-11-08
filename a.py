import numpy as np

from misen import Experiment, task
from misen.utils.det_hash import deterministic_hashing


@task(uuid="QuNP")
def add(x, y):
    return x + y


@task(uuid="def")
def multiply(x, y, z: int = 0, **kwargsx):
    return x * y


def double(x):
    return x * 2


class MultiplyExperiment(Experiment):
    def calls(self):
        return multiply(add(double(1), 4), add(np.csingle(3), 4), hello=np.datetime64("2005-02-25"))


if __name__ == "__main__":
    from time import time

    print(MultiplyExperiment().step_graph.__repr__())

    with deterministic_hashing():
        z = MultiplyExperiment().step_graph
        start = time()
        print(hash(z))
        print(time() - start)

    print(MultiplyExperiment().calls())
