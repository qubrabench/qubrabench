import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pytest


@pytest.fixture()
def rng():
    return np.random.default_rng(seed=12)


@dataclass
class Timer:
    """Wrapper class to store execution time"""

    delta: float | None = None
    """seconds"""


@pytest.fixture()
def time_execution():
    """Context to time execution of the wrapped code

    .. code:: python

        with time_execution() as timer:
            ...
        print(timer.delta)
    """

    @contextmanager
    def ctx():
        timer = Timer()
        start = time.time_ns()
        try:
            yield timer
        except Exception as e:
            raise e
        finally:
            end = time.time_ns()
            timer.delta = (end - start) / 1e9

    return ctx
