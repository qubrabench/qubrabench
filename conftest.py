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
    delta: float | None = None
    """seconds"""


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


@pytest.fixture()
def time_execution():
    return ctx
