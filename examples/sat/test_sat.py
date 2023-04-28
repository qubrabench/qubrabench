import time
import pytest
from pytest_check import check
import random

import numpy as np
from sat import run_specific_instance as run


def equals(x):
    return lambda y: x == y


def isclose(x):
    return lambda y: np.isclose(x, y)


def test_simple_sat():
    # TODO
    np.random.seed(3)
    random.seed(3)

    n = 3
    k = 2
    m = 3

    stats = run()

    def verify(stat, checker):
        with check:
            assert checker(
                stats[stat]
            ), f"Stat `{stat}` does not match! got: {stats[stat]}"

    verify("n", equals(n))
    verify("k", equals(k))
    verify("m", equals(m))

    verify("T", equals(2))

    verify("classical_actual_queries", equals(2))
    verify("classical_expected_queries", equals(3))
    verify("quantum_expected_classical_queries", isclose(4))
    verify("quantum_expected_quantum_queries", isclose(0))
