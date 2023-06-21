"""This module collects test functions surrounding the SAT bruteforce example."""

import pytest
import numpy as np
from qubrabench.stats import QueryStats
from sat import SatInstance
from bruteforce import bruteforce_solve


def test_solve(rng) -> None:
    """Perform an example run of bruteforcing SAT and assure statistic collection.

    Args:
        rng (np.rng): Source of randomness provided by test fixtures
    """
    # solve a simple SAT instance
    inst = SatInstance(
        k=2,
        clauses=np.array([[1, 1, 0], [1, -1, 0], [0, 1, 1], [0, -1, -1]], dtype=int),
    )
    stats = QueryStats()
    x = bruteforce_solve(inst, rng=rng, error=10**-5, stats=stats)
    assert x is not None

    # check that found a solution
    assert inst.evaluate(x)

    # check stats
    assert stats == QueryStats(
        classical_control_method_calls=0,
        classical_actual_queries=2,
        classical_expected_queries=3,
        quantum_expected_classical_queries=pytest.approx(4),
        quantum_expected_quantum_queries=pytest.approx(0),
    )
