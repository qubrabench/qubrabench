"""This module collects test functions surrounding the SAT bruteforce example."""

import pytest
import numpy as np

import qubrabench as qb

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

    @qb.benchmark.oracle
    def evaluate(y):
        return inst.evaluate(y)

    with qb.benchmark.track_queries() as frame:
        x = bruteforce_solve(inst, evaluate, rng=rng, error=10**-5)
        # check that found a solution
        assert x is not None and inst.evaluate(x)

        # check stats
        assert frame.get_stats(evaluate) == qb.benchmark.QueryStats(
            classical_actual_queries=2,
            classical_expected_queries=3,
            quantum_expected_classical_queries=pytest.approx(4),
            quantum_expected_quantum_queries=pytest.approx(0),
        )
