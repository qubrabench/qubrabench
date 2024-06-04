"""This module collects test functions surrounding the SAT bruteforce example."""

import numpy as np
import pytest
from bruteforce import bruteforce_solve
from sat import SatInstance

import qubrabench as qb


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

    with qb.benchmark.track_queries() as frame:
        x = bruteforce_solve(inst, rng=rng, error=10**-5)

        # check stats
        assert frame.get_stats(inst.evaluate) == qb.benchmark.QueryStats(
            classical_actual_queries=2,
            classical_expected_queries=3,
            quantum_expected_classical_queries=pytest.approx(4),
            quantum_expected_quantum_queries=pytest.approx(0),
            quantum_worst_case_classical_queries=pytest.approx(0),
            quantum_worst_case_quantum_queries=pytest.approx(291.4393894717541),
        )

        # validate solution
        assert x is not None and inst.evaluate(x)
