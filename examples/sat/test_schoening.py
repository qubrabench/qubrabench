"""This module collects test functions for the examples.sat.schoening module."""

import numpy as np
from pytest_check import check

from qubrabench.stats import QueryStats
from sat import SatInstance
from schoening import schoening_solve


def test_solve(rng) -> None:
    """Test Schöning's algorithm and quantum statistic generation

    Args:
        rng (np.rng): Source of randomness provided by test fixtures
    """
    # solve a simple SAT instance
    inst = SatInstance(
        k=3,
        clauses=np.array([[1, 1, -1], [1, -1, 1], [-1, 1, 1], [1, -1, -1]], dtype=int),
    )
    stats = QueryStats()
    x = schoening_solve(inst, rng=rng, eps=10**-5, stats=stats)
    assert x is not None

    # check that found a solution
    assert inst.evaluate(x)

    # check stats
    check.equal(stats.classical_control_method_calls, 0)
    check.equal(stats.classical_actual_queries, 1)
    check.almost_equal(stats.classical_expected_queries, 1.0714455822814957)
    check.almost_equal(stats.quantum_expected_classical_queries, 1.0714460684249203)

    check.almost_equal(stats.quantum_expected_quantum_queries, 0)
