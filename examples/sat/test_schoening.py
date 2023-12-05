"""This module collects test functions for the examples.sat.schoening module."""

import pytest
import numpy as np

from qubrabench.benchmark import track_queries, QueryStats

from sat import SatInstance
from schoening import schoening_solve, schoening_with_randomness, schoening_bruteforce_steps


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

    with track_queries() as tracker:
        x = schoening_solve(inst, rng=rng, error=10**-5)

        # check that found a solution
        assert x is not None and inst.evaluate(x)

        # check stats
        assert tracker.get_stats(schoening_with_randomness) == QueryStats(
            classical_actual_queries=2,
            classical_expected_queries=pytest.approx(2.0009153259895374),
            quantum_expected_classical_queries=pytest.approx(2.0009153318077804),
            quantum_expected_quantum_queries=pytest.approx(0),
        )
def test_bruteforce_steps(rng) -> None:
    """Test Schöning's algorithm and quantum statistic generation

    Args:
        rng (np.rng): Source of randomness provided by test fixtures
    """
    # solve a simple SAT instance
    inst = SatInstance(
        k=3,
        clauses=np.array([[1, 1, -1], [1, -1, 1], [-1, 1, 1], [1, -1, -1]], dtype=int),
    )

    with track_queries() as tracker:
        x = schoening_bruteforce_steps(inst, rng=rng, error=10**-5)

        # check that found a solution
        assert x is not None and inst.evaluate(x)

        # check stats
        assert tracker.get_stats(inst) == QueryStats(
            classical_actual_queries=2,
            classical_expected_queries=pytest.approx(1.0009153259895374),
            quantum_expected_classical_queries=pytest.approx(1.0009153318077804),
            quantum_expected_quantum_queries=pytest.approx(0),
        )
