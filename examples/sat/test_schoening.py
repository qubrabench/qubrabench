"""This module collects test functions for the examples.sat.schoening module."""

import numpy as np
import pytest
from sat import SatInstance
from schoening import (
    schoening_solve,
    schoening_solve__bruteforce_over_starting_assigment,
)

from qubrabench.benchmark import QueryStats, track_queries


def test_solve(rng) -> None:
    """Test Schoening's algorithm and quantum statistic generation

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

        # check stats
        assert tracker.get_stats(inst.evaluate) == QueryStats(
            classical_actual_queries=4,
            classical_expected_queries=380,
            quantum_expected_classical_queries=pytest.approx(16.22222222222222),
            quantum_expected_quantum_queries=pytest.approx(0),
        )

        # validate solution
        assert x is not None and inst.evaluate(x)


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
        x = schoening_solve__bruteforce_over_starting_assigment(
            inst, rng=rng, error=10**-5
        )

        # check stats
        assert tracker.get_stats(inst.evaluate) == QueryStats(
            classical_actual_queries=7,
            classical_expected_queries=pytest.approx(636),
            quantum_expected_classical_queries=pytest.approx(29.703703703703702),
            quantum_expected_quantum_queries=pytest.approx(0),
        )

        # validate solution
        assert x is not None and inst.evaluate(x)
