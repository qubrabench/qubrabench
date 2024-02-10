"""This module collects test functions for the examples.sat.schoening module."""

import numpy as np
import pytest
from sat import SatInstance
from schoening import (
    schoening_bruteforce_steps,
    schoening_solve,
    schoening_with_randomness,
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

        # check that found a solution
        assert x is not None and inst.evaluate(x)

        # check stats
        assert tracker.get_stats(schoening_with_randomness) == QueryStats(
            classical_actual_queries=2,
            classical_expected_queries=pytest.approx(64),
            quantum_expected_classical_queries=pytest.approx(3.3703703703703702),
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
        assert tracker.get_stats(schoening_with_randomness) == QueryStats(
            classical_actual_queries=3,
            classical_expected_queries=pytest.approx(254),
            quantum_expected_classical_queries=pytest.approx(11.481481481481481),
            quantum_expected_quantum_queries=pytest.approx(0),
        )
