"""This module collects test functions for the examples.sat.schoening module."""

import pytest
import numpy as np
from qubrabench.stats import QueryStats
from sat import SatInstance
from schoening_variants import (
    schoening_bruteforce_assignment,
    schoening_bruteforce_steps,
)


def test_bruteforce_assignments(rng) -> None:
    """Test variant of Schöning's algorithm and quantum statistic generation

    Args:
        rng (np.rng): Source of randomness provided by test fixtures
    """
    # solve a simple SAT instance
    inst = SatInstance(
        k=3,
        clauses=np.array([[1, 1, -1], [1, -1, 1], [-1, 1, 1], [1, -1, -1]], dtype=int),
    )
    stats = QueryStats()
    x = schoening_bruteforce_assignment(inst, rng=rng, error=10**-5, stats=stats)
    assert x is not None

    # check that found a solution
    assert inst.evaluate(x)

    # check stats
    assert stats == QueryStats(
        classical_control_method_calls=0,
        classical_actual_queries=1,
        classical_expected_queries=pytest.approx(1.0),
        quantum_expected_classical_queries=pytest.approx(1.0),
        quantum_expected_quantum_queries=pytest.approx(0),
    )


def test_bruteforce_steps(rng) -> None:
    """Test variant of Schöning's algorithm and quantum statistic generation

    Args:
        rng (np.rng): Source of randomness provided by test fixtures
    """
    # solve a simple SAT instance
    inst = SatInstance(
        k=3,
        clauses=np.array([[1, 1, -1], [1, -1, 1], [-1, 1, 1], [1, -1, -1]], dtype=int),
    )
    stats = QueryStats()
    x = schoening_bruteforce_steps(inst, rng=rng, error=10**-5, stats=stats)
    assert x is not None

    # check that found a solution
    assert inst.evaluate(x)

    # check stats
    assert stats == QueryStats(
        classical_control_method_calls=0,
        classical_actual_queries=1,
        classical_expected_queries=pytest.approx(1.0),
        quantum_expected_classical_queries=pytest.approx(1.0),
        quantum_expected_quantum_queries=pytest.approx(0),
    )
