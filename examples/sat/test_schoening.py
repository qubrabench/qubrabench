"""This module collects test functions for the examples.sat.schoening module."""

import pytest
from qubrabench.stats import QueryStats
from sat import SatInstance
from schoening import schoening_solve


def test_solve(rng) -> None:
    """Test Schöning's algorithm and quantum statistic generation

    Args:
        rng (np.rng): Source of randomness provided by test fixtures
    """
    # solve a simple SAT instance
    inst = SatInstance.random(3, 30, 20, rng=rng)
    stats = QueryStats()
    x = schoening_solve(inst, rng=rng, error=10**-5, stats=stats)
    assert x is not None

    # check that found a solution
    assert inst.evaluate(x)

    # check stats
    assert stats == QueryStats(
        classical_control_method_calls=0,
        classical_actual_queries=1,
        classical_expected_queries=pytest.approx(1.0009153259895374),
        quantum_expected_classical_queries=pytest.approx(1.0009153318077804),
        quantum_expected_quantum_queries=pytest.approx(0),
    )
