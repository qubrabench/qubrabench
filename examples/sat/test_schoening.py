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
    inst = SatInstance.random(k=3, n=70, m=230, rng=rng)
    stats = QueryStats()
    x = schoening_solve(inst, rng=rng, error=10**-5, stats=stats)
    assert x is not None

    # check that found a solution
    assert inst.evaluate(x)

    # check stats
    assert stats == QueryStats(
        classical_control_method_calls=0,
        classical_actual_queries=34,
        classical_expected_queries=pytest.approx(556815815.9282745),
        quantum_expected_classical_queries=pytest.approx(130),
        quantum_expected_quantum_queries=pytest.approx(106284.22802859028),
    )
