"""This module collects test functions for the qubrabench.search method."""

import pytest

from qubrabench.algorithms.search import search
from qubrabench.stats import QueryStats


def test_search(rng):
    """Tests the qubrabench search function on a trivial search space.

    Args:
        rng (np.rng): Source of randomness provided by test fixtures
    """
    # test functionality
    domain = range(0, 100)
    stats = QueryStats()
    result = search(domain, lambda it: it == 50, eps=10**-5, stats=stats, rng=rng)
    assert result == 50

    # test stats
    assert stats == QueryStats(
        classical_control_method_calls=0,
        classical_actual_queries=45,
        classical_expected_queries=50.5,
        quantum_expected_classical_queries=72.9245740488006,
        quantum_expected_quantum_queries=18.991528740664712,
    )


def test_search_raises_on_stats_requested_and_eps_missing(rng):
    """Test that a ValueError is thrown when epsilon (failure rate) is not provided to the search function.

    Args:
        rng (np.rng): Source of randomness provided by test fixtures
    """
    with pytest.raises(ValueError, match="eps not provided"):
        stats = QueryStats()
        search(range(100), lambda it: it == 42, rng=rng, stats=stats)
