"""This module collects test functions for the qubrabench.search method."""

import pytest
import re

from qubrabench.algorithms.search import search
from qubrabench.benchmark import QueryStats, track_queries, oracle


def test_search_linear_scan():
    """Test linear search through a list"""
    with track_queries() as tracker:
        domain = range(0, 100)

        @oracle
        def check(it):
            return it == 50

        result = search(domain, check, error=10**-5)
        assert result == 50

        # test stats
        assert tracker.get_stats(check) == QueryStats(
            classical_actual_queries=51,
            classical_expected_queries=51,
            quantum_expected_classical_queries=72.9245740488006,
            quantum_expected_quantum_queries=18.991528740664712,
        )


def test_search_linear_scan_classical_queries(rng):
    """Check that classical expected and actual queries match"""
    for N in rng.integers(20, 500, size=20):
        k = rng.integers(N)

        @oracle
        def check(it):
            return it == k

        with track_queries() as tracker:
            result = search(range(N), check, error=10**-5)
            assert result == k
            stats: QueryStats = tracker.get_stats(check)
            assert (
                stats.classical_actual_queries
                == stats.classical_expected_queries
                == k + 1
            )


def test_search_with_shuffle(rng):
    """Tests the qubrabench search function on a trivial search space.

    Args:
        rng (np.rng): Source of randomness provided by test fixtures
    """
    with track_queries() as tracker:
        domain = range(0, 100)

        @oracle
        def check(it):
            return it == 50

        result = search(domain, check, error=10**-5, rng=rng)
        assert result == 50

        # test stats
        assert tracker.get_stats(check) == QueryStats(
            classical_actual_queries=45,
            classical_expected_queries=50.5,
            quantum_expected_classical_queries=72.9245740488006,
            quantum_expected_quantum_queries=18.991528740664712,
        )


def test_search_raises_on_stats_requested_and_eps_missing(rng):
    """Test that a ValueError is thrown when 'error' (failure rate) is not provided to the search function.

    Args:
        rng (np.rng): Source of randomness provided by test fixtures
    """
    with pytest.raises(
        ValueError,
        match=re.escape(
            "search() parameter 'error' not provided, cannot compute quantum query statistics"
        ),
    ):
        with track_queries():
            search(range(100), lambda it: it == 42, rng=rng)


def test_variable_time_key(rng):
    @oracle
    def is_prime(i: int) -> bool:
        for j in range(2, i):
            if i % j == 0:
                return False
        return True

    with track_queries() as tracker:
        twin_primes = search(
            range(2, 10),
            key=lambda i: is_prime(i) and is_prime(i + 2),
            rng=rng,
            error=10**-5,
        )
        assert twin_primes == 3  # (3, 5)
        stats = tracker.get_stats(is_prime)
        print(stats)
