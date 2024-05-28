"""This module collects test functions for the qubrabench.search method."""

import numpy as np
from pytest import approx

from qubrabench.algorithms.search import search
from qubrabench.benchmark import QueryStats, default_tracker, oracle
from qubrabench.datastructures.qlist import QList


def test_search_linear_scan():
    """Test linear search through a list"""
    domain = range(0, 100)

    @oracle
    def check(it):
        return it == 50

    result = search(domain, check, max_fail_probability=10**-5)
    assert result == 50

    # test stats
    assert default_tracker().get_stats(check) == QueryStats(
        classical_actual_queries=51,
        classical_expected_queries=51,
        quantum_expected_classical_queries=72.9245740488006,
        quantum_expected_quantum_queries=18.991528740664712,
    )


def test_search_linear_scan_with_qlist():
    domain = QList(range(0, 100))

    def check(it):
        return it == 50

    result = search(domain, check, max_fail_probability=10**-5)
    assert result == 50

    # test stats
    assert domain.stats == QueryStats(
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

        result = search(range(N), check, max_fail_probability=10**-5)
        assert result == k
        stats: QueryStats = default_tracker().get_stats(check)
        assert (
            stats.classical_actual_queries == stats.classical_expected_queries == k + 1
        )


def test_search_with_shuffle(rng):
    """Tests the qubrabench search function on a trivial search space."""
    domain = np.arange(100)

    @oracle
    def check(it):
        return it == 50

    result = search(domain, check, max_fail_probability=10**-5, rng=rng)
    assert result == 50

    # test stats
    assert default_tracker().get_stats(check) == QueryStats(
        classical_actual_queries=45,
        classical_expected_queries=50.5,
        quantum_expected_classical_queries=72.9245740488006,
        quantum_expected_quantum_queries=18.991528740664712,
    )


def test_search_with_shuffle_qlist(rng):
    domain = QList(np.arange(0, 100))

    def check(it):
        return it == 50

    result = search(domain, check, max_fail_probability=10**-5, rng=rng)
    assert result == 50

    # test stats
    assert domain.stats == QueryStats(
        classical_actual_queries=45,
        classical_expected_queries=50.5,
        quantum_expected_classical_queries=72.9245740488006,
        quantum_expected_quantum_queries=18.991528740664712,
    )


def test_variable_time_key(rng):
    @oracle
    def is_prime(i: int) -> bool:
        for j in range(2, i):
            if i % j == 0:
                return False
        return True

    twin_primes = search(
        range(2, 10),
        key=lambda i: is_prime(i) and is_prime(i + 2),
        rng=rng,
        max_fail_probability=10**-5,
    )
    assert twin_primes == 3  # (3, 5)
    stats = default_tracker().get_stats(is_prime)
    assert stats == QueryStats(
        classical_actual_queries=2,
        classical_expected_queries=6,
        quantum_expected_classical_queries=approx(8),
        quantum_expected_quantum_queries=approx(0),
    )
