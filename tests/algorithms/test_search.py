"""This module collects test functions for the qubrabench.search method."""

import numpy as np
import pytest

from qubrabench.algorithms.search import search
from qubrabench.benchmark import QueryStats, default_tracker, oracle
from qubrabench.datastructures.qlist import QList


def list_100_search_stats(*, actual_queries: int, is_random: bool) -> QueryStats:
    return QueryStats(
        classical_actual_queries=actual_queries,
        classical_expected_queries=pytest.approx(50.5 if is_random else actual_queries),
        quantum_expected_classical_queries=pytest.approx(72.9245740488006),
        quantum_expected_quantum_queries=pytest.approx(37.983057481329425),
        quantum_worst_case_classical_queries=pytest.approx(0.0),
        quantum_worst_case_quantum_queries=pytest.approx(497.9317227558481),
    )


def test_search_linear_scan():
    """Test linear search through a list"""
    domain = range(0, 100)

    @oracle
    def check(it):
        return it == 50

    result = search(domain, check, max_fail_probability=10**-5)
    assert result == 50

    assert check.get_stats() == list_100_search_stats(
        actual_queries=51, is_random=False
    )


def test_search_linear_scan_with_qlist():
    domain = QList(range(0, 100))

    def check(it):
        return it == 50

    result = search(domain, check, max_fail_probability=10**-5)
    assert result == 50

    assert domain.stats == list_100_search_stats(actual_queries=51, is_random=False)


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

    assert check.get_stats() == list_100_search_stats(actual_queries=45, is_random=True)


def test_search_with_shuffle_qlist(rng):
    domain = QList(np.arange(0, 100))

    def check(it):
        return it == 50

    result = search(domain, check, max_fail_probability=10**-5, rng=rng)
    assert result == 50

    assert domain.stats == list_100_search_stats(actual_queries=45, is_random=True)


@pytest.mark.xfail
def test_nested_search():
    n = 10
    eps = 1e-5

    @oracle
    def my_oracle():
        pass

    def check_entry(i: int, j: int):
        for _ in range(i + j + 1):
            my_oracle()
        return j == n - 1

    def check_row(i: int) -> bool:
        marked = search(
            range(n),
            key=lambda j: check_entry(i, j),
            max_fail_probability=(eps / 2) / n,
        )
        assert marked == n - 1
        return i == n - 1

    sol = search(range(n), key=check_row, max_fail_probability=eps / 2)
    assert sol == n - 1

    assert my_oracle.get_stats() == QueryStats()
