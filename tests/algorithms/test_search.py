import pytest
from pytest_check import check

from qubrabench.algorithms.search import search
from qubrabench.stats import QueryStats


def test_search(rng):
    # test functionality
    domain = range(0, 100)
    stats = QueryStats()
    result = search(domain, lambda it: it == 50, eps=10**-5, stats=stats, rng=rng)
    assert result == 50

    # test stats
    check.equal(result, 50)
    check.equal(stats.classical_control_method_calls, 0)
    check.equal(stats.classical_actual_queries, 45)
    check.equal(stats.classical_expected_queries, 50.5)
    check.equal(stats.quantum_expected_classical_queries, 72.9245740488006)
    check.equal(stats.quantum_expected_quantum_queries, 18.991528740664712)


def test_search_raises_on_stats_requested_and_eps_missing(rng):
    with pytest.raises(ValueError, match="eps not provided"):
        stats = QueryStats()
        search(range(100), lambda it: it == 42, rng=rng, stats=stats)
