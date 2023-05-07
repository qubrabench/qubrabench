import numpy as np
import pytest
from pytest_check import check
import re

from qubrabench.algorithms.search import search
from qubrabench.stats import QueryStats


def test_search():
    list_to_search = range(0, 100)
    stats = QueryStats()

    result = search(
        list_to_search,
        lambda it: it == 50,
        eps=10**-5,
        stats=stats,
        rng=np.random.default_rng(seed=12),
    )

    check.equal(result, 50)
    check.equal(stats.classical_control_method_calls, 0)
    check.equal(stats.classical_actual_queries, 45)
    check.equal(stats.classical_expected_queries, 50.5)
    check.equal(stats.quantum_expected_classical_queries, 72.9245740488006)
    check.equal(stats.quantum_expected_quantum_queries, 18.991528740664712)


def test_search_raises_on_stats_requested_and_eps_missing():
    with pytest.raises(
        ValueError, match=re.escape("search() eps not provided, cannot compute stats")
    ):
        stats = QueryStats()
        search(range(100), lambda it: it == 42, stats=stats)
