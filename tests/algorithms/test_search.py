import random
from pytest_check import check

from qubrabench.algorithms.search import search
from qubrabench.bench.stats import QueryStats


def test_search():
    list_to_search = range(0, 100)
    stats = QueryStats()

    random.seed(12)
    result = search(list_to_search, lambda it: it == 50, eps=10**-5, stats=stats)

    check.equal(result, 50)
    check.equal(stats.classical_control_method_calls, 0)
    check.equal(stats.classical_actual_queries, 4)
    check.equal(stats.classical_expected_queries, 50.5)
    check.equal(stats.quantum_expected_classical_queries, 72.9245740488006)
    check.equal(stats.quantum_expected_quantum_queries, 18.991528740664712)
