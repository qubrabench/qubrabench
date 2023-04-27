import random
from pytest_check import check

from qubrabench.algorithms.search import qsearch
from qubrabench.bench.stats import QueryStats


def test_qsearch():
    list_to_search = range(0, 100)
    stats = QueryStats()

    random.seed(12)

    with check:
        assert (
            qsearch(list_to_search, lambda it: it == 50, eps=10**-5, stats=stats)
            == 50
        )
    with check:
        assert stats.classical_control_method_calls == 0
    with check:
        assert stats.classical_actual_queries == 4
    with check:
        assert stats.classical_expected_queries == 50.5
    with check:
        assert stats.quantum_expected_classical_queries == 72.9245740488006
    with check:
        assert stats.quantum_expected_quantum_queries == 18.991528740664712
