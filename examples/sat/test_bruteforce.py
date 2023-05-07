import numpy as np
from pytest_check import check

from qubrabench.stats import QueryStats
from sat import SatInstance
from bruteforce import bruteforce_solve


def test_solve() -> None:
    # solve a simple SAT instance
    inst = SatInstance(
        k=2,
        clauses=np.array([[1, 1, 0], [1, -1, 0], [0, 1, 1], [0, -1, -1]], dtype=int),
    )
    stats = QueryStats()
    x = bruteforce_solve(
        inst, rng=np.random.default_rng(seed=12), eps=10**-5, stats=stats
    )

    # check that found a solution
    assert inst.evaluate(x)

    # check stats
    check.equal(stats.classical_control_method_calls, 0)
    check.equal(stats.classical_actual_queries, 2)
    check.almost_equal(stats.classical_expected_queries, 3)
    check.almost_equal(stats.quantum_expected_classical_queries, 4)
    check.almost_equal(stats.quantum_expected_quantum_queries, 0)
