import numpy as np
import pytest

from qubrabench.algorithms.linalg import qlsa_query_count, solve
from qubrabench.benchmark import track_queries
from qubrabench.datastructures.matrix import Qndarray


def random_instance(rng, N: int) -> tuple[Qndarray, Qndarray]:
    A = rng.random(size=(N, N))
    b = rng.random(size=N)

    return Qndarray(A), Qndarray(b)


@pytest.mark.parametrize("N", [5, 10, 15, 20, 25, 30])
def test_solve(rng, N: int):
    A, b = random_instance(rng, N)
    enc_y = solve(A, b)
    np.testing.assert_allclose(A.get_raw_data() @ enc_y.matrix, b.get_raw_data())


def test_solve_stats(rng):
    N = 10
    A, b = random_instance(rng, N)
    enc_y = solve(A, b, error=1e-5)

    with track_queries() as tracker:
        enc_y.get()
        queries_y = tracker.get_stats(enc_y).quantum_expected_quantum_queries
        queries_A = tracker.get_stats(A).quantum_expected_quantum_queries
        queries_b = tracker.get_stats(b).quantum_expected_quantum_queries

    assert queries_y == 1
    expected_query_count_A = 2 * qlsa_query_count(
        N, max(np.linalg.cond(A.get_raw_data()), np.sqrt(12)), 1e-5
    )
    assert queries_A == expected_query_count_A
    assert queries_b == pytest.approx(2 * queries_A)
