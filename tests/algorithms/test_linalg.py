import numpy as np
import pytest

from qubrabench.algorithms.linalg import (
    qlsa_query_count_with_failure_probability,
    solve,
)
from qubrabench.datastructures.qndarray import Qndarray, array


def random_instance(rng, N: int) -> tuple[Qndarray, Qndarray]:
    A = array(rng.random(size=(N, N)))
    b = array(rng.random(size=N))
    return A, b


@pytest.mark.parametrize("N", [5, 10, 15, 20, 25, 30])
def test_solve(rng, N: int):
    A, b = random_instance(rng, N)
    enc_y = solve(A, b, max_fail_probability=0, precision=1e-5)
    np.testing.assert_allclose(A.get_raw_data() @ enc_y.matrix, b.get_raw_data())


def test_solve_stats(rng):
    N = 10
    max_fail_probability = 1 / 3
    precision = 1e-5

    A, b = random_instance(rng, N)
    enc_y = solve(
        A,
        b,
        max_fail_probability=max_fail_probability,
        precision=precision,
    )
    enc_y.access()

    queries_A = A.stats.quantum_expected_quantum_queries
    queries_b = b.stats.quantum_expected_quantum_queries

    expected_query_count_A = 2 * qlsa_query_count_with_failure_probability(
        block_encoding_subnormalization_A=N,
        condition_number_A=max(np.linalg.cond(A.get_raw_data()), np.sqrt(12)),
        l1_precision=precision,
        max_fail_probability=max_fail_probability,
    )
    assert expected_query_count_A == pytest.approx(46947402.37303277)
    assert queries_A == pytest.approx(expected_query_count_A)
    assert queries_b == pytest.approx(2 * queries_A)
