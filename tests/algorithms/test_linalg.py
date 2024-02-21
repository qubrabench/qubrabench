import numpy as np
import pytest

from qubrabench.algorithms.linalg import (
    qlsa_query_count_with_failure_probability,
    solve,
)
from qubrabench.benchmark import track_queries
from qubrabench.datastructures.qndarray import (
    Qndarray,
    block_encode_matrix,
    state_preparation_unitary,
)


def random_instance(rng, N: int) -> tuple[Qndarray, Qndarray]:
    A = Qndarray(rng.random(size=(N, N)))
    b = Qndarray(rng.random(size=N))
    return A, b


@pytest.mark.parametrize("N", [5, 10, 15, 20, 25, 30])
def test_solve(rng, N: int):
    A, b = random_instance(rng, N)

    enc_A = block_encode_matrix(A, eps=0)
    enc_b = state_preparation_unitary(b, eps=0)
    enc_y = solve(enc_A, enc_b)

    np.testing.assert_allclose(enc_A.matrix @ enc_y.matrix, enc_b.matrix)


def test_solve_stats(rng):
    N = 10
    max_failure_probability = 1
    precision = 1e-5

    A, b = random_instance(rng, N)
    enc_A = block_encode_matrix(A, eps=0)
    enc_b = state_preparation_unitary(b, eps=0)
    enc_y = solve(
        enc_A,
        enc_b,
        max_failure_probability=max_failure_probability,
        precision=precision,
    )

    with track_queries() as tracker:
        enc_y.access()
        queries_y = tracker.get_stats(enc_y).quantum_expected_quantum_queries
        queries_A = tracker.get_stats(A).quantum_expected_quantum_queries
        queries_b = tracker.get_stats(b).quantum_expected_quantum_queries

    assert queries_y == 1
    expected_query_count_A = 2 * qlsa_query_count_with_failure_probability(
        block_encoding_subnormalization_A=N,
        condition_number_A=max(np.linalg.cond(A.get_raw_data()), np.sqrt(12)),
        l1_precision=precision,
        max_failure_probability=max_failure_probability,
    )
    assert expected_query_count_A == pytest.approx(21122805.429928094)
    assert queries_A == pytest.approx(expected_query_count_A)
    assert queries_b == pytest.approx(2 * queries_A)
