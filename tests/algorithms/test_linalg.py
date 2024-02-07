import numpy as np
import pytest

from qubrabench.algorithms.linalg import solve
from qubrabench.benchmark import BlockEncoding, track_queries


def random_instance(rng, N: int) -> tuple[BlockEncoding, BlockEncoding]:
    A = rng.random(size=(N, N))
    b = rng.random(size=N)

    enc_A = BlockEncoding(A, alpha=np.linalg.norm(A), error=0)
    enc_b = BlockEncoding(b, alpha=np.linalg.norm(b), error=0)

    return enc_A, enc_b


@pytest.mark.parametrize("N", [5, 10, 15, 20, 25, 30])
def test_solve(rng, N: int):
    enc_A, enc_b = random_instance(rng, N)
    enc_y = solve(enc_A, enc_b)
    np.testing.assert_allclose(enc_A.matrix @ enc_y.matrix, enc_b.matrix)


def test_solve_stats(rng):
    N = 10
    enc_A, enc_b = random_instance(rng, N)
    enc_y = solve(enc_A, enc_b, error=1e-5)

    with track_queries() as tracker:
        enc_y.get()
        queries_y = tracker.get_stats(enc_y).quantum_expected_quantum_queries
        queries_A = tracker.get_stats(enc_A).quantum_expected_quantum_queries
        queries_b = tracker.get_stats(enc_b).quantum_expected_quantum_queries

    assert queries_y == 1
    assert queries_A == pytest.approx(840211.9909199567)
    assert queries_b == pytest.approx(2 * queries_A)
