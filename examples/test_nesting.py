import numpy as np
import pytest
from nesting import (
    classical_algorithm,
    generate_random_matrix_of_condition_number,
    quantum_algorithm,
)

from qubrabench.benchmark import QueryStats, track_queries
from qubrabench.datastructures.qndarray import Qndarray


@pytest.mark.parametrize("N", [4, 10, 20])
def test_example_planted(rng, N: int):
    for _ in range(20):
        A = rng.random((N, N))
        x = rng.random(N)
        x /= np.linalg.norm(x)
        b = A @ x

        expected = classical_algorithm(A, b)
        actual = quantum_algorithm(A, b)
        assert expected == actual


def test_example_planted_stats(rng):
    N = 20
    A = rng.random((N, N))
    x = rng.random(N)
    x /= np.linalg.norm(x)
    b = A @ x

    assert np.linalg.cond(A) == 406.1253740708785

    with track_queries() as tracker:
        A = Qndarray(A)
        _ = quantum_algorithm(A, b)

        stats = tracker.get_stats(A)
        assert stats == QueryStats(
            classical_actual_queries=0,
            classical_expected_queries=0,
            quantum_expected_classical_queries=0,
            quantum_expected_quantum_queries=pytest.approx(1.2866482371334009e23),
        )


@pytest.mark.parametrize("N", [4, 8, 16, 50, 100, 200])
@pytest.mark.parametrize("kappa", [10, 100, 1000])
def test_random_matrix_with_condition_number(rng, N: int, kappa: float):
    for _ in range(5):
        U = generate_random_matrix_of_condition_number(N, kappa, rng=rng)
        assert np.linalg.cond(U) == pytest.approx(kappa)