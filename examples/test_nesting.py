import numpy as np
import pytest
from nesting import (
    generate_random_matrix_of_condition_number,
    has_solution_large_entry,
    has_solution_large_entry_quantum,
)

from qubrabench.benchmark import QueryStats
from qubrabench.datastructures.qndarray import Qndarray


@pytest.mark.parametrize("N", [4, 10, 20])
def test_example_planted(rng, N: int):
    for _ in range(20):
        A = rng.random((N, N))
        x = rng.random(N)
        x /= np.linalg.norm(x)
        b = A @ x

        expected = has_solution_large_entry(A, b)
        actual = has_solution_large_entry_quantum(A, b)
        assert expected == actual


def test_example_planted_stats(rng):
    N = 20
    A = rng.random((N, N))
    x = rng.random(N)
    x /= np.linalg.norm(x)
    b = A @ x

    assert np.linalg.cond(A) == pytest.approx(406.1253740708785)

    A = Qndarray(A)
    _ = has_solution_large_entry_quantum(A, b)

    # quantum_expected and quantum_worst_case use different algorithms
    # which one is better depends on the input data (i.e. if there are more than one solutions, the expected case algorighm is usually faster).
    assert A.stats == QueryStats(
        classical_actual_queries=0,
        classical_expected_queries=0.0,
        quantum_expected_classical_queries=0.0,
        quantum_expected_quantum_queries=pytest.approx(1.3335885731302106e27),
        quantum_worst_case_classical_queries=0.0,
        quantum_worst_case_quantum_queries=pytest.approx(4.7511657956811674e26),
    )


@pytest.mark.parametrize("N", [4, 8, 16, 50])
@pytest.mark.parametrize("kappa", [10, 100, 1000])
def test_random_matrix_with_condition_number(rng, N: int, kappa: float):
    for _ in range(5):
        U = generate_random_matrix_of_condition_number(N, kappa, rng=rng)
        assert np.linalg.cond(U) == pytest.approx(kappa)
