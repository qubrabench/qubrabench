import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy

import qubrabench.algorithms as qba
from qubrabench.benchmark import QueryStats, track_queries
from qubrabench.datastructures.qndarray import (
    Qndarray,
    block_encode_matrix,
    state_preparation_unitary,
)


def classical_algorithm(A: npt.NDArray, b: npt.NDArray) -> bool:
    assert A.ndim == 2 and A.shape[0] == A.shape[1]
    N = A.shape[0]
    x = np.linalg.solve(A, b)
    return any(np.abs(x[i]) >= 0.5 for i in range(N))


def example(A: npt.NDArray, b: npt.NDArray, *, error: float = 1e-5) -> bool:
    r"""Find x s.t. Ax = b, and check if x has an entry x_i s.t. $x_i >= 0.5$"""
    assert A.ndim == 2 and A.shape[0] == A.shape[1]
    N = A.shape[0]
    eps: float = 1e-5

    A = block_encode_matrix(A, eps=0)
    b = state_preparation_unitary(b, eps=0)
    x = qba.linalg.solve(A, b, error=eps / 2)

    return (
        qba.search.search(
            range(N),
            key=(
                lambda i: qba.amplitude.estimate_amplitude(
                    x, i, precision=eps, failure_probability=error
                )
                >= 0.25
            ),
            max_failure_probability=error,
            max_classical_queries=0,
        )
        is not None
    )


def generate_random_matrix_of_condition_number(
    N: int, condition_number: float, *, rng: np.random.Generator, l2_norm: float = 1
):
    r"""Generate a random N x N matrix with bounded condition number.

    Args:
        N: dimension of the matrix
        condition_number: approximate condition number of the required matrix
        rng: random generator
        l2_norm: upper bound on the spectral norm of the output matrix

    Returns:
        Random N x N matrix with condition number atmost the given value.
    """
    U = scipy.stats.special_ortho_group.rvs(N, random_state=rng)
    D = rng.random(size=N)
    D_scaled = np.interp(D, (D.min(), D.max()), (l2_norm / condition_number, l2_norm))
    V = scipy.stats.special_ortho_group.rvs(N, random_state=rng)

    return U @ np.diag(D_scaled) @ V


def run(
    N: int,
    condition_number: float,
    *,
    error: float,
    rng: np.random.Generator,
    n_runs: int = 5,
) -> pd.DataFrame:
    """Benchmark the example.

    Args:
        N: dimension of A and b.
        condition_number: condition number of A
        error: upper bound on failure probability of the entire algorithm
        n_runs: number of repetitions
        rng: random generator
    """
    history = []
    for _ in range(n_runs):
        A = generate_random_matrix_of_condition_number(N, condition_number, rng=rng)
        b = rng.random(N)

        with track_queries() as tracker:
            A = Qndarray(A)
            b = Qndarray(b)
            _ = example(A, b, error=error)

            stats_A: QueryStats = tracker.get_stats(A)
            stats_b: QueryStats = tracker.get_stats(b)

            history.append(
                {
                    "N": N,
                    "k_A": condition_number,
                    "queries_A": stats_A.quantum_expected_quantum_queries,
                    "queries_b": stats_b.quantum_expected_quantum_queries,
                }
            )

    df = pd.DataFrame(
        [list(row.values()) for row in history], columns=list(history[0].keys())
    )
    return df
