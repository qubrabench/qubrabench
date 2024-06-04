"""An example using nested subroutine calls, inspired by Simplex's pivot finding method.

Problem: Find x s.t. Ax = b, and check if x has an entry x_i s.t. $x_i >= 0.5$
"""

import numpy
import numpy as np
import pandas as pd
import scipy

# from numpy.typing import NDArray
from numpy import ndarray

import qubrabench as qb
from qubrabench.benchmark import _BenchmarkManager


def has_solution_large_entry(A: ndarray, b: ndarray) -> bool:
    assert A.ndim == 2 and A.shape[0] == A.shape[1]
    n = A.shape[0]
    x = numpy.linalg.solve(A, b)
    norm_x = numpy.linalg.norm(x)
    for i in range(n):
        if np.abs(x[i]) >= 0.5 * norm_x:
            return True
    return False


def has_solution_large_entry_quantum(
    A: ndarray, b: ndarray, *, max_fail_prob: float = 1e-5
) -> bool:
    assert A.ndim == 2 and A.shape[0] == A.shape[1]
    n = A.shape[0]

    x = qb.linalg.solve(
        A,
        b,
        precision=1e-9,
        max_fail_probability=max_fail_prob / (4 * n * n),
    )

    return (
        qb.search(
            range(n),
            key=(
                lambda i: qb.estimate_amplitude(
                    x,
                    i,
                    precision=2e-9,
                    max_fail_probability=max_fail_prob / (4 * n),
                )
                >= 0.25
            ),
            max_fail_probability=max_fail_prob / 2,
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

        A = qb.array(A)
        b = qb.array(b)
        _ = has_solution_large_entry_quantum(A, b, max_fail_prob=error)

        if _BenchmarkManager.is_tracking():
            history.append(
                {
                    "N": N,
                    "k_A": condition_number,
                    "queries_A": A.stats.quantum_expected_quantum_queries,
                    "queries_b": b.stats.quantum_expected_quantum_queries,
                }
            )

    if len(history) > 0:
        return pd.DataFrame(
            [list(row.values()) for row in history], columns=list(history[0].keys())
        )
