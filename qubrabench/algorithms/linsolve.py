from typing import Optional
import numpy as np

from ..benchmark import _BenchmarkManager, QueryStats
from ..datastructures.blockencoding import BlockEncoding


def linear_solver(
    A: BlockEncoding,
    b: BlockEncoding,
    *,
    error: Optional[float] = None,
    condition_number_A: Optional[float] = None,
):
    """Quantum Linear Solver as described in https://arxiv.org/abs/2305.11352.

    Given block-encodings for $A$ and $b$, find an approximation for $A^{-1}b$.

    Args:
        A: block-encoded input matrix
        b: block-encoded input vector
        error: approximation factor of the solution vector
        condition_number_A: An upper-bound on the condition number of A. Optional, will be calculated if not provided.

    Returns:
        Block-encoded solution vector.

    Raises:
        ValueError: when in benchmarking mode but error is not provided.
    """

    if _BenchmarkManager.is_benchmarking():
        if error is None:
            raise ValueError(
                "linear_solver() parameter 'error' not provided, cannot compute quantum query statistics"
            )

        if condition_number_A is None:
            condition_number_A = np.linalg.cond(A.matrix)

        condition_number_A = max(condition_number_A, np.sqrt(12))
        error = min(error, 0.24)

        alpha, k, eps = A.alpha, condition_number_A, error

        q_star_term_1 = (
            (1741 * alpha * eps / 500)
            * np.sqrt(k**2 + 1)
            * (
                (133 / 125 + 4 / (25 * np.power(k, 1.0 / 3)))
                * np.pi
                * np.log(2 * k + 3)
                + 1
            )
        )
        q_star_term_2 = (
            (351 / 50)
            * np.log(2 * k + 3) ** 2
            * (np.log(451 * np.log(2 * k + 3) ** 2 / eps) + 1)
        )
        q_star_term_3 = alpha * k * np.log(32 / eps)

        q_star = q_star_term_1 + q_star_term_2 + q_star_term_3

        _BenchmarkManager.current_frame()._add_quantum_expected_queries(
            hash(A),
            base_stats=QueryStats(
                quantum_expected_quantum_queries=1
            ),  # TODO: maybe use the cost of implementing A?
            queries_quantum=q_star,
        )
        _BenchmarkManager.current_frame()._add_quantum_expected_queries(
            hash(b),
            base_stats=QueryStats(
                quantum_expected_quantum_queries=1
            ),  # TODO: maybe use the cost of implementing A?
            queries_quantum=2 * q_star,
        )

    x = np.linalg.solve(A.matrix, b.matrix)
    return BlockEncoding(x, np.linalg.norm(x), error)
