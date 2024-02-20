from typing import Optional

import numpy as np

from ..benchmark import BlockEncoding, quantum_subroutine

__all__ = ["solve"]


@quantum_subroutine
def solve(
    A: BlockEncoding,
    b: BlockEncoding,
    *,
    error: float = 0.24,
    condition_number_A: Optional[float] = None,
) -> BlockEncoding:
    """Quantum Linear Solver as described in https://arxiv.org/abs/2305.11352.

    Given block-encodings for $A$ and $b$, find an approximation of $y$ satisfying $Ay = b$.

    Args:
        A: block-encoded input matrix
        b: block-encoded input vector
        error: approximation factor of the solution vector
        condition_number_A: An upper-bound on the condition number of A. Optional, will be calculated if not provided.

    Returns:
        Block-encoded solution vector.
    """

    if not np.isclose(A.error, 0):
        raise ValueError(
            f"solve expects a zero-error block-encoding of A, but input has an error of {A.error}"
        )
    if not np.isclose(b.error, 0):
        raise ValueError(
            f"solve expects a zero-error block-encoding of b, but input has an error of {b.error}"
        )

    if condition_number_A is None:
        condition_number_A = np.linalg.cond(A.matrix)

    condition_number_A = max(condition_number_A, np.sqrt(12))
    error = min(error, 0.24)

    q = qlsa_query_count(A.alpha, condition_number_A, error)

    y = np.linalg.solve(A.matrix, b.matrix)
    return BlockEncoding(
        y, alpha=np.linalg.norm(y), error=error, uses=[(A, q), (b, 2 * q)]
    )


def qlsa_query_count(alpha: float, kappa: float, eps: float) -> float:
    q_star_term_1 = (
        (1741 * alpha * np.e / 500)
        * np.sqrt(kappa**2 + 1)
        * (
            (133 / 125 + 4 / (25 * np.power(kappa, 1.0 / 3)))
            * np.pi
            * np.log(2 * kappa + 3)
            + 1
        )
    )
    q_star_term_2 = (
        (351 / 50)
        * np.log(2 * kappa + 3) ** 2
        * (np.log(451 * np.log(2 * kappa + 3) ** 2 / eps) + 1)
    )
    q_star_term_3 = alpha * kappa * np.log(32 / eps)

    q_star = q_star_term_1 + q_star_term_2 + q_star_term_3
    q = q_star / (0.39 - 0.201 * eps)

    return q
