from typing import Hashable, Optional

import numpy as np

from ..benchmark import BlockEncoding, QueryStats, quantum_subroutine
from ..datastructures.matrix import block_encoding_of_matrix, ndarray

__all__ = ["solve", "qlsa"]


@quantum_subroutine
def solve(
    A: ndarray,
    b: ndarray,
    *,
    error: Optional[float] = None,
    condition_number_A: Optional[float] = None,
) -> BlockEncoding:
    enc_A = block_encoding_of_matrix(A, eps=0)
    enc_b = block_encoding_of_matrix(b, eps=0)
    return qlsa(enc_A, enc_b, error=error, condition_number_A=condition_number_A)


@quantum_subroutine
def qlsa(
    A: BlockEncoding,
    b: BlockEncoding,
    *,
    error: Optional[float] = None,
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

    # costs: dict[Hashable, QueryStats] = {}

    if error is not None:
        if condition_number_A is None:
            condition_number_A = np.linalg.cond(A.matrix)

        if not np.isclose(A.error, 0):
            raise ValueError(
                f"solve expects a zero-error block-encoding of A, but input has an error of {A.error}"
            )
        if not np.isclose(b.error, 0):
            raise ValueError(
                f"solve expects a zero-error block-encoding of b, but input has an error of {b.error}"
            )

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
        q = q_star / (0.39 - 0.201 * eps)
    else:
        q = 0

    y = np.linalg.solve(A.matrix, b.matrix)
    return BlockEncoding(
        y, alpha=np.linalg.norm(y), error=error, uses=[(A, q), (b, 2 * q)]
    )
