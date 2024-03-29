from typing import Optional

import numpy as np

from ..benchmark import BlockEncoding, quantum_subroutine
from ..datastructures.qndarray import (
    QNDArrayLike,
    block_encode_matrix,
    state_preparation_unitary,
)

__all__ = ["solve", "qlsa"]


@quantum_subroutine
def solve(
    A: QNDArrayLike,
    b: QNDArrayLike,
    *,
    max_fail_probability: float,
    precision: float,
    condition_number_A: Optional[float] = None,
) -> BlockEncoding:
    """Solve the linear system Ax = b, producing a quantum state encoding x.

    See `qlsa` for query costs.

    Args:
        A: input matrix
        b: input vector
        max_fail_probability: upper bound on probability of failure
        precision: the l1 norm distance of the output unit vector to the actual solution (scaled to unit)
        condition_number_A: An upper-bound on the condition number of A. Optional, will be calculated if not provided.

    Returns:
        Block-encoded solution vector.
    """
    enc_A = block_encode_matrix(A, eps=0)
    enc_b = state_preparation_unitary(b, eps=0)
    return qlsa(
        enc_A,
        enc_b,
        max_fail_probability=max_fail_probability,
        precision=precision,
        condition_number_A=condition_number_A,
    )


@quantum_subroutine
def qlsa(
    A: BlockEncoding,
    b: BlockEncoding,
    *,
    max_fail_probability: float,
    precision: float,
    condition_number_A: Optional[float] = None,
) -> BlockEncoding:
    """Quantum Linear Solver as described in https://doi.org/10.48550/arXiv.2305.11352

    Given block-encodings for $A$ and $b$, find an approximation of $y$ satisfying $Ay = b$.

    Args:
        A: block-encoded input matrix
        b: block-encoded input vector
        max_fail_probability: probability of failure
        precision: the l1 norm distance of the output unit vector to the actual solution (scaled to unit)
        condition_number_A: An upper-bound on the condition number of A. Optional, will be calculated if not provided.

    Raises:
        ValueError: if input block-encodings do not have zero precision.

    Returns:
        Block-encoded solution vector.
    """

    if not np.isclose(A.precision, 0):
        raise ValueError(
            f"solve expects a zero-error block-encoding of A, but input has an error of {A.precision}"
        )
    if not np.isclose(b.precision, 0):
        raise ValueError(
            f"solve expects a zero-error block-encoding of b, but input has an error of {b.precision}"
        )

    if condition_number_A is None:
        condition_number_A = np.linalg.cond(A.matrix)

    condition_number_A = max(condition_number_A, np.sqrt(12))
    precision = min(precision, 0.24)

    q = qlsa_query_count_with_failure_probability(
        A.subnormalization_factor,
        condition_number_A,
        precision,
        max_fail_probability,
    )

    y = np.linalg.solve(A.matrix, b.matrix)
    return BlockEncoding(
        y,
        subnormalization_factor=np.linalg.norm(y),
        precision=precision,
        uses=[(A, q), (b, 2 * q)],
    )


def qlsa_query_count_with_failure_probability(
    block_encoding_subnormalization_A: float,
    condition_number_A: float,
    l1_precision: float,
    max_fail_probability: float,
) -> float:
    """Query cost expression from https://doi.org/10.48550/arXiv.2305.11352, accounting for arbitrary success probability."""
    q_star = qlsa_query_count(
        alpha=block_encoding_subnormalization_A,
        kappa=condition_number_A,
        eps=l1_precision,
    )

    # Q* queries for success probability at least (0.39 - 0.201 * \epsilon)
    expected_success_probability = 0.39 - 0.201 * l1_precision

    # repeat if neccessary
    n_repeat_expected: int
    if 1 - max_fail_probability <= expected_success_probability:
        n_repeat_expected = 1
    else:
        n_repeat_expected = np.emath.logn(
            1 - expected_success_probability,
            max_fail_probability,
        )

    return q_star * n_repeat_expected


def qlsa_query_count(alpha: float, kappa: float, eps: float) -> float:
    """Query cost expression from Theorem 1 of https://doi.org/10.48550/arXiv.2305.11352"""
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

    return q_star
