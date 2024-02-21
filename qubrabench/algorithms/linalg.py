from typing import Optional

import numpy as np

from ..benchmark import BlockEncoding, quantum_subroutine

__all__ = ["solve"]


@quantum_subroutine
def solve(
    A: BlockEncoding,
    b: BlockEncoding,
    *,
    max_failure_probability: float,
    precision: float,
    condition_number_A: Optional[float] = None,
) -> BlockEncoding:
    """Quantum Linear Solver as described in https://doi.org/10.48550/arXiv.2305.11352


    Given block-encodings for $A$ and $b$, find an approximation of $y$ satisfying $Ay = b$.

    Args:
        A: block-encoded input matrix
        b: block-encoded input vector
        max_failure_probability: probability of failure
        precision: how close should the solution be
        condition_number_A: An upper-bound on the condition number of A. Optional, will be calculated if not provided.

    Raises:
        ValueError: Raised when (1) any block-encoding used is expected to have zero-error, but doesn't
        (2) max_failure_probability provided is too low

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

    q = qlsa_query_count_with_failure_probability(
        A.subnormalization_factor,
        condition_number_A,
        precision,
        max_failure_probability,
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
    max_failure_probability: float,
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
    if 1 - max_failure_probability <= expected_success_probability:
        n_repeat_expected = 1
    else:
        n_repeat_expected = np.emath.logn(
            1 - expected_success_probability,
            max_failure_probability,
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
