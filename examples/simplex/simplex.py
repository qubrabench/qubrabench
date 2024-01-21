"""An end-to-end implementation of the simplex algorithm by described in the paper "Fast quantum subroutines for the simplex method" https://arxiv.org/pdf/1910.10649.pdf. """
from typing import TypeAlias, Optional
from enum import Enum
import numpy as np
from numpy.typing import NDArray

from qubrabench.benchmark import QueryStats
from qubrabench.datastructures.blockencoding import (
    BlockEncoding,
    block_encoding_of_matrix,
)
import qubrabench.algorithms as qba

# TODO: success flag of QLSA

Matrix: TypeAlias = NDArray[np.float_]
"""n x m real matrix"""

Vector: TypeAlias = NDArray[np.float_]
"""n x 1 real vector"""

Basis: TypeAlias = NDArray[np.int_]
"""array of column indices"""


class ResultFlag(Enum):
    Optimal = 0
    Unbounded = 1
    BasisUpdated = 2


def Simplex(A: Matrix, b: Vector, c: Vector) -> Optional[Vector]:
    r"""Simplex method for linear optimization

    Find an $x \in \mathbb{R}^n$ such that $Ax = b$ and minimizing $c^T x$.

    Args:
        A: an $m \times n$ real matrix
        b: an $m$-dimensional column vector
        c: an $n$-dimensional column vector

    Returns:
        x: an $n$-dimensional column vector solution to the above optimization.
    """
    # TODO compute valid initial basic solution
    B: Basis = np.arange(A.shape[0])

    while True:
        result = SimplexIter(A, B, b, c)
        if result == ResultFlag.Optimal:
            break
        if result == ResultFlag.Unbounded:
            return None

    return np.linalg.solve(A[:, B], b)  # TODO quantum costs


def SimplexIter(
    A: Matrix, B: Basis, b: Vector, c: Vector, epsilon: float, delta: float, t: float
) -> ResultFlag:
    """Algorithm 1 [C->C]: Run one iteration of the simplex method

    Args:
        A: n x m matrix, n >= m
        B: basis of size m
        b: RHS vector (i.e. $Ax = b$) of size m
        c: cost vector of size n
        epsilon: precision parameter
        delta: precision parameter
        t: precision parameter

    Returns:
        Optimal - solution is found
        Unbounded - no bounded solution exists
        Updated - pivot was performed, and more iterations may be neccessary.
    """
    # Normalize c so that \norm{c_B} = 1
    c /= np.linalg.norm(c[B])

    # Normalize A so that \norm{A_B} <= 1
    A /= np.linalg.norm(A[:, B])

    if IsOptimal(A, B, c, epsilon):
        return ResultFlag.Optimal

    k = FindColumn(A, B, c, epsilon)
    assert (
        k is not None
    ), "FindColumn should not return None when IsOptimal returns False!"

    if IsUnbounded(A[:, B], A[:, k], delta):
        return ResultFlag.Unbounded

    el = FindRow(A[:, B], A[:, k], b, delta, t)

    B[el] = k

    return ResultFlag.BasisUpdated


def Interfere(U: BlockEncoding, V: BlockEncoding) -> BlockEncoding:
    r"""Algorithm 2 [Q->Q]: Interference Routine

    Given unitaries $U, V$ such that $U\ket{0} = \sum_j \alpha_j \ket j$ and $U\ket{0} = \sum_j \beta_j \ket j$,
    returns a unitary that prepares

        $$\frac12 \ket0 \otimes \sum_j (\alpha_j + \beta_j) \ket j + \frac12 \ket1 \otimes \sum_j (\beta_j - \alpha_j) \ket j$$

    Args:
        U: encodes vector alpha
        V: encodes vector beta

    Returns:
        Unitary that prepares above described output vector
    """
    if U.matrix.shape != V.matrix.shape:
        raise ValueError("U and V must block-encode same sized matrices")

    u = U.matrix / U.alpha
    v = V.matrix / V.alpha
    return BlockEncoding(
        0.5 * np.concatenate((u + v, v - u)),
        alpha=1,
        error=max(U.error / U.alpha, V.error / V.alpha),
        costs={
            U: QueryStats(quantum_expected_quantum_queries=1),
            V: QueryStats(quantum_expected_quantum_queries=1),
        },
    )


def SignEstNFN(U: BlockEncoding, k: int, epsilon) -> bool:
    r"""Algorithm 3 [Q->C]: Sign estimation routine with no false negatives

    Args:
        U: Block-encodes vector $\alpha$ of length 2^q
        k: index between 0 and 2^q - 1
        epsilon: precision

    Returns:
        True if $\alpha_k \ge -\epsilon$, with probability at least 3/4.
    """
    # TODO quantum query costs
    if U.matrix[k] >= -epsilon:
        return True
    return False


def SignEstNFP(U: BlockEncoding, k: int, epsilon) -> bool:
    r"""Algorithm 11 [Q->Q] Sign estimation routine with no false positives

    Args:
        U: Block-encodes vector $\alpha$ of length 2^q
        k: index between 0 and 2^q - 1
        epsilon: precision

    Returns:
        False if $\alpha_k \le -\epsilon$, with probability at least 3/4.
    """
    # TODO quantum query costs
    if U.matrix[k] <= -epsilon:
        return False
    return True


def RedCost(
    A_B: Matrix, A_k: Vector, c: Vector, epsilon: float
) -> Optional[BlockEncoding]:
    """Algorithm 4 [C->Q]: Determining the reduced cost of a column"""
    raise NotImplementedError
    # lhs_mat = BlockEncoding()
    # rhs_vec = BlockEncoding()
    # return qba.linalg.solve(lhs_mat, rhs_vec, error=epsilon / (10 * np.sqrt(2)))


def CanEnter(A_B: Matrix, A_k: Vector, c: Vector, epsilon: float) -> bool:
    r"""Algorithm 5 [C->C]: Determine is a column is eligible to enter the basis

    Args:
        A_B: Basic square sub-matrix of A with norm at most 1
        A_k: nonbasic k-th column
        c: cost vector s.t. $\norm{c_B} = 1$
        epsilon: precision

    Returns:
        1 if the nonbasic column $A_k$ has reduced cost $< \epsilon$;
        0 otherwise
    """
    U_r = RedCost(A_B, A_k, c, epsilon)
    if U_r is None:
        return False
    return SignEstNFN(U_r, 0, 11 * epsilon / (10 * np.sqrt(2)))


def FindColumn(A: Matrix, B: Basis, c: Vector, epsilon: float) -> Optional[int]:
    r"""Algorithm 6 [C->C]: Determine the next column to enter the basis

    Args:
        A: matrix s.t. $\norm{A_B} \le 1$
        B: basis
        c: cost vector s.t. $\norm{c_B} = 1$
        epsilon: precision

    Returns:
        index of column $k$ with $\bar{c}_k < \epsilon \norm{(A_B^{-1} A_k, c_k)} if one exists, with bounded probability.
    """
    non_basic = set(range(A.shape[1])) - set(B)
    return qba.search.search(
        non_basic, key=lambda k: CanEnter(A[:, B], A[:, k], c, epsilon), error=1.0 / 3.0
    )


def IsOptimal(A: Matrix, B: Basis, c: Vector, epsilon: float) -> bool:
    r"""Algorithm 6b [C->C]

    Args:
        A: matrix s.t. $\norm{A_B} \le 1$
        B: basis
        c: cost vector s.t. $\norm{c_B} = 1$
        epsilon: precision
    """
    raise NotImplementedError


def SteepestEdgeCompare(A_B, A_j, A_k, c, epsilon) -> bool:
    """Algorithm 7 [C->C]: Compare the steepest edge price of two columns"""
    raise NotImplementedError


def IsUnbounded(A_B, A_k, delta) -> bool:
    r"""Algorithm 8 [C->C]: Determine if the problem is unbounded from below

    Args:
        A_B: basis columns of A, with norm at most 1.
        A_k: nonbasic column
        delta: precision

    Returns:
        True if $A_B^{-1} A_k < \delta \textbf{1}_m \norm{A_B^{-1} A_k}$
    """
    m = A_B.shape[0]
    enc_A_B = block_encoding_of_matrix(A_B, eps=0)
    enc_A_k = block_encoding_of_matrix(A_k, eps=0)
    U_LS = qba.linalg.solve(enc_A_B, enc_A_k, error=0.1 * delta)
    result = qba.search.search(
        range(m),
        key=lambda el: SignEstNFN(
            U_LS, el, 0.9 * delta
        ),  # and success flag of QLSA = 1
    )
    return result is None


def FindRow(A_B: Matrix, A_k: Vector, b: Vector, delta: float, t: float) -> int:
    """Algorithm 9 [C->C]: Determine the basic variable (row) leaving the basis

    Args:
        A_B: basis columns of A, with norm atmost 1.
        A_k: nonbasic column
        b: RHS
        delta: precision
        t: TODO what is this

    Returns:
        index of the row that should leave the basis according to the ratio test,
        with bounded probability
    """

    delta_signest = delta / np.linalg.norm(np.linalg.solve(A_B, A_k))

    def U(r: float) -> BlockEncoding:
        enc_A_B = block_encoding_of_matrix(A_B, eps=0)
        enc_rhs = block_encoding_of_matrix(b - r * A_k, eps=0)
        qlsa = qba.linalg.solve(enc_A_B, enc_rhs)
        # TODO amplitude estimation of
        #      SignEstNFN(qlsa, epsilon=delta_signest)
        raise NotImplementedError

    raise NotImplementedError


def IsFeasible(A_B, b, delta) -> bool:
    """Algorithm 10 [C->C]: Determine if a basic solution is feasible"""
    raise NotImplementedError
