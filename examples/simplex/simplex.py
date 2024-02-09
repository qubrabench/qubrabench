"""An end-to-end implementation of the simplex algorithm by described in the paper "Fast quantum subroutines for the simplex method" https://arxiv.org/pdf/1910.10649.pdf. """

from enum import Enum
from typing import Optional, Sequence, TypeAlias

import numpy as np
import numpy.typing as npt

import qubrabench.algorithms as qba
from qubrabench.benchmark import BlockEncoding, quantum_subroutine
from qubrabench.datastructures.qndarray import (
    Qndarray,
    block_encode_matrix,
    state_preparation_unitary,
)

Matrix: TypeAlias = npt.NDArray[np.float_] | Qndarray
"""n x m real matrix"""

Vector: TypeAlias = npt.NDArray[np.float_] | Qndarray
"""n x 1 real vector"""

Basis: TypeAlias = Sequence[int]
"""array of column indices"""


@quantum_subroutine
def solve_linear_system(A: Matrix, b: Vector, *, eps: float) -> BlockEncoding:
    enc_A = block_encode_matrix(A, eps=0)
    enc_b = state_preparation_unitary(b, eps=0)
    return qba.linalg.solve(enc_A, enc_b, error=eps)


@quantum_subroutine
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
        uses=[(U, 1), (V, 1)],
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
    one_shot_k = np.zeros(U.matrix.shape)
    one_shot_k[k] = 1
    V = state_preparation_unitary(one_shot_k, eps=0)

    a = qba.amplitude.estimate_amplitude(
        Interfere(U, V), k, precision=epsilon, failure_probability=3 / 4
    )
    return a >= 0.5


def SignEstNFP(U: BlockEncoding, k: int, epsilon) -> bool:
    r"""Algorithm 11 [Q->C] Sign estimation routine with no false positives

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
    raise NotImplementedError(
        "this method is only provided as an example usecase,"
        "but is not numerically stable enough to solve a linear program."
    )

    # TODO compute valid initial basic solution
    B: Basis = np.arange(A.shape[0])

    while True:
        result = SimplexIter(A, B, b, c, epsilon=1e-5, delta=1e-5)
        if result == ResultFlag.Optimal:
            break
        if result == ResultFlag.Unbounded:
            return None

    qlsa = solve_linear_system(A[:, B], b, eps=1e-5)
    return qlsa.matrix


def SimplexIter(
    A: Matrix, B: Basis, b: Vector, c: Vector, epsilon: float, delta: float
) -> ResultFlag:
    """Algorithm 1 [C->C]: Run one iteration of the simplex method

    Args:
        A: n x m matrix, n >= m
        B: basis of size m
        b: RHS vector (i.e. $Ax = b$) of size m
        c: cost vector of size n
        epsilon: precision parameter
        delta: precision parameter

    Returns:
        Optimal - solution is found
        Unbounded - no bounded solution exists
        Updated - pivot was performed, and more iterations may be neccessary.
    """
    raise NotImplementedError(
        "this method is only provided as an example usecase,"
        "but is not numerically stable enough to solve a linear program."
    )
    # Normalize c so that \norm{c_B} = 1
    c /= np.linalg.norm(c[B])

    # Normalize A so that \norm{A_B} <= 1
    scale_A = np.linalg.norm(A[:, B])
    A /= scale_A
    b /= scale_A

    # if IsOptimal(A, B, c, epsilon):
    #     return ResultFlag.Optimal

    k = FindColumn(A, B, c, epsilon)
    if k is None:
        return ResultFlag.Optimal

    if IsUnbounded(A[:, B], A[:, k], delta):
        return ResultFlag.Unbounded

    el = FindRow(A[:, B], A[:, k], b, delta)

    B[el] = k

    return ResultFlag.BasisUpdated


@quantum_subroutine
def direct_sum_of_ndarrays(a: Matrix | Vector, b: Matrix | Vector) -> BlockEncoding:
    if a.ndim != b.ndim:
        raise ValueError(
            f"number of dimensions of both arguments must match! instead got {a.ndim} and {b.ndim}"
        )
    rank = a.ndim
    if rank not in [1, 2]:
        raise ValueError("direct sum only works for 1D or 2D matrices")

    a = Qndarray(a)
    b = Qndarray(b)

    res: npt.NDArray
    alpha: float
    if rank == 2:
        res = np.block(
            [
                [a.get_raw_data(), np.zeros((a.shape[0], b.shape[1]))],
                [np.zeros((b.shape[0], a.shape[1])), b.get_raw_data()],
            ]
        )
        alpha = np.sqrt(res.size)
    else:
        res = np.block([a.get_raw_data(), b.get_raw_data()])
        alpha = np.linalg.norm(res)

    return BlockEncoding(res, alpha=alpha, error=0, uses=[(a, rank), (b, rank)])


@quantum_subroutine
def RedCost(
    A_B: Matrix, A_k: Vector, c: Vector, k: int, B: Basis, epsilon: float
) -> BlockEncoding:
    """Algorithm 4 [C->Q]: Determining the reduced cost of a column"""
    lhs_mat = direct_sum_of_ndarrays(A_B, np.array([[1]]))
    rhs_vec = direct_sum_of_ndarrays(A_k, c[k : k + 1])
    sol = qba.linalg.solve(lhs_mat, rhs_vec, error=epsilon / (10 * np.sqrt(2)))

    c = Qndarray(c)

    return BlockEncoding(
        np.array([np.inner(np.block([-c.get_raw_data()[B], 1]), sol.matrix)]),
        alpha=sol.alpha * np.sqrt(2),
        error=sol.error,
        uses=[(sol, 1), (c, 1)],
    )


def CanEnter(
    A_B: Matrix, A_k: Vector, c: Vector, k: int, B: Basis, epsilon: float
) -> bool:
    r"""Algorithm 5 [C->C]: Determine is a column is eligible to enter the basis

    Args:
        A_B: Basic square sub-matrix of A with norm at most 1
        A_k: nonbasic k-th column
        c: cost vector s.t. $\norm{c_B} = 1$
        k: index of nonbasic column
        B: Basis
        epsilon: precision

    Returns:
        1 if the nonbasic column $A_k$ has reduced cost $< \epsilon$;
        0 otherwise
    """
    U_r = RedCost(A_B, A_k, c, k, B, epsilon)
    return not SignEstNFN(U_r, 0, 11 * epsilon / (10 * np.sqrt(2)))


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
        non_basic,
        key=lambda k: CanEnter(A[:, B], A[:, k], c, k, B, epsilon),
        error=1.0 / 3.0,
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
    U_LS = solve_linear_system(A_B, A_k, eps=0.1 * delta)
    result = qba.search.search(
        range(m),
        key=lambda el: SignEstNFN(
            U_LS, el, 0.9 * delta
        ),  # and success flag of QLSA = 1
    )
    return result is None


def FindRow(A_B: Matrix, A_k: Vector, b: Vector, delta: float) -> int:
    """Algorithm 9 [C->C]: Determine the basic variable (row) leaving the basis

    Args:
        A_B: basis columns of A, with norm atmost 1.
        A_k: nonbasic column
        b: RHS
        delta: precision

    Returns:
        index of the row that should leave the basis according to the ratio test,
        with bounded probability
    """

    delta_scaled = delta / np.linalg.norm(np.linalg.solve(A_B, A_k))
    m = A_B.shape[0]

    def U(r: float) -> Optional[int]:
        qlsa = solve_linear_system(A_B, b - r * A_k, eps=delta_scaled)
        row = qba.search.search(
            range(1, m + 1),
            key=lambda el: not SignEstNFN(qlsa, el, epsilon=delta / 2),
            error=delta_scaled,
        )
        return row

    # binary search for `r`
    r_low, r_high = 0.0, 100.0  # TODO compute proper starting upper-bound
    while r_high - r_low > delta_scaled / 2:
        r = (r_low + r_high) / 2
        if U(r) is not None:
            r_high = r
        else:
            r_low = r

    row = U(r_high)
    assert row is not None
    return row


def IsFeasible(A_B: Matrix, b: Vector, delta: float) -> bool:
    r"""Algorithm 10 [C->C]: Determine if a basic solution is feasible

    Args:
        A_B: matrix of basis columns, with norm at most 1
        b: rhs vector
        delta: precision

    Returns:
        Whether $A_B^{-1} b \ge  −\delta 1_m$, with bounded probability.
    """
    m = A_B.shape[0]
    delta_scaled = delta / np.linalg.norm(np.linalg.solve(A_B, b))

    qlsa = solve_linear_system(A_B, b, eps=delta_scaled * 0.1)
    result = qba.search.search(
        range(1, m + 1),
        key=lambda el: not SignEstNFP(qlsa, el, epsilon=delta_scaled * 0.45),
        error=1e-5,  # TODO check
    )
    return result is not None
