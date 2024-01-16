"""An end-to-end implementation of the simplex algorithm by described in the paper "Fast quantum subroutines for the simplex method" https://arxiv.org/pdf/1910.10649.pdf. """
from typing import Union, TypeAlias
from enum import Enum
import numpy as np
from numpy.typing import NDArray


from qubrabench.datastructures.blockencoding import BlockEncoding

Matrix: TypeAlias = NDArray[np.complex_]
"""n x m complex matrix"""

Vector: TypeAlias = NDArray[np.complex_]
"""n x 1 complex vector"""

Basis: TypeAlias = NDArray[np.int_]
"""array of column indices"""


class ResultFlag(Enum):
    Optimal = 0
    Unbounded = 1


def Simplex(A: Matrix, b: Vector, c: Vector):
    r"""Simplex method for linear optimization

    Find an $x \in \mathbb{R}^n$ such that $Ax = b$ and minimizing $c^T x$.

    Args:
        A: an $m \times n$ real matrix
        b: an $m$-dimensional column vector
        c: an $n$-dimensional column vector

    Returns:
        x: an $n$-dimensional column vector solution to the above optimization.
    """
    raise NotImplementedError


def SimplexIter(
    A: Matrix, B: Basis, c: Vector, epsilon: float, delta: float, t: float
) -> Union[ResultFlag, tuple[int, int]]:
    """Algorithm 1 [C->C]: Run one iteration of the simplex method

    Args:
        A: matrix
        B: basis
        c: cost vector
        epsilon: precision parameter
        delta: precision parameter
        t: precision parameter

    Returns:
        Flag "optimal", "unbounded",
        or a pair (k, l) where
            - k is a nonbasic variable with negative reduced cost
            - l is the basic variable that should leave the basis if k enters.
    """
    # TODO Normalize c so that \norm{c_B} = 1

    # TODO Normalize A so that \norm{A_B} <= 1

    if IsOptimal(A, B, epsilon):
        return ResultFlag.Optimal

    k = FindColumn(A, B, epsilon)

    if IsUnbounded(A_B, A_k, delta):
        return ResultFlag.Unbounded

    l = FindRow(A_B, A_k, b, delta)

    # Update basis B <- (B \ {B(l)}) \cup {k}

    return k, l


def Interfere(U: BlockEncoding, V: BlockEncoding) -> BlockEncoding:
    r"""Algorithm 2 [Q->Q]: Interference Routine

    Given unitaries $U, V$ such that $U\ket{0} = \sum_j \alpha_j \ket j$ and $U\ket{0} = \sum_j \beta_j \ket j$,
    returns a unitary that prepares

        $$\frac12 \ket0 \otimes \sum_j (\alpha_j + \beta_j) \ket j + \frac12 \ket1 \otimes \sum_j (\beta_j - \alpha_j) \ket j$$

    Args:
        U: encodes vector alpha
        V: encodes vector beta

    Returns:
        Above described output vector
    """
    raise NotImplementedError


def SignEstNFN(U, k, epsilon) -> bool:
    """Algorithm 3 [Q->C]: Sign estimation routine"""
    raise NotImplementedError


def RedCost(A_B, A_k, c, epsilon):
    """Algorithm 4 [Q->Q]: Determining the reduced cost of a column"""
    raise NotImplementedError


def CanEnter(A_B: Matrix, A_k: Vector, c: Vector, epsilon: float) -> bool:
    r"""Algorithm 5 [C->C]: Determine is a column is eligible to enter the basis

    Returns:
        1 if the nonbasic column $A_k$ has reduced cost $< \epsilon$;
        0 otherwise
    """
    raise NotImplementedError


def FindColumn(A: Matrix, B: Basis, c: Vector, epsilon: float) -> int:
    r"""Algorithm 6 [C->C]: Determine the next column to enter the basis

    Args:
        A: matrix s.t. $\norm{A_B} \le 1$
        B: basis
        c: cost vector s.t. $\norm{c_B} = 1$
        epsilon: precision

    Returns:
        index of column $k$ with $\bar{c}_k < \epsilon \norm{(A_B^{-1} A_k, c_k)} if one exists, with bounded probability.
    """
    raise NotImplementedError


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
    """Algorithm 8 [C->C]: Determine if the problem is unbounded from below"""
    raise NotImplementedError


def FindRow(A_B, A_k, b, delta, t) -> int:
    """Algorithm 9 [C->C]: Determine the basic variable (row) leaving the basis"""
    raise NotImplementedError


def IsFeasible(A_B, b, delta) -> bool:
    """Algorithm 10 [C->C]: Determine if a basic solution is feasible"""
    raise NotImplementedError
