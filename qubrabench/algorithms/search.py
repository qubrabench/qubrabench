""" This module provides a generic search interface that executes classically 
    and calculates the expected quantum query costs to a predicate function
"""

from typing import Callable, Iterable, Optional, TypeVar
import numpy as np
from qubrabench.stats import QueryStats


__all__ = ["search"]


E = TypeVar("E")


def search(
    iterable: Iterable[E],
    key: Callable[[E], bool],
    *,
    rng: np.random.Generator,
    error: Optional[float] = None,
    max_classical_queries: int = 130,
    stats: Optional[QueryStats] = None,
) -> Optional[E]:
    """Search a list in random order for an element satisfying the given predicate, while keeping track of query statistics.
    >>> search([1,2,3,4,5], lambda x: x % 2 == 0, rng=np.random.default_rng(1))
    2

    Args:
        iterable: iterable to be searched over
        key: function to test if an element satisfies the predicate
        rng: np.random.Generator instance as source of randomness
        error: upper bound on the failure probability of the quantum algorithm.
        max_classical_queries: maximum number of classical queries before entering the quantum part of the algorithm.
        stats: keeps track of statistics.

    Raises:
        ValueError: Raised when the error bound is not provided and statistics cannot be calculated.

    Returns:
        An element that satisfies the predicate, or None if no such argument can be found.
    """
    iterable = list(iterable)

    # collect stats
    if stats:
        if error is None:
            raise ValueError(
                "search() parameter 'error' not provided, cannot compute quantum query statistics"
            )
        N = len(iterable)
        T = sum(1 for x in iterable if key(x))
        stats.classical_expected_queries += (N + 1) / (T + 1)
        stats.quantum_expected_classical_queries += (
            cade_et_al_expected_classical_queries(N, T, max_classical_queries)
        )
        stats.quantum_expected_quantum_queries += cade_et_al_expected_quantum_queries(
            N, T, error, max_classical_queries
        )

    # run the classical sampling-without-replacement algorithms
    rng.shuffle(iterable)  # type: ignore
    for x in iterable:
        if stats:
            stats.classical_actual_queries += 1
        if key(x):
            return x

    return None


def cade_et_al_expected_quantum_queries(N: int, T: int, eps: float, K: int) -> float:
    """Upper bound on the number of *quantum* queries made by Cade et al's quantum search algorithm.

    Args:
        N: number of elements of search space
        T: number of solutions (marked elements)
        eps: upper bound on the failure probability
        K: maximum number of classical queries before entering the quantum part of the algorithm

    Returns:
        the upper bound on the number of quantum queries
    """
    if T == 0:
        return 9.2 * np.ceil(np.log(1 / eps) / np.log(3)) * np.sqrt(N)  # type: ignore

    F = cade_et_al_F(N, T)
    return (1 - (T / N)) ** K * F * (1 + (1 / (1 - (F / (9.2 * np.sqrt(N))))))  # type: ignore


def cade_et_al_expected_classical_queries(N: int, T: int, K: int) -> float:
    """Upper bound on the number of *classical* queries made by Cade et al's quantum search algorithm.

    Args:
        N: number of elements of search space
        T: number of solutions (marked elements)
        K: maximum number of classical queries before entering the quantum part of the algorithm

    Returns:
        float: the upper bound on the number of classical queries
    """
    if T == 0:
        return K

    return (N / T) * (1 - (1 - (T / N)) ** K)


def cade_et_al_F(N: int, T: int) -> float:
    """
    Return quantity F defined in Eq. (3) of Cade et al.
    """
    F = 2.0344
    if 1 <= T < (N / 4):
        F = (
            (9 / 4) * (N / (np.sqrt((N - T) * T)))
            + np.ceil(np.log((N / (2 * np.sqrt((N - T) * T)))) / np.log(6 / 5))
            - 3
        )
    return F
