"""
This module provides a generic maximum finding interface that executes classically
and calculates the expected quantum query costs to a predicate function.
"""

from typing import Iterable, TypeVar, Optional, Callable, Any
import numpy as np

from .search import cade_et_al_F
from ..stats import QueryStats

__all__ = ["max"]

E = TypeVar("E")


def max(
    iterable: Iterable[E],
    *,
    error: Optional[float] = None,
    default: Optional[E] = None,
    key: Optional[Callable[[E], Any]] = None,
    stats: Optional[QueryStats] = None,
) -> E:
    """
    Find the largest element in a list, while keeping track of query statistics.

    Args:
        iterable: iterable to find the maximum in
        error: upper bound on the failure probability of the quantum algorithm.
        default: default value to return if iterable is empty.
        key: function that maps iterable elements to values that are comparable. By default, use the iterable elements.
        stats: object that keeps track of statistics.

    Raises:
        ValueError: Raised when the failure rate `error` is not provided and statistics cannot be calculated.
        ValueError: Raised when iterable is an empty sequence and no default is provided.

    Returns:
        the desired maximum element
    """
    if key is None:

        def default_key(x: E) -> Any:
            return x

        key = default_key

    iterable = list(iterable)

    max_elem: Optional[E] = None
    for elem in iterable:
        if stats:
            stats.classical_actual_queries += 1
        if max_elem is None or key(elem) > key(max_elem):
            max_elem = elem

    if max_elem is None:
        if default is None:
            raise ValueError(
                "max() arg is an empty sequence, and no default value provided"
            )
        max_elem = default

    max_elem_occurrences = 0
    for elem in iterable:
        if key(elem) == max_elem:
            max_elem_occurrences += 1

    if stats:
        if error is None:
            raise ValueError(
                "max() parameter 'error' not provided, cannot compute quantum query statistics"
            )
        stats.quantum_expected_quantum_queries += cade_et_al_expected_quantum_queries(
            len(iterable), max_elem_occurrences, 1, error
        )

    return max_elem


def cade_et_al_expected_quantum_queries(N: int, T: int, cq: int, error: float) -> float:
    """Upper bound on the number of quantum queries made by Cade et al's quantum max algorithm.

    Args:
        N: number of elements of search space
        T: number of solutions (marked elements)
        cq: the quantum cost factor
        error: upper bound on the failure probability of the quantum algorithm.


    Returns:
        the upper bound on the number of quantum queries
    """
    # assume cq corresponds to the number of classical comparisons corresponding to oracle O_{f_i} in paper
    sum_of_ts: float = 0
    for i in range(T, N):
        sum_of_ts += cade_et_al_F(N, i) / (i + 1)
    return np.ceil(np.log(1 / error) / np.log(3)) * 3 * (cq * sum_of_ts)  # type: ignore
