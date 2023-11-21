"""
This module provides a generic maximum finding interface that executes classically
and calculates the expected quantum query costs to a predicate function.
"""

from typing import Iterable, TypeVar, Optional, Callable, Any
import numpy as np

from .search import cade_et_al_F

__all__ = ["max"]

E = TypeVar("E")


def max(
    iterable: Iterable[E],
    *,
    default: Optional[E] = None,
    key: Optional[Callable[[E], Any]] = None,
    error: Optional[float] = None,
) -> E:
    """
    Find the largest element in a list, while keeping track of query statistics.

    Args:
        iterable: iterable to find the maximum in
        default: default value to return if iterable is empty.
        key: function that maps iterable elements to values that are comparable. By default, use the iterable elements.
        error: upper bound on the failure probability of the quantum algorithm.
        context: benchmarking context

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

    # if context:
    #     sub_frames = []

    N = 0  # number of elements in `iterable`

    max_elem: Optional[E] = None
    key_of_max_elem = None
    for elem in iterable:
        N += 1
        # if context:
        #     with context.recurse() as frame:
        #         key(elem)
        #     sub_frames.append(frame)

        key_of_elem = key(elem)
        if max_elem is None or key_of_elem > key_of_max_elem:
            max_elem = elem
            key_of_max_elem = key_of_elem

    if max_elem is None:
        if default is None:
            raise ValueError(
                "max() arg is an empty sequence, and no default value provided"
            )
        max_elem = default

    # if context:
    #     if error is None:
    #         raise ValueError(
    #             "max() parameter 'error' not provided, cannot compute quantum query statistics"
    #         )
    #
    #     for oracle, c_queries, q_queries in context.get_subroutine_costs(sub_frames):
    #         context.add_classical_expected_queries(oracle, c=np.sum(c_queries))
    #
    #         # see qubrabench.search for more info
    #         # [improved] select with uniform access
    #         # TODO: possibly missing polylog and constant factors
    #         quantum_query_weight = np.max(q_queries)
    #
    #         context.add_quantum_expected_queries(
    #             oracle,
    #             q=cade_et_al_expected_quantum_queries(N, error) * quantum_query_weight,
    #         )

    return max_elem


def cade_et_al_expected_quantum_queries(N: int, error: float) -> float:
    """Upper bound on the number of quantum queries made by Cade et al's quantum max algorithm.
    https://doi.org/10.48550/arXiv.2203.04975, Corollary 1

    Args:
        N: number of elements of search space
        error: upper bound on the failure probability of the quantum algorithm.


    Returns:
        the upper bound on the number of quantum queries
    """
    sum_of_ts: float = sum([cade_et_al_F(N, t) / (t + 1) for t in range(1, N)])
    return np.ceil(np.log(1 / error) / np.log(3)) * 3 * sum_of_ts
