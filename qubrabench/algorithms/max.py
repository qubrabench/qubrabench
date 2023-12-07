"""
This module provides a generic maximum finding interface that executes classically
and calculates the expected quantum query costs to a predicate function.
"""

from typing import Iterable, TypeVar, Optional, Callable, Any
import numpy as np

from .search import cade_et_al_F
from ..benchmark import (
    _BenchmarkManager,
    BenchmarkFrame,
    track_queries,
    _already_benchmarked,
)
from .._internals import _absent, OptionalParameter

__all__ = ["max"]

E = TypeVar("E")


def max(
    iterable: Iterable[E],
    *,
    default: OptionalParameter[E] = _absent,
    key: Optional[Callable[[E], Any]] = None,
    error: Optional[float] = None,
) -> E:
    """Find the largest element in a list.

    Args:
        iterable: iterable to find the maximum in
        default: default value to return if iterable is empty.
        key: function that maps iterable elements to values that are comparable. By default, use the iterable elements.
        error: upper bound on the failure probability of the quantum algorithm.

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

    # collect stats
    if _BenchmarkManager.is_tracking():
        if error is None:
            raise ValueError(
                "max() parameter 'error' not provided, cannot compute quantum query statistics"
            )

        N = 0

        sub_frames: list[BenchmarkFrame] = []
        it = iter(iterable)
        iterable_copy = []
        while True:
            with track_queries() as sub_frame:
                try:
                    x = next(it)
                except StopIteration:
                    break
                iterable_copy.append(x)
                N += 1
                key(x)
                sub_frames.append(sub_frame)
        iterable = iterable_copy

        frame = _BenchmarkManager.combine_subroutine_frames(sub_frames)

        for obj_hash, stats in frame.stats.items():
            _BenchmarkManager.current_frame()._add_classical_expected_queries(
                obj_hash, queries=N, base_stats=stats
            )

            _BenchmarkManager.current_frame()._add_quantum_expected_queries(
                obj_hash,
                queries_classical=0,
                queries_quantum=cade_et_al_expected_quantum_queries(N, error),
                base_stats=stats,
            )

    max_elem: OptionalParameter[E] = None
    with _already_benchmarked():
        key_of_max_elem = None
        for elem in iterable:
            key_of_elem = key(elem)
            if max_elem is None or key_of_elem > key_of_max_elem:
                max_elem = elem
                key_of_max_elem = key_of_elem

        if max_elem is None:
            if default is _absent:
                raise ValueError(
                    "max() arg is an empty sequence, and no default value provided"
                )
            max_elem = default

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
