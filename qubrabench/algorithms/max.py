"""
This module provides a generic maximum finding interface that executes classically
and calculates the expected quantum query costs to a predicate function.
"""

from typing import Any, Callable, Iterable, Optional, TypeVar

import numpy as np

from .._internals import OptionalParameter, _absent
from ..benchmark import (
    BenchmarkFrame,
    _already_benchmarked,
    _BenchmarkManager,
    _qubrabench_method,
    track_queries,
)
from .search import cade_et_al_F

__all__ = ["max"]

E = TypeVar("E")


@_qubrabench_method
def max(
    iterable: Iterable[E],
    *,
    max_fail_probability: float,
    default: OptionalParameter[E] = _absent,
    key: Optional[Callable[[E], Any]] = None,
) -> E:
    """Find the largest element in a list.

    Args:
        iterable: iterable to find the maximum in
        default: default value to return if iterable is empty.
        key: function that maps iterable elements to values that are comparable. By default, use the iterable elements.
        max_fail_probability: upper bound on the failure probability of the quantum algorithm.

    Raises:
        ValueError: Raised when iterable is an empty sequence and no default is provided.

    Returns:
        the desired maximum element
    """
    if key is None:

        def default_key(x: E) -> Any:
            return x

        key = default_key

    is_benchmarking = _BenchmarkManager.is_benchmarking()

    # collect stats
    if is_benchmarking:
        N = 0

        sub_frames_access: list[BenchmarkFrame] = []
        sub_frames_eval: list[BenchmarkFrame] = []

        it = iter(iterable)
        iterable_copy = []
        while True:
            with track_queries() as sub_frame_access:
                try:
                    x = next(it)
                except StopIteration:
                    break
                N += 1
                iterable_copy.append(x)
                sub_frames_access.append(sub_frame_access)

            with track_queries() as sub_frame_eval:
                key(x)
                sub_frames_eval.append(sub_frame_eval)

        frame_access = _BenchmarkManager.combine_subroutine_frames(sub_frames_access)
        frame_eval = _BenchmarkManager.combine_subroutine_frames(sub_frames_eval)
        frame = _BenchmarkManager.combine_sequence_frames([frame_access, frame_eval])

        for obj, stats in frame.stats.items():
            _BenchmarkManager.current_frame()._add_classical_expected_queries(
                obj, queries=N, base_stats=stats
            )

            _BenchmarkManager.current_frame()._add_quantum_expected_queries(
                obj,
                queries_classical=0,
                queries_quantum=cade_et_al_expected_quantum_queries(
                    N, max_fail_probability
                ),
                base_stats=stats,
            )

        iterable = iterable_copy

    max_elem: OptionalParameter[E] = None
    with _already_benchmarked():
        if is_benchmarking:
            for sub_frame in sub_frames_access:
                for obj, stats in sub_frame.stats.items():
                    _BenchmarkManager.current_frame()._get_or_init_stats(
                        obj
                    ).classical_actual_queries += stats.classical_actual_queries

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
