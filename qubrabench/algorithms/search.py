""" This module provides a generic search interface that executes classically 
    and calculates the expected quantum query costs to a predicate function
"""

from typing import Callable, Iterable, Optional, TypeVar
import numpy as np
from ..benchmark import (
    _BenchmarkManager,
    BenchmarkFrame,
    track_queries,
    _already_benchmarked,
)

__all__ = ["search"]

E = TypeVar("E")


def search(
    iterable: Iterable[E],
    key: Callable[[E], bool],
    *,
    rng: np.random.Generator,
    error: Optional[float] = None,
    max_classical_queries: int = 130,
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

    Raises:
        ValueError: Raised when the error bound is not provided and statistics cannot be calculated.

    Returns:
        An element that satisfies the predicate, or None if no such argument can be found.
    """
    # TODO maybe use Sequence instead of Iterable:
    #      as Iterable might get consumed during the first iteration,
    #      making the benchmark implementation a ugly
    #      Also: the iteration itself can make queries (e.g. when using QList), so this must be captured very carefully.

    # collect stats
    if _BenchmarkManager.is_tracking():
        if error is None:
            raise ValueError(
                "search() parameter 'error' not provided, cannot compute quantum query statistics"
            )

        N = 0
        T = 0

        sub_frames_access: list[BenchmarkFrame] = []
        sub_frames_eval_key: list[BenchmarkFrame] = []
        it = iter(iterable)
        iterable_copy = []
        while True:
            with track_queries() as sub_frame_access:
                try:
                    x = next(it)
                except StopIteration:
                    break
                iterable_copy.append(x)
                N += 1
                sub_frames_access.append(sub_frame_access)

            with track_queries() as sub_frame_eval_key:
                if key(x):
                    T += 1
                sub_frames_eval_key.append(sub_frame_eval_key)

        frame_access = _BenchmarkManager.combine_subroutine_frames(sub_frames_access)
        frame_eval_key = _BenchmarkManager.combine_subroutine_frames(
            sub_frames_eval_key
        )
        frame = _BenchmarkManager.combine_sequence_frames(
            [frame_access, frame_eval_key]
        )

        for obj_hash, stats in frame.stats.items():
            _BenchmarkManager.current_frame()._add_classical_expected_queries(
                obj_hash,
                base_stats=stats,
                queries=(N + 1) / (T + 1),
            )

            _BenchmarkManager.current_frame()._add_quantum_expected_queries(
                obj_hash,
                base_stats=stats,
                queries_classical=cade_et_al_expected_classical_queries(
                    N, T, max_classical_queries
                ),
                queries_quantum=cade_et_al_expected_quantum_queries(
                    N, T, error, max_classical_queries
                ),
            )

        # iterable already consumed, account for true queries during iteration into the parent frame
        for obj_hash, stats in _BenchmarkManager.combine_sequence_frames(
            sub_frames_access
        ).stats.items():
            _BenchmarkManager.current_frame()._get_stats_from_hash(
                obj_hash
            ).classical_actual_queries += stats.classical_actual_queries

        iterable = iterable_copy
    else:
        iterable = list(iterable)

    with _already_benchmarked():
        # run the classical sampling-without-replacement algorithm
        rng.shuffle(iterable)  # type: ignore
        for x in iterable:
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
    F: float = 2.0344
    if 1 <= T < (N / 4):
        term: float = N / (2 * np.sqrt((N - T) * T))
        F = 9 / 2 * term + np.ceil(np.log(term) / np.log(6 / 5)) - 3
    return F
