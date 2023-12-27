""" This module provides a generic search interface that executes classically 
    and calculates the expected quantum query costs to a predicate function
"""

from abc import ABC, abstractmethod
from typing import Callable, Iterable, Optional, TypeVar, Generic
import numpy as np

from ..benchmark import (
    _BenchmarkManager,
    BenchmarkFrame,
    track_queries,
    _already_benchmarked,
)

__all__ = ["search", "search_by_sampling_with_replacement", "SamplingDomain"]

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

    is_benchmarking = _BenchmarkManager.is_benchmarking()

    # collect stats
    if is_benchmarking:
        if error is None:
            raise ValueError(
                "search() parameter 'error' not provided, cannot compute quantum query statistics"
            )

        N = 0
        T = 0

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
                iterable_copy.append((x, sub_frame_access))
                sub_frames_access.append(sub_frame_access)

            with track_queries() as sub_frame_eval:
                if key(x):
                    T += 1
                sub_frames_eval.append(sub_frame_eval)

        frame_access = _BenchmarkManager.combine_subroutine_frames(sub_frames_access)
        frame_eval = _BenchmarkManager.combine_subroutine_frames(sub_frames_eval)
        frame = _BenchmarkManager.combine_sequence_frames([frame_access, frame_eval])

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

        iterable = iterable_copy

    with _already_benchmarked():
        # run the classical sampling-without-replacement algorithm
        try:
            rng.shuffle(iterable)
        except TypeError:
            pass

        for x in iterable:
            if is_benchmarking:
                elem, sub_frame = x
                # account for iterable access stats
                for obj_hash, stats in sub_frame.stats.items():
                    _BenchmarkManager.current_frame()._get_stats_from_hash(
                        obj_hash
                    ).classical_actual_queries += stats.classical_actual_queries
            else:
                elem = x
            if key(elem):
                return elem

    return None


class SamplingDomain(ABC, Generic[E]):
    """
    Class used for search routines requiring elaborate sampling with replacement. To search against a predicate, please
    refer to the function search_custom.
    If your use-case involves simpler search spaces, e.g. ones that can be stored as a list in memory, please use the
    function search for that.
    """

    @abstractmethod
    def get_size(self) -> int:
        pass

    @abstractmethod
    def get_probability_of_sampling_solution(self, key) -> float:
        pass

    @abstractmethod
    def get_random_sample(self, rng: np.random.Generator) -> E:
        pass


def search_by_sampling_with_replacement(
    domain: SamplingDomain[E],
    key: Callable[[E], bool],
    *,
    rng: np.random.Generator,
    error: float,
    max_classical_queries: int = 130,
) -> Optional[E]:
    """Search a domain by sampling with replacement, for an element satisfying the given predicate, while keeping track of query statistics.

    Args:
        domain: sampling domain to be searched over
        key: function to test if an element satisfies the predicate
        rng: np.random.Generator instance as source of randomness
        error: upper bound on the failure probability of the quantum algorithm.
        max_classical_queries: maximum number of classical queries before entering the quantum part of the algorithm.

    Returns:
        An element that satisfies the predicate, or None if no such argument can be found.
    """

    N = domain.get_size()
    f = domain.get_probability_of_sampling_solution(key)
    max_iterations = int(3 * np.log(error) / np.log(1.0 - f))

    is_benchmarking = _BenchmarkManager.is_benchmarking()

    # collect stats
    if is_benchmarking:
        sub_frames_access: list[BenchmarkFrame] = []
        sub_frames_eval: list[BenchmarkFrame] = []

        for _ in range(max_iterations):
            with track_queries() as sub_frame_access:
                x = domain.get_random_sample(rng)
                sub_frames_access.append(sub_frame_access)

            with track_queries() as sub_frame_eval:
                key(x)
                sub_frames_eval.append(sub_frame_eval)

        frame_access = _BenchmarkManager.combine_subroutine_frames(sub_frames_access)
        frame_eval = _BenchmarkManager.combine_subroutine_frames(sub_frames_eval)
        frame = _BenchmarkManager.combine_sequence_frames([frame_access, frame_eval])

        for obj_hash, stats in frame.stats.items():
            _BenchmarkManager.current_frame()._add_classical_expected_queries(
                obj_hash,
                base_stats=stats,
                queries=max_iterations,
            )

            _BenchmarkManager.current_frame()._add_quantum_expected_queries(
                obj_hash,
                base_stats=stats,
                queries_classical=cade_et_al_expected_classical_queries(
                    N, f * N, max_classical_queries
                ),
                queries_quantum=cade_et_al_expected_quantum_queries(
                    N, f * N, error, max_classical_queries
                ),
            )

    with _already_benchmarked():
        for _ in range(max_iterations):
            x = domain.get_random_sample(rng)
            if key(x):
                return x


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
