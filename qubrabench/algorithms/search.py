""" This module provides a generic search interface that executes classically 
    and calculates the expected quantum query costs to a predicate function
"""

from abc import ABC, abstractmethod
from typing import Callable, Generic, Iterable, Optional, TypeVar

import numpy as np

from ..benchmark import (
    BenchmarkFrame,
    _already_benchmarked,
    _BenchmarkManager,
    track_queries,
)

__all__ = ["search", "search_by_sampling", "SamplingDomain"]

E = TypeVar("E")


def search(
    iterable: Iterable[E],
    key: Callable[[E], bool],
    *,
    rng: Optional[np.random.Generator] = None,
    error: Optional[float] = None,
    max_classical_queries: int = 130,
) -> Optional[E]:
    """Search a list for an element satisfying the given predicate.

    >>> search([1,2,3,4,5], lambda x: x % 2 == 0)
    2

    By default performs a linear scan on the iterable and returns the first element satisfying the `key`.
    If the optional random generator `rng` is provided, instead shuffles the input iterable before scanning for a solution.

    >>> search([1,2,3,4,5], lambda x: x % 2 == 0, rng=np.random.default_rng(42))
    4

    Args:
        iterable: iterable to be searched over
        key: function to test if an element satisfies the predicate
        rng: random generator - if provided shuffle the input before scanning
        error: upper bound on the failure probability of the quantum algorithm
        max_classical_queries: maximum number of classical queries before entering the quantum part of the algorithm

    Raises:
        ValueError: Raised when the error bound is not provided and statistics cannot be calculated.

    Returns:
        An element that satisfies the predicate, or None if no such argument can be found.
    """

    is_benchmarking = _BenchmarkManager.is_benchmarking()
    classical_is_random_search = rng is not None

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

        sub_frames_linear_scan: list[BenchmarkFrame] = []
        linear_scan_solution_found = False

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
                if not linear_scan_solution_found:
                    sub_frames_linear_scan.append(sub_frame_access)

            with track_queries() as sub_frame_eval:
                solution_found = False
                if key(x):
                    T += 1
                    solution_found = True
                sub_frames_eval.append(sub_frame_eval)

                if not linear_scan_solution_found:
                    sub_frames_linear_scan.append(sub_frame_eval)
                linear_scan_solution_found |= solution_found

        frame_access = _BenchmarkManager.combine_subroutine_frames(sub_frames_access)
        frame_eval = _BenchmarkManager.combine_subroutine_frames(sub_frames_eval)
        frame = _BenchmarkManager.combine_sequence_frames([frame_access, frame_eval])

        for obj_hash, stats in frame.stats.items():
            if classical_is_random_search:
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

        # classical expected queries for a linear scan
        if not classical_is_random_search:
            frame_linear_scan = _BenchmarkManager.combine_sequence_frames(
                sub_frames_linear_scan
            )
            for obj_hash, stats in frame_linear_scan.stats.items():
                _BenchmarkManager.current_frame()._add_classical_expected_queries(
                    obj_hash,
                    base_stats=stats,
                    queries=1,
                )

        iterable = iterable_copy

    with _already_benchmarked():
        # run the classical sampling-without-replacement algorithm
        if classical_is_random_search:
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
    """Base class for domains supporting search by random sampling.

    Define a space that is too large to fully enumerate, but can be efficiently sampled from.
    This is used by the `search_by_sampling` method to repeatedly sample an element till a solution one is found.
    """

    @abstractmethod
    def get_size(self) -> int:
        """Total size of the domain (number of elements)."""

    @abstractmethod
    def get_probability_of_sampling_solution(self, key) -> float:
        """A lower-bound on the probability that a single sample is a solution.

        Used to compute classical and quantum expected query counts.
        """

    @abstractmethod
    def get_random_sample(self, rng: np.random.Generator) -> E:
        """Produce a single random sample from the space."""


def search_by_sampling(
    domain: SamplingDomain[E],
    key: Callable[[E], bool],
    *,
    rng: np.random.Generator,
    error: float,
    max_classical_queries: int = 130,
) -> Optional[E]:
    r"""Search a domain by repeated sampling for an element satisfying the given predicate.

    This method assumes that the solutions are distributed uniformly at random through the search domain.
    If the probability of a single sample being a solution is $f$,
    and we want a failure probability (error) at most $\epsilon$,
    then we sample $3 * \log_{1 - f}(\epsilon)$ elements.

    The query counts are computed assuming that this is a representative sample of key evaluation cost.

    Caution:
        If there are anamolies in the space (solutions not evenly distributed, or a few elements have largely different key costs),
        then the stats computed here may not be accurate.

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

        # TODO: Is this a valid approximation?
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
