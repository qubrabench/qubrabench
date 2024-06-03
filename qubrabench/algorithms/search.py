""" This module provides a generic search interface that executes classically 
    and calculates the expected quantum query costs to a predicate function
"""

from abc import ABC, abstractmethod
from typing import Callable, Generic, Iterable, Iterator, Optional, TypeVar

import numpy as np

from ..benchmark import BenchmarkFrame, _BenchmarkManager, track_queries

__all__ = ["search", "search_by_sampling", "SamplingDomain"]

E = TypeVar("E")


def search(
    iterable: Iterable[E],
    key: Callable[[E], bool],
    *,
    max_fail_probability: float,
    rng: Optional[np.random.Generator] = None,
    max_classical_queries: int = 130,
) -> Optional[E]:
    """Search a list for an element satisfying the given predicate.

    >>> search([1,2,3,4,5], lambda x: x % 2 == 0, max_fail_probability=1/3)
    2

    By default performs a linear scan on the iterable and returns the first element satisfying the `key`.
    If the optional random generator `rng` is provided, instead shuffles the input iterable before scanning for a solution.

    >>> search([1,2,3,4,5], lambda x: x % 2 == 0, rng=np.random.default_rng(42), max_fail_probability=1/3)
    4

    Args:
        iterable: iterable to be searched over
        key: function to test if an element satisfies the predicate
        max_fail_probability: upper bound on the failure probability of the quantum algorithm
        rng: random generator - if provided shuffle the input before scanning
        max_classical_queries: maximum number of classical queries before entering the quantum part of the algorithm

    Returns:
        An element that satisfies the predicate, or None if no such argument can be found.
    """

    is_benchmarking = _BenchmarkManager.is_benchmarking()
    classical_is_random_search = rng is not None

    # collect stats
    if is_benchmarking:
        N = 0
        T = 0

        sub_frames: list[BenchmarkFrame] = []
        is_solution: list[bool] = []

        it: Iterator[E] = iter(iterable)
        iterable_copy: list[E] = []
        while True:
            with track_queries() as sub_frame:
                try:
                    x = next(it)
                except StopIteration:
                    break
                N += 1

                solution_found = False
                if key(x):
                    T += 1
                    solution_found = True

                iterable_copy.append(x)
                sub_frames.append(sub_frame)
                is_solution.append(solution_found)

        frame = _BenchmarkManager.combine_subroutine_frames(sub_frames)
        current_frame = _BenchmarkManager.current_frame()

        # compute expected cost for the classical random sampling algorithm
        if classical_is_random_search:
            for sub_frame, is_sol in zip(sub_frames, is_solution):
                for obj, stats in sub_frame.stats.items():
                    current_frame.stats[
                        obj
                    ].classical_expected_queries += stats.classical_expected_queries / (
                        T if is_sol else T + 1
                    )

        # compute costs for the quantum algorithm
        quantum_classical_queries = cade_et_al_expected_classical_queries(
            N, T, max_classical_queries
        )
        quantum_quantum_queries = cade_et_al_expected_quantum_queries(
            N, T, max_fail_probability, max_classical_queries
        )
        quantum_worst_case_queries = cade_et_al_worst_case_quantum_queries(
            N, max_fail_probability
        )
        c_q = 2  # number of queries to the standard oracle (that uses workspace registers) to build the phase oracle

        for obj, stats in frame.stats.items():
            current_frame._add_queries_for_quantum(
                obj,
                base_stats=stats,
                expected_classical_queries=quantum_classical_queries,
                expected_quantum_queries=c_q * quantum_quantum_queries,
                worst_case_classical_queries=0,
                worst_case_quantum_queries=c_q * quantum_worst_case_queries,
            )

        indices = np.arange(N)
        if classical_is_random_search:
            rng.shuffle(indices)

        for i in indices:
            for obj, stats in sub_frames[i].stats.items():
                if not classical_is_random_search:
                    current_frame.stats[
                        obj
                    ].classical_expected_queries += stats.classical_expected_queries

                current_frame.stats[
                    obj
                ].classical_actual_queries += stats.classical_actual_queries

            if is_solution[i]:
                return iterable_copy[i]
        return None
    else:
        # run the classical sampling-without-replacement algorithm
        if classical_is_random_search:
            try:
                rng.shuffle(iterable)
            except TypeError:
                pass

        for x in iterable:
            if key(x):
                return x


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
    max_fail_probability: float,
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
        max_fail_probability: upper bound on the failure probability of the quantum algorithm.
        max_classical_queries: maximum number of classical queries before entering the quantum part of the algorithm.

    Returns:
        An element that satisfies the predicate, or None if no such argument can be found.
    """
    N = domain.get_size()
    f = domain.get_probability_of_sampling_solution(key)
    max_iterations = int(3 * np.log(max_fail_probability) / np.log(1.0 - f))

    if _BenchmarkManager.is_benchmarking():
        # collect stats
        sub_frames: list[BenchmarkFrame] = []
        sub_frames_till_solution: list[BenchmarkFrame] = []
        solution = None

        T = 0

        # TODO: Is this a valid approximation?
        for _ in range(max_iterations):
            with track_queries() as sub_frame:
                if solution is None:
                    sub_frames_till_solution.append(sub_frame)

                x = domain.get_random_sample(rng)
                if key(x):
                    if solution is None:
                        solution = x
                    T += 1

                sub_frames.append(sub_frame)

        frame = _BenchmarkManager.combine_subroutine_frames(sub_frames)
        current_frame = _BenchmarkManager.current_frame()

        classical_queries = (max_iterations + 1) / (T + 1)
        quantum_classical_queries = cade_et_al_expected_classical_queries(
            N, f * N, max_classical_queries
        )
        quantum_quantum_queries = cade_et_al_expected_quantum_queries(
            N, f * N, max_fail_probability, max_classical_queries
        )
        c_q = 2  # number of queries to the standard oracle (that uses workspace registers) to build the phase oracle

        for obj, stats in frame.stats.items():
            current_frame.stats[obj].classical_expected_queries += (
                classical_queries * stats.classical_expected_queries
            )

            current_frame._add_queries_for_quantum(
                obj,
                base_stats=stats,
                expected_classical_queries=quantum_classical_queries,
                expected_quantum_queries=c_q * quantum_quantum_queries,
            )

        for sub_frame in sub_frames_till_solution:
            for obj, stats in sub_frame.stats.items():
                current_frame.stats[
                    obj
                ].classical_actual_queries += stats.classical_actual_queries

        return solution
    else:
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


def cade_et_al_worst_case_quantum_queries(N: int, eps: float) -> float:
    """Lemma 8: worst case cost given by the QSearch_Zalka variant

    Args:
        N: size of the search space
        eps: upper bound on failure probability

    Returns:
        Worst case query cost of quantum search
    """
    eps_term = np.ceil(np.log(1 / eps) / (2 * np.log(4 / 3)))
    return 5 * eps_term + np.pi * np.sqrt(N * eps_term)
