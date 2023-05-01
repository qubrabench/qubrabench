import random
from typing import Callable, Iterable, Optional, TypeVar
import numpy as np
from qubrabench.bench.stats import QueryStats
from qubrabench.bench.bounds import calculate_F

__all__ = ["search"]


E = TypeVar("E")


def search(
    iterable: Iterable[E],
    predicate: Callable[[E], bool],
    *,
    eps: float,
    K: int = 130,
    stats: Optional[QueryStats] = None,
) -> Optional[E]:
    """
    Search a list by linear search, while keeping track of query statistics.

    This function random sampling (and keep track of classical and quantum stats).

    Arguments:
    :param int N: number of elements of search space
    :param int T: number of solutions / marked elements
    :param float eps: upper bound on the failure probability of the quantum algorithm
    :param int K: maximum number of classical queries before entering the quantum part of the algorithm
    :param QueryStats stats: object that keeps track of statistics
    """
    iterable = list(iterable)

    # collect stats
    if stats:
        N = len(iterable)
        T = sum(1 for x in iterable if predicate(x))
        stats.classical_expected_queries += (N + 1) / (T + 1)
        stats.quantum_expected_classical_queries += (
            cade_et_al_expected_classical_queries(N, T, K)
        )
        stats.quantum_expected_quantum_queries += cade_et_al_expected_quantum_queries(
            N, T, eps, K
        )

    # run the classical sampling-without-replacement algorithms
    # TODO: should provide an rng for this shuffle
    random.shuffle(iterable)
    for x in iterable:
        if stats:
            stats.classical_actual_queries += 1
        if predicate(x):
            return x

    return None


def cade_et_al_expected_quantum_queries(N: int, T: int, eps, K: int):
    """
    Upper bound on the number of *quantum* queries made by Cade et al's quantum search algorithm.

    :param int N: number of elements of search space
    :param int T: number of solutions / marked elements
    :param float eps: upper bound on the failure probability
    :param int K: maximum number of classical queries before entering the quantum part of the algorithm
    """
    if T == 0:
        return 9.2 * np.ceil(np.log(1 / eps) / np.log(3)) * np.sqrt(N)

    F = calculate_F(N, T)

    return pow((1 - (T / N)), K) * F * (1 + (1 / (1 - (F / (9.2 * np.sqrt(N))))))


def cade_et_al_expected_classical_queries(N: int, T: int, K: int):
    """
    Upper bound on the number of *classical* queries made by Cade et al's quantum
    search algorithm.

    :param int N: number of elements of search space
    :param int T: number of solutions / marked elements
    :param int K: maximum number of classical queries before entering the quantum
    part of the algorithm
    """
    if T == 0:
        return K

    return (N / T) * (1 - (1 - (T / N)) ** K)
