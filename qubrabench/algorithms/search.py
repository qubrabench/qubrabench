from typing import Callable, Iterable, TypeVar
from qubrabench.bench.stats import QueryStats

import numpy as np

import random


T = TypeVar("T")


def qsearch(
    seq: Iterable[T],
    predicate: Callable[[T], bool],
    *,
    eps,
    K=130,
    stats: QueryStats = None,
):
    """
    Search a list by random sampling (and keep track of classical and quantum stats).

    TODO: Think about how to interpret eps for the classical algorithm.
    """
    seq = list(seq)

    # collect stats
    if stats:
        N = len(seq)
        T = sum(1 for x in seq if predicate(x))
        stats.classical_expected_queries += (N + 1) / (T + 1)
        stats.quantum_expected_classical_queries += estimate_classical_queries(N, T, K)
        stats.quantum_expected_quantum_queries += estimate_quantum_queries(N, T, eps, K)

    # run the classical sampling-without-replacement algorithms
    # TODO: should provide an rng for this shuffle
    random.shuffle(seq)
    for x in seq:
        if stats:
            stats.classical_actual_queries += 1
        if predicate(x):
            return x


# MW: epsilon should not have a default value
def estimate_quantum_queries(N, T, epsilon=10**-5, K=130):
    if T == 0:
        # approximate epsilon if it isn't provided
        return 9.2 * np.ceil(np.log(1 / epsilon) / np.log(3)) * np.sqrt(N)

    F = 2.0344
    if 1 <= T < (N / 4):
        F = (
            (9 / 4) * (N / (np.sqrt((N - T) * T)))
            + np.ceil(np.log((N / (2 * np.sqrt((N - T) * T)))) / np.log(6 / 5))
            - 3
        )

    return pow((1 - (T / N)), K) * F * (1 + (1 / (1 - (F / (9.2 * np.sqrt(N))))))


def estimate_classical_queries(N, T, K=130):
    if T == 0:
        return K
    else:
        return (N / T) * (1 - pow((1 - (T / N)), K))
