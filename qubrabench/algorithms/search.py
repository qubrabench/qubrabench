from typing import Callable, Iterable, TypeVar
from qubrabench.bench.stats import QueryStats
from qubrabench.bench import qsearch

import random


T = TypeVar("T")


def search(
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
        stats.quantum_expected_classical_queries += qsearch.estimate_classical_queries(
            N, T, K
        )
        stats.quantum_expected_quantum_queries += qsearch.estimate_quantum_queries(
            N, T, eps, K
        )

    # run the classical sampling-without-replacement algorithms
    random.shuffle(seq)
    for x in seq:
        if stats:
            stats.classical_actual_queries += 1
        if predicate(x):
            return x
