from typing import Iterable, TypeVar, Optional

from qubrabench.bench.bounds import calculate_F
from qubrabench.bench.stats import QueryStats

import numpy as np

T = TypeVar("T")


def max(
    iterable: Iterable[T],
    *,
    default: Optional[T] = None,
    key=None,
    eps=10**-5,
    stats: QueryStats = None,
):
    iterable = list(iterable)
    iterator = iter(iterable)
    try:
        max_val = next(iterator)
    except StopIteration:
        if default is None:
            raise ValueError("max() arg is an empty sequence")
        return default
    if key is None:
        key = lambda x: x
    for elem in iterator:
        stats.classical_actual_queries += 1
        if key(elem) > key(max_val):
            max_val = elem
    max_val_occurrences = 0
    for elem in iterator:
        if key(elem) == max_val:
            max_val_occurrences += 1
    stats.quantum_expected_quantum_queries = estimate_quantum_queries(
        len(iterable), max_val_occurrences, stats.classical_actual_queries
    )


def estimate_quantum_queries(N, T, cq, epsilon=10**-5):
    # i assume cq corresponds to the number of classical comparisons corresponding to oracle O_f_i in paper
    sum_of_ts = 0
    for i in range(T, N):
        sum_of_ts += calculate_F(N, i) / (i + 1)
    return np.ceil(np.log(1 / epsilon) / np.log(3)) * 3 * (cq * sum_of_ts)
