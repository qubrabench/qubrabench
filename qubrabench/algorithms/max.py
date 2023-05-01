from typing import Iterable, TypeVar, Optional
import numpy as np

from qubrabench.bench.bounds import calculate_F
from qubrabench.bench.stats import QueryStats

E = TypeVar("E")


# TODO: add type for `key`
def max(
    iterable: Iterable[E],
    *,
    eps: float,
    default: Optional[E] = None,
    key=None,
    stats: Optional[QueryStats] = None,
) -> Optional[E]:
    if key is None:

        def key(x):
            return x

    iterable = list(iterable)

    max_val: Optional[E] = None
    for elem in iterable:
        if stats:
            stats.classical_actual_queries += 1
        if max_val is None or key(elem) > key(max_val):
            max_val = elem

    if max_val is None:
        if default is None:
            raise ValueError(
                "max() arg is an empty sequence, and no default value provided"
            )
        max_val = default

    max_val_occurrences = 0
    for elem in iterable:
        if key(elem) == max_val:
            max_val_occurrences += 1

    if stats:
        stats.quantum_expected_quantum_queries += estimate_quantum_queries(
            len(iterable), max_val_occurrences, stats.classical_actual_queries, eps
        )

    return max_val


def estimate_quantum_queries(N, T, cq, eps=10**-5):
    # assume cq corresponds to the number of classical comparisons corresponding to oracle O_f_i in paper
    sum_of_ts = 0
    for i in range(T, N):
        sum_of_ts += calculate_F(N, i) / (i + 1)
    return np.ceil(np.log(1 / eps) / np.log(3)) * 3 * (cq * sum_of_ts)
