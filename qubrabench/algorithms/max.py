from typing import Iterable, TypeVar, Optional
from qubrabench.bench.stats import QueryStats

T = TypeVar("T")


def max(
    iterable: Iterable[T],
    *,
    default: Optional[T] = None,
    key=None,
    eps=10**-5,
    stats: QueryStats = None,
):
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
    return max_val
