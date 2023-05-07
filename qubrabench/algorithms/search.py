from typing import Callable, Iterable, Optional, TypeVar
import numpy as np
from ..stats import QueryStats


__all__ = ["search"]


E = TypeVar("E")


def search(
    iterable: Iterable[E],
    predicate: Callable[[E], bool],
    *,
    rng: Optional[np.random.Generator] = None,
    eps: Optional[float] = None,
    K: int = 130,
    stats: Optional[QueryStats] = None,
) -> Optional[E]:
    """
    Search a list in random order for an element satisfying the given predicate, while keeping track of query statistics.

    Arguments:
    :param iterable: iterable to be searched over
    :param predicate: function to test if an element is marked
    :param rng: np.random.Generator object
    :param float eps: upper bound on the failure probability of the quantum algorithm
    :param int K: maximum number of classical queries before entering the quantum part of the algorithm
    :param QueryStats stats: object that keeps track of statistics
    """
    if rng is None:
        rng = np.random.default_rng()

    iterable = list(iterable)

    # collect stats
    if stats:
        if eps is None:
            raise ValueError("search() eps not provided, cannot compute stats")
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
    rng.shuffle(iterable)  # type: ignore
    for x in iterable:
        if stats:
            stats.classical_actual_queries += 1
        if predicate(x):
            return x

    return None


def cade_et_al_expected_quantum_queries(N: int, T: int, eps: float, K: int) -> float:
    """
    Upper bound on the number of *quantum* queries made by Cade et al's quantum search algorithm.

    :param int N: number of elements of search space
    :param int T: number of solutions / marked elements
    :param float eps: upper bound on the failure probability
    :param int K: maximum number of classical queries before entering the quantum part of the algorithm
    """
    if T == 0:
        return 9.2 * np.ceil(np.log(1 / eps) / np.log(3)) * np.sqrt(N)  # type: ignore

    F = cade_et_al_F(N, T)
    return (1 - (T / N)) ** K * F * (1 + (1 / (1 - (F / (9.2 * np.sqrt(N))))))  # type: ignore


def cade_et_al_expected_classical_queries(N: int, T: int, K: int) -> float:
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


def cade_et_al_F(N: int, T: int) -> float:
    """
    Return quantity F defined in Eq. (3) of Cade et al.
    """
    F = 2.0344
    if 1 <= T < (N / 4):
        F = (
            (9 / 4) * (N / (np.sqrt((N - T) * T)))
            + np.ceil(np.log((N / (2 * np.sqrt((N - T) * T)))) / np.log(6 / 5))
            - 3
        )
    return F
