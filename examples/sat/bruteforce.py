from dataclasses import asdict
import numpy as np
import scipy
from typing import Callable, Iterable, Optional, TypeVar

from sat import SatInstance
from qubrabench.bench.stats import QueryStats
from qubrabench.algorithms.search import search as search


def brute_force(
    inst: SatInstance,
    *,
    stats: Optional[QueryStats] = None,
):
    n = inst.n
    search_space = np.full((2**n, n), 1, dtype=int)
    for i in range(n):
        for start in range(0, 2**n, 2 ** (i + 1)):
            for j in range(2**i):
                search_space[start + j, i] = -1

    search(search_space, lambda x: inst.evaluate(x), stats=stats, eps=10**-5)

    T = 0
    for x in list(search_space):
        if inst.evaluate(x):
            T += 1
    return T


def run_specific_instance(inst: SatInstance, *, n_runs=5):
    stats = QueryStats()

    T = brute_force(inst, stats=stats)

    stats = asdict(stats)
    stats["n"] = inst.n
    stats["k"] = inst.k
    stats["m"] = inst.m

    stats["T"] = T

    return stats
