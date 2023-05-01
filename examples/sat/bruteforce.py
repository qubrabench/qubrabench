from dataclasses import asdict
import numpy as np
import pandas as pd
import scipy
from typing import Callable, Iterable, Optional, TypeVar

from sat import SatInstance
from qubrabench.bench.stats import QueryStats
from qubrabench.algorithms.search import search


def run_specific_instance(
    inst: SatInstance,
    *,
    n_runs: int = 5,
    eps: float = 10**-5,
    rng: np.random.Generator = np.random.default_rng(),
):
    n = inst.n
    search_space = np.full((2**n, n), 1, dtype=int)
    for i in range(n):
        for start in range(0, 2**n, 2 ** (i + 1)):
            for j in range(2**i):
                search_space[start + j, i] = -1

    T = 0
    for x in list(search_space):
        if inst.evaluate(x):
            T += 1

    history = []
    for run_ix in range(n_runs):
        stats = QueryStats()
        search(search_space, inst.evaluate, eps=eps, stats=stats, rng=rng)

        stats = asdict(stats)
        stats["n"] = inst.n
        stats["k"] = inst.k
        stats["m"] = inst.m
        stats["T"] = T
        history.append(stats)

    history = pd.DataFrame(
        [list(row.values()) for row in history],
        columns=stats.keys(),
    )

    return history
