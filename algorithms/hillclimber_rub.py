from dataclasses import asdict
from typing import Callable, Iterable, TypeVar
import numpy as np
import random
import logging
import pandas as pd

from algorithms.maxsat import MaxSatInstance
from bench.stats import QueryStats
import bench.qsearch as qsearch

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
        stats.quantum_expected_classical_queries += qsearch.estimate_classical_queries(N, T, K)
        stats.quantum_expected_quantum_queries += qsearch.estimate_quantum_queries(N, T, eps, K)

    # run the classical sampling-without-replacement algorithms
    random.shuffle(seq)
    for x in seq:
        if stats:
            stats.classical_actual_queries += 1
        if predicate(x):
            return x


def simple_hill_climber(
        inst: MaxSatInstance, *, eps=10 ** -5, stats: QueryStats = None
):
    # precompute some matrices (see 4.3.2 in Cade et al)
    n = inst.n
    ones = np.ones(n, dtype=int)
    flip_mat = np.outer(ones, ones) - 2 * np.eye(n, dtype=int)

    # start with a random assignment
    x = np.random.choice([-1, 1], n)
    w = inst.weight(x)

    while True:
        # compute all Hamming neighbors (row by row) and their weights
        neighbors = flip_mat * np.outer(ones, x)
        weights = inst.weight(neighbors)
        # better = np.flatnonzero(weights > w)

        # find improved directions
        stats.classical_control_method_calls += 1
        result = search(  # TODO this is called too often!
            zip(neighbors, weights), lambda it: it[1] > w, eps=eps, stats=stats
        )
        if not result:
            return x
        x, w = result


def run(k, r, n, runs, seed, dest):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    history = []
    for run in range(runs):
        # if verbose:
        logging.debug(f"k={k}, r={r}, n={n}, seed={seed}, #{run}")
        stats = QueryStats()
        inst = MaxSatInstance.random(k=k, n=n, m=r * n, seed=seed)
        simple_hill_climber(inst, stats=stats)
        stats = asdict(stats)
        stats["impl"] = "RUB"
        stats["n"] = n
        stats["k"] = k
        stats["r"] = r
        history.append(stats)

    history = pd.DataFrame(
        [list(row.values()) for row in history],
        columns=stats.keys(),
    )

    # print summary
    logging.info(history.groupby(["k", "r", "n"]).mean())

    # save
    if dest is not None:
        logging.info(f"saving to {dest}...")
        orig = pd.read_json(dest, orient="split") if dest.exists() else None
        history = pd.concat([orig, history])
        with dest.open("w") as f:
            f.write(history.to_json(orient="split"))

    return history
