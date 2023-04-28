from dataclasses import asdict
import numpy as np
import random
import logging
import pandas as pd

from maxsat import MaxSatInstance
from qubrabench.bench.stats import QueryStats
from qubrabench.algorithms.search import search


# MW: should not return QueryStats
def simple_hill_climber(
    inst: MaxSatInstance, *, eps=10**-5, stats: QueryStats = None
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

        # TODO: remove this once KIT and RUB in harmony
        neighbors = np.array(sorted(map(tuple, neighbors)))

        # find improved directions
        stats.classical_control_method_calls += 1

        # TODO this is called too often!
        # MW: Cade et al propose picking eps = eps_overall / n as a heuristic (see 4.3.1 in Cade et al), we should do the same

        # OPTION 1: "realistic" implementation
        # result = search(neighbors, lambda x: inst.weight(x) > w, eps=eps, stats=stats)
        # if result is None:
        #     return x
        # x, w = result, inst.weight(result)

        # OPTION 2: faster implementation (for our instance sizes)
        weights = inst.weight(neighbors)
        result = search(
            zip(neighbors, weights), lambda it: it[1] > w, eps=eps, stats=stats
        )
        if result is None:
            return x
        x, w = result


def run(k, r, n, *, runs, seed=None, random_weights=None, dest=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    history = []
    for run in range(runs):
        # if verbose:
        logging.debug(f"k={k}, r={r}, n={n}, seed={seed}, #{run}")
        stats = QueryStats()
        inst = MaxSatInstance.random(
            k=k, n=n, m=r * n, seed=seed, random_weights=random_weights
        )
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
    logging.info(history.groupby(["k", "r", "n"]).mean(numeric_only=True))

    # save
    if dest is not None:
        logging.info(f"saving to {dest}...")
        orig = pd.read_json(dest, orient="split") if dest.exists() else None
        history = pd.concat([orig, history])
        with dest.open("w") as f:
            f.write(history.to_json(orient="split"))

    return history
