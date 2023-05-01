from dataclasses import asdict
from typing import Optional
import numpy as np
import logging
import pandas as pd

from sat import WeightedSatInstance
from qubrabench.bench.stats import QueryStats
from qubrabench.algorithms.search import search


# MW: should not return QueryStats
def simple_hill_climber(
    inst: WeightedSatInstance,
    *,
    eps: float = 10**-5,
    stats: Optional[QueryStats] = None,
    rng: np.random.Generator = np.random.default_rng(),
):
    # precompute some matrices (see 4.3.2 in Cade et al)
    n = inst.n
    ones = np.ones(n, dtype=int)
    flip_mat = np.outer(ones, ones) - 2 * np.eye(n, dtype=int)

    # start with a random assignment
    x = rng.choice([-1, 1], n)
    w = inst.weight(x)

    while True:
        # compute all Hamming neighbors (row by row) and their weights
        neighbors = flip_mat * np.outer(ones, x)

        # find improved directions
        if stats:
            stats.classical_control_method_calls += 1

        # MW: Cade et al propose picking eps = eps_overall / n as a heuristic (see 4.3.1 in Cade et al), we should do the same

        # OPTION 1: "realistic" implementation (what should be done in case we cared about really large instances)
        # result = search(neighbors, lambda x: inst.weight(x) > w, eps=eps, stats=stats)
        # if result is None:
        #     return x
        # x, w = result, inst.weight(result)

        # OPTION 2: faster implementation (for our instance sizes)
        weights = inst.weight(neighbors)
        result = search(
            zip(neighbors, weights), lambda it: it[1] > w, eps=eps, stats=stats, rng=rng
        )
        if result is None:
            return x
        x, w = result


def run(
    k,
    r,
    n,
    *,
    n_runs,
    rng: np.random.Generator = np.random.default_rng(),
    random_weights=None,
    dest=None,
):
    history = []
    for run_ix in range(n_runs):
        # if verbose:
        logging.debug(f"k={k}, r={r}, n={n}, #{run_ix}")
        stats = QueryStats()
        inst = WeightedSatInstance.random(
            k=k, n=n, m=r * n, rng=rng, random_weights=random_weights
        )
        simple_hill_climber(inst, stats=stats, rng=rng)
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
