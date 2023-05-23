from dataclasses import asdict
from typing import Optional, Tuple, Callable
import logging
import numpy as np
import numpy.typing as npt
import pandas as pd

from sat import WeightedSatInstance, Assignment, W
from qubrabench.stats import QueryStats
from qubrabench.algorithms.search import search
from qubrabench.algorithms.max import max


# TODO: should not return QueryStats
def hill_climber(
    inst: WeightedSatInstance,
    *,
    rng: np.random.Generator,
    eps: Optional[float] = None,
    stats: Optional[QueryStats] = None,
    steep: bool = False
) -> Optional[Assignment]:
    if rng is None:
        rng = np.random.default_rng()

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

        # MW: Cade et al. propose picking eps = eps_overall / n as a heuristic (see 4.3.1 in Cade et al.), we should do the same

        # OPTION 1: "realistic" implementation (what should be done in case we cared about really large instances)
        # result = search(neighbors, lambda x: inst.weight(x) > w, eps=eps, stats=stats)
        # if result is None:
        #     return x
        # x, w = result, inst.weight(result)

        # OPTION 2: faster implementation (for our instance sizes)
        weights = inst.weight(neighbors)

        def pred(it: Tuple[Assignment, np.float_]) -> bool:
            return bool(it[1] > w)

        if steep:
            result = max(zip(neighbors, weights), key=lambda it: it[1], eps=eps, stats=stats)
            nx, nw = result
            if nw > w:
                x, w = result
            else:
                return x
        else:
            result = search(zip(neighbors, weights), pred, eps=eps, stats=stats, rng=rng)
            if result is None:
                return x



def run(
    k: int,
    r: int,
    n: int,
    *,
    n_runs: int,
    rng: np.random.Generator,
    eps: Optional[float] = None,
    random_weights: Optional[Callable[[int], npt.NDArray[W]]] = None,
    steep: bool = False
) -> pd.DataFrame:
    history = []
    for run_ix in range(n_runs):
        logging.debug(f"k={k}, r={r}, n={n}, steep={steep}, #{run_ix}")

        # run hill climber on random instance
        stats = QueryStats()
        inst = WeightedSatInstance.random(
            k=k, n=n, m=r * n, rng=rng, random_weights=random_weights
        )
        hill_climber(inst, eps=eps, stats=stats, rng=rng, steep=steep)

        # save record to history
        rec = asdict(stats)
        rec["impl"] = "QuBRA"
        rec["n"] = n
        rec["k"] = k
        rec["r"] = r
        history.append(rec)

    # return pandas dataframe
    df = pd.DataFrame([list(row.values()) for row in history], columns=list(rec))
    logging.info(
        df.groupby(["k", "r", "n"]).mean(numeric_only=True)
    )  # TODO: get rid of numeric_only once 'impl' is gone
    return df
