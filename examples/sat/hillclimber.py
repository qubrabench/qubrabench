"""This module contains the hillclimber examples as seen in Cade et al.'s 2022 paper Grover beyond asymptotics."""
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


def hill_climber(
    inst: WeightedSatInstance,
    *,
    rng: np.random.Generator,
    error: Optional[float] = None,
    stats: Optional[QueryStats] = None,
    steep: bool = False,
) -> Optional[Assignment]:
    """A hillclimbing heuristic to find maximizing assignments to weighted SAT instances
        by progressively transitioning to better solutions using neighborhood search.


    Args:
        inst: The SAT instance to be solved.
        rng: Source of randomness
        error: upper bound on the failure probability. Defaults to None.
        stats: Statistics instance keeping track of costs. Defaults to None.
        steep: True when the neighborhood search is performed greedily, otherwise randomly. Defaults to False.

    Returns:
        Optional[Assignment]: The best assignment found by the heuristic.
    """
    if rng is None:
        rng = np.random.default_rng()

    # precompute some matrices (see 4.3.2 in Cade et al)
    n = inst.n
    ones = np.ones(n, dtype=int)
    flip_mat = np.outer(ones, ones) - 2 * np.eye(n, dtype=int)

    # start with a random assignment
    x = rng.choice([-1, 1], n)
    w = inst.weight(x)

    # error probability per hillclimb step, assuming a maximum of `n` rounds (see 4.3.1 in https://arxiv.org/pdf/2203.04975.pdf)
    if error is not None:
        error /= n

    while True:
        # compute all Hamming neighbors (row by row) and their weights
        neighbors = flip_mat * np.outer(ones, x)

        # find improved directions
        if stats:
            stats.classical_control_method_calls += 1

        # OPTION 1: "realistic" implementation (what should be done in case we cared about really large instances)
        # result = search(neighbors, lambda x: inst.weight(x) > w, error=error, stats=stats)
        # if result is None:
        #     return x
        # x, w = result, inst.weight(result)

        # OPTION 2: faster implementation (for our instance sizes)
        weights = inst.weight(neighbors)

        def pred(it: Tuple[Assignment, np.float_]) -> bool:
            return bool(it[1] > w)

        if steep:
            result = max(
                zip(neighbors, weights),
                key=lambda it: it[1],
                error=error,
                stats=stats,
            )
            nx, nw = result
            if nw > w:
                x, w = result
            else:
                return x
        else:
            result = search(
                zip(neighbors, weights),
                pred,
                error=error,
                stats=stats,
                rng=rng,
            )
            if result is None:
                return x
            x, w = result


def run(
    k: int,
    r: int,
    n: int,
    *,
    n_runs: int,
    rng: np.random.Generator,
    error: Optional[float] = None,
    random_weights: Optional[Callable[[int], npt.NDArray[W]]] = None,
    steep: bool = False,
) -> pd.DataFrame:
    """External interface to generate weighted sat instances, run the hillclimber algorithm and return statistics.

    Args:
        k: Number of literals in a clause
        r: Factor for the number of clauses
        n: size (variable number) of the SAT instances
        n_runs: number of runs to perform in each group
        rng: Source of randomness
        error: Upper bound on the failure rate. Defaults to None.
        random_weights: Optionally providable weights for SAT instance generation. Defaults to None.
        steep: Whether to perform hillclimb steep (greedily). Defaults to False.

    Returns:
        Dataframe holding benchmarking statistics of the runs performed.
    """
    assert n_runs >= 1

    history = []
    for run_ix in range(n_runs):
        logging.debug(f"k={k}, r={r}, n={n}, steep={steep}, #{run_ix}")

        # run hill climber on random instance
        stats = QueryStats()
        inst = WeightedSatInstance.random(
            k=k, n=n, m=r * n, rng=rng, random_weights=random_weights
        )
        hill_climber(inst, error=error, stats=stats, rng=rng, steep=steep)

        # save record to history
        rec = asdict(stats)
        rec["n"] = n
        rec["k"] = k
        rec["r"] = r
        history.append(rec)

    # return pandas dataframe
    df = pd.DataFrame(
        [list(row.values()) for row in history], columns=list(history[0].keys())
    )
    return df
