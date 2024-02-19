"""This module contains the hillclimber examples as seen in Cade et al.'s 2022 paper Grover beyond asymptotics."""

import logging
from dataclasses import asdict
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from sat import Assignment, W, WeightedSatInstance

from qubrabench.algorithms.max import max
from qubrabench.algorithms.search import search
from qubrabench.benchmark import oracle, track_queries


def hill_climber(
    inst: WeightedSatInstance,
    *,
    rng: np.random.Generator,
    error: Optional[float] = None,
    steep: bool = False,
) -> Optional[Assignment]:
    """A hillclimbing heuristic to find maximizing assignments to weighted SAT instances
        by progressively transitioning to better solutions using neighborhood search.


    Args:
        inst: The SAT instance to be solved.
        rng: Source of randomness
        error: upper bound on the failure probability. Defaults to None.
        steep: True when the neighborhood search is performed greedily, otherwise randomly. Defaults to False.

    Returns:
        Optional[Assignment]: The best assignment found by the heuristic.
    """
    if rng is None:
        rng = np.random.default_rng()

    # precompute some matrices (see 4.3.2 in Cade et al)
    n = inst.n
    # generates a 1xN matrix containing only ones
    ones = np.ones(n, dtype=int)
    # produce nxn matrix, where all diagonal entries are -1, all other entries are 1
    # 1. np.outer generates outer product, which results in matrix containing only ones
    # 2. np.eye generates identity matrix, this is multiplied by 2 and then subtracted from the ones matrix
    flip_mat = np.outer(ones, ones) - 2 * np.eye(n, dtype=int)

    # start with a random assignment, -1 represents false, 1 represents true
    x = rng.choice([-1, 1], n)
    w = inst.weight(x)

    # error probability per hillclimb step, assuming a maximum of `n` rounds (see 4.3.1 in https://arxiv.org/pdf/2203.04975.pdf)
    if error is not None:
        error /= n

    @oracle
    def hillclimb_step():
        nonlocal x, w, inst

        # compute all Hamming neighbors (row by row)
        # np.outer(ones, x) generates matrix containing the current assignment x in n rows
        # flip_matrix used to flip values on diagonal to compute all possible neighbours
        # this only takes into account neighbours with a hamming distance of 1
        # uses numpys __mul__ operation, which performs element-wise multiplication, not matrix multiplication
        neighbors = flip_mat * np.outer(ones, x)

        # OPTION 1: "realistic" implementation (what should be done in case we cared about really large instances)
        # result = search(neighbors, lambda x: inst.weight(x) > w, error=error, stats=stats)
        # if result is None:
        #     return x
        # x, w = result, inst.weight(result)

        # OPTION 2: faster implementation (for our instance sizes)
        # precompute the weight for each possible neighbour
        weights = inst.weight(neighbors)

        if steep:
            result = max(
                zip(neighbors, weights),
                key=oracle(lambda it: it[1]),
                error=error,
            )
            if result is None:
                return x

            nx, nw = result
            if nw > w:
                x, w = result
            else:
                return x
        else:
            result = search(
                zip(neighbors, weights),
                key=oracle(lambda it: it[1] > w),
                error=error,
                rng=rng,
            )
            if result is None:
                return x
            x, w = result

    while True:
        res = hillclimb_step()
        if res is not None:
            return res


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
        inst = WeightedSatInstance.random(
            k=k, n=n, m=r * n, rng=rng, random_weights=random_weights
        )
        with track_queries() as tracker:
            hill_climber(inst, error=error, rng=rng, steep=steep)
            stats = tracker.get_function_stats_by_qualname(
                "hill_climber.<locals>.hillclimb_step.<locals>.<lambda>"
            )

            # save record to history
            rec = asdict(stats)
            rec["n"] = n
            rec["k"] = k
            rec["r"] = r
            rec["classical_control_method_calls"] = tracker.get_function_stats_by_name(
                "hillclimb_step"
            ).classical_actual_queries
            history.append(rec)

    # return pandas dataframe
    df = pd.DataFrame(
        [list(row.values()) for row in history], columns=list(history[0].keys())
    )
    return df
