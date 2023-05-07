from typing import Optional
import itertools
import numpy as np
from sat import SatInstance, Assignment
from qubrabench.stats import QueryStats
from qubrabench.algorithms.search import search


# TODO: should not return QueryStats
def bruteforce_solve(
    inst: SatInstance,
    *,
    rng: np.random.Generator,
    eps: Optional[float] = None,
    stats: Optional[QueryStats] = None,
) -> Optional[Assignment]:
    # list of all bitstrings
    domain = [np.array(x, dtype=int) for x in itertools.product([-1, 1], repeat=inst.n)]

    # brute-force search
    return search(domain, inst.evaluate, eps=eps, stats=stats, rng=rng)
