"""Module concerning solving SAT instance by brute forcing."""

from typing import Optional
import itertools
import numpy as np
from sat import SatInstance, Assignment
from qubrabench.stats import QueryStats
from qubrabench.algorithms.search import search


def bruteforce_solve(
    inst: SatInstance,
    *,
    rng: np.random.Generator,
    eps: Optional[float] = None,
    stats: Optional[QueryStats] = None,
) -> Optional[Assignment]:
    """Find a satisfying solution to a SAT instance by randomly searching for solutions.

    Args:
        inst (SatInstance): The instance to be solved
        rng (np.random.Generator): Source of randomness
        eps (Optional[float], optional): Upper bound on the quantum failure rate. Defaults to None.
        stats (Optional[QueryStats], optional): Statistics instance to keep track of query costs. Defaults to None.

    Returns:
        Optional[Assignment]: The found assignment satisfying the instance.
    """
    # list of all bitstrings
    domain = [np.array(x, dtype=int) for x in itertools.product([-1, 1], repeat=inst.n)]

    # brute-force search
    return search(domain, inst.evaluate, eps=eps, stats=stats, rng=rng)
