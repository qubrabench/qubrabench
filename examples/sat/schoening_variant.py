"""This module provides the Schöning example for solving SAT instances."""
from typing import Optional
import numpy as np
import itertools
from sat import SatInstance, Assignment
from qubrabench.stats import QueryStats
from qubrabench.algorithms.search import search
from schoening import schoening_with_randomness

__all__ = ["schoening_variant_solve"]


def schoening_variant_solve(
    inst: SatInstance,
    *,
    rng: np.random.Generator,
    error: Optional[float] = None,
    stats: Optional[QueryStats] = None,
) -> Optional[Assignment]:
    """
    Find a satisfying assignment of a 3-SAT formula by using a variant of Schöning's algorithm,
    bruteforcing over all assignments.

    Args:
        inst: The 3-SAT Instance for which to find a satisfying assignment.
        rng: Random number generator.
        error: Allowed failure probability.
        stats: Object that keeps track of statistics about evaluation queries to the SAT instance.

    Returns:
        Satisfying assignment if found, None otherwise.
    """
    assert inst.k == 3

    # prepare search domain comprising the random steps
    n = inst.n
    assignments = [np.array(x, dtype=int) for x in itertools.product([-1, 1], repeat=n)]
    domain = [
        np.array(s, dtype=int) for s in itertools.product([0, 1, 2], repeat=3 * n)
    ]

    # variable for the oracle to store satisfying assignment
    sat_assignment = None

    # find a choice of randomness that makes Schoening's algorithm accept by bruteforcing over assignments
    def pred(steps):
        for assignment in assignments:
            # if current assignment is satisfying, sets variable in outer scope
            if schoening_with_randomness((assignment, steps), inst) is not None:
                nonlocal sat_assignment
                sat_assignment = assignment
                return True
        return False

    randomness = search(domain, pred, error=error, stats=stats, rng=rng)
    sat_pair = (sat_assignment, randomness)

    # return satisfying assignment (if any was found)
    if sat_assignment is not None:
        return schoening_with_randomness(sat_pair, inst)
    return None
