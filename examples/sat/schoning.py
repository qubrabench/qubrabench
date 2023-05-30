"""This module provides the Schöning example for solving SAT instances."""
from typing import Optional
import numpy as np
import itertools
from functools import partial
from sat import SatInstance, Assignment
from qubrabench.stats import QueryStats
from qubrabench.algorithms.search import search


def schoening_solve(
    inst: SatInstance,
    *,
    rng: np.random.Generator,
    eps: Optional[float] = None,
    stats: Optional[QueryStats] = None,
) -> Optional[Assignment]:
    """ Find a satisfying assignment by incrementally flipping bits contained in unsatisfied clauses,
        as done in Schoening's algorithm for Sat solving.

    Args:
        inst (SatInstance): The Sat Instance for which to find a satisfying assignment.
        rng (np.random.Generator): Source of randomness
        eps (Optional[float], optional): Upper bound on the failure probability. Defaults to None.
        stats (Optional[QueryStats], optional): Statistics instance, allows collecting classical
        and estimated quantum query count. Defaults to None.

    Returns:
        Optional[Assignment]: Returns either a satisfying assignment, together with the string of random bits
        representing the assignment flips. If no such assignment is found, returns a None.
    """
    n = inst.n
    assignments = [np.array(x, dtype=int) for x in itertools.product([-1, 1], repeat=n)]
    seeds = [np.array(x, dtype=int) for x in itertools.product([0, 1, 2], repeat=3 * n)]

    # elements of the form (assignment) X (Random bits)
    domain = itertools.product(assignments, seeds)

    # Supply schoening with SatInstance to check against
    schoening_with_sat = partial(schoening, inst=inst)

    # Predicate for the search function. Uses SatInstance and rng from context.
    return search(domain, schoening_with_sat, eps=eps, stats=stats, rng=rng)


def schoening(
    x: np.array,
    inst: SatInstance,
) -> bool:
    """ Search function employing Schoenings algorithm for sat solving.

    Args:
        x (np.array): An array containing a starting assignment, as well as a random bit string,
        encoding positions of bit flips.
        inst (SatInstance): The sat instance for which to compute a satisfying assignment.

    Returns:
        bool: Returns True if the initial assignment collapses into a satisfying clauses
        at some point during the sequence of incremental bit flips.
        Returns None if that does not happen.
    """
    # Split input into assignment and seed
    (assignment, seed) = zip(x)
    assignment = np.copy(assignment[0])
    seed = seed[0]

    for i in range(0, len(seed)):
        if inst.evaluate(assignment):
            return True
        # Flip a bit
        assignment[seed[i]] *= -1

    return False
