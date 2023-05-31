"""This module provides the Schöning example for solving SAT instances."""
from typing import Optional
import numpy as np
import itertools
from functools import partial
from sat import SatInstance, Assignment
from qubrabench.stats import QueryStats
from qubrabench.algorithms.search import search

__all__ = ["schoening_solve"]


def schoening_solve(
    inst: SatInstance,
    *,
    rng: np.random.Generator,
    eps: Optional[float] = None,
    stats: Optional[QueryStats] = None,
) -> Optional[Assignment]:
    """
    Find a satisfying assignment of a 3-SAT formula by using Schöning's algorithm,
    which incrementally flipping bits contained in unsatisfied clauses.

    Arguments:
        inst: The 3-SAT Instance for which to find a satisfying assignment.
        rng: Random number generator
        eps (optional): Allowed failure probability.
        stats (optional): Object that keeps track of statistics about evaluation queries to the SAT instance.

    Returns: Satisfying assignment if found, None otherwise.
    """
    assert inst.k == 3

    # prepare search domain (all randomness used by Schoening's algorithm)
    n = inst.n
    assignments = [np.array(x, dtype=int) for x in itertools.product([-1, 1], repeat=n)]
    steps = [np.array(s, dtype=int) for s in itertools.product([0, 1, 2], repeat=3 * n)]
    domain = itertools.product(assignments, steps)

    # find a choice of randomness that makes Schoening's algorithm accept
    def pred(x):
        return schoening_with_randomness(x, inst) is not None

    randomness = search(domain, pred, eps=eps, stats=stats, rng=rng)

    # return satisfying assignment (if any was found)
    if randomness is not None:
        return schoening_with_randomness(randomness, inst)
    return None


def schoening_with_randomness(randomness: np.array, inst: SatInstance) -> bool:
    """
    Run Schoening's algorithm with fixed random choices.

    Arguments:
        randomness: An array containing a starting assignment, as well as a random bit string,
        encoding positions of bit flips.
        inst: The sat instance for which to compute a satisfying assignment.

    Returns: atisfying assignment if found, None otherwise.
    """
    # split randomness into initial assignment and steps
    assignment, steps = randomness
    assignment = np.copy(assignment)

    for i in range(0, len(steps)):
        # done?
        if inst.evaluate(assignment):
            return assignment

        # Evaluate all clauses with current assignment
        sat_clauses = (inst.clauses @ assignment.T) > -inst.k

        # Get the index of the first unsatisfied clauses
        clause_index = np.amin(np.argwhere(sat_clauses == False))

        # Get indices of variables contained in that clause
        target_clause = np.argwhere(inst.clauses[clause_index] != 0)

        # Select index of a variable that appears in unsatisfied clause
        target_variable = target_clause[steps[i]]

        # flip variable that appears in an unsatisfied clause
        # FIXME: this is not what the code currently does
        assignment[target_variable] *= -1

    return None
