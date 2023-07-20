"""This module provides the Schöning example for solving SAT instances."""
from typing import Optional
import numpy as np
from sat import SatInstance, Assignment
from qubrabench.stats import QueryStats
from qubrabench.algorithms.search import search

__all__ = ["schoening_solve"]


def schoening_solve(
    inst: SatInstance,
    *,
    rng: np.random.Generator,
    error: Optional[float] = None,
    stats: Optional[QueryStats] = None,
) -> Optional[Assignment]:
    """
    Find a satisfying assignment of a 3-SAT formula by using Schöning's algorithm,
    which incrementally flipping bits contained in unsatisfied clauses.

    Args:
        inst: The 3-SAT Instance for which to find a satisfying assignment.
        rng: Random number generator.
        error: Allowed failure probability.
        stats: Object that keeps track of statistics about evaluation queries to the SAT instance.

    Returns:
        Satisfying assignment if found, None otherwise.
    """
    assert inst.k == 3

    # prepare search domain (all randomness used by Schoening's algorithm)
    n = inst.n

    # this function returns a pair of integers
    def domain(i):
        k = i % (2**n)
        assignment = list(
            map(
                lambda x: (int(x) == 0) * -1 + (not (int(x) == 0)) * int(x),
                list(f"{k:0{inst.n}b}"),
            )
        )
        steps = list(map(int, np.base_repr(i - k, base=3).zfill(3 * inst.n)))
        return (assignment, steps)

    size = 2 ** (n) * 3 ** (3*n)

    # find a choice of randomness that makes Schoening's algorithm accept
    def pred(x):
        return schoening_with_randomness(x, inst) is not None

    randomness = search(domain, pred, size, 100000, error=error, stats=stats, rng=rng)

    # return satisfying assignment (if any was found)
    if randomness is not None:
        return schoening_with_randomness(randomness, inst)
    return None


def schoening_with_randomness(randomness, inst: SatInstance) -> Optional[Assignment]:
    """
    Run Schoening's algorithm with fixed random choices.

    Args:
        randomness: An pair consisting of a starting assignment (bit string) and steps (trit string).
        inst: The 3-SAT Instance for which to find a satisfying assignment.

    Returns:
        Satisfying assignment if found, None otherwise.
    """
    # split randomness into initial assignment and steps
    # branchless way to convert zeroes into -1. Done for easier handling with instance
    assignment, steps = randomness
    assignment = np.copy(assignment)

    for i in range(0, len(steps)):
        # done?
        if inst.evaluate(assignment):
            return assignment

        # choose an unsatisfied clause
        unsat_clauses = (inst.clauses @ assignment.T) == -inst.k
        unsat_clause = inst.clauses[unsat_clauses][0]

        # select a variable that appears in unsatisfied clause, and flip it
        vars_ = np.argwhere(unsat_clause != 0)
        var = vars_[steps[i]]
        assignment[var] *= -1

    return None
