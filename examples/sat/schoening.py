"""This module provides the Schöning example for solving SAT instances."""

import itertools
from typing import Optional

import numpy as np
from sat import Assignment, SatInstance

from qubrabench.algorithms.search import search
from qubrabench.benchmark import oracle

__all__ = ["schoening_solve"]


def schoening_solve(
    inst: SatInstance,
    *,
    rng: np.random.Generator,
    error: Optional[float] = None,
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
    assignments = [np.array(x, dtype=int) for x in itertools.product([-1, 1], repeat=n)]
    steps = [np.array(s, dtype=int) for s in itertools.product([0, 1, 2], repeat=3 * n)]
    domain = itertools.product(assignments, steps)

    # find a choice of randomness that makes Schoening's algorithm accept
    randomness = search(
        domain,
        key=lambda x: schoening_with_randomness(inst, x) is not None,
        error=error,
        rng=rng,
    )

    # return satisfying assignment (if any was found)
    if randomness is not None:
        return schoening_with_randomness(inst, randomness)
    return None


@oracle
def schoening_with_randomness(inst: SatInstance, randomness) -> Optional[Assignment]:
    """
    Run Schoening's algorithm with fixed random choices.

    Args:
        randomness: An pair consisting of a starting assignment (bit string) and steps (trit string).
        inst: The 3-SAT Instance for which to find a satisfying assignment.

    Returns:
        Satisfying assignment if found, None otherwise.
    """
    # split randomness into initial assignment and steps
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
