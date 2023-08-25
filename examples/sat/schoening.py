"""This module provides the Schöning example for solving SAT instances."""
from typing import Optional
import numpy as np
from sat import SatInstance, Assignment
from qubrabench.stats import QueryStats
from qubrabench.algorithms.search import search, sample, size, num_solutions

__all__ = ["schoening_solve"]


class Schoening:
    def __init__(self, n):
        self.n = n

@sample.instance(Schoening)
def schoening_sample(instance : Schoening, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    assignment = rng.integers(2, size=instance.n) * 2 - 1
    steps = rng.integers(3, size=3 * instance.n)
    return assignment, steps

@size.instance(Schoening)
def schoening_size(instance: Schoening) -> int:
    N_assignment = 2 ** instance.n
    N_steps = 3 ** (3 * instance.n)
    return N_assignment * N_steps

@num_solutions.instance(Schoening)
def schoening_num_solutions(instance: Schoening, _) -> float:
    return pow(0.75, instance.n) * size(instance)


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
    domain = Schoening(inst.n)

    # find a choice of randomness that makes Schoening's algorithm accept
    def pred(x):
        return schoening_with_randomness(x, inst) is not None

    randomness = search(
        domain,
        pred,
        error=error,
        stats=stats,
        rng=rng,
    )

    # return satisfying assignment (if any was found)
    if randomness is not None:
        return schoening_with_randomness(randomness, inst)
    return None


def schoening_with_randomness(
    randomness: tuple[np.ndarray, np.ndarray], inst: SatInstance
) -> Optional[Assignment]:
    """
    Run Schoening's algorithm with fixed random choices.

    Args:
        randomness: An pair consisting of a starting assignment (bit string) and steps (trit string).
        inst: The 3-SAT Instance for which to find a satisfying assignment.

    Returns:
        Satisfying assignment if found, None otherwise.
    """
    assignment, steps = randomness
    assignment = np.copy(assignment)  # to avoid editing the input

    for step in steps:
        # done?
        if inst.evaluate(assignment):
            return assignment

        # find the first unsatisfied clause
        unsat_clauses = (inst.clauses @ assignment.T) == -inst.k
        unsat_clause = inst.clauses[unsat_clauses][0]

        # select a variable that appears in unsatisfied clause, and flip it
        vars_ = np.argwhere(unsat_clause != 0)
        var = vars_[step]
        assignment[var] *= -1

    return None
