"""This module provides the Schöning example for solving SAT instances."""

from typing import Optional, TypeAlias

import attrs
import numpy as np
import numpy.typing as npt
from sat import Assignment, SatInstance

from qubrabench.algorithms.search import (
    SamplingDomain,
    search_by_sampling_with_replacement,
)
from qubrabench.benchmark import oracle

__all__ = [
    "schoening_solve",
    "schoening_solve__bruteforce_over_starting_assigment",
    "schoening_random_walk",
]

WalkSteps: TypeAlias = npt.NDArray[np.int_]


@oracle
def schoening_random_walk(
    inst: SatInstance, initial_assignment: Assignment, walk_steps: WalkSteps
) -> Optional[Assignment]:
    """
    Run Schoening's algorithm with fixed random choices.

    Args:
        initial_assignment: starting assignment (bit string)
        walk_steps: choices to take at each step of the random walk
        inst: The 3-SAT Instance for which to find a satisfying assignment.

    Returns:
        Satisfying assignment if found, None otherwise.
    """
    assignment = np.copy(initial_assignment)

    if inst.evaluate(assignment):
        return assignment

    # iterate over the steps of the random walk
    for step in walk_steps:
        # choose an unsatisfied clause
        unsat_clauses = (inst.clauses @ assignment.T) == -inst.k
        unsat_clause = inst.clauses[unsat_clauses][0]

        # select a variable that appears in unsatisfied clause, and flip it
        vars_ = np.argwhere(unsat_clause != 0)
        var = vars_[step]
        assignment[var] *= -1

        if inst.evaluate(assignment):
            return assignment

    return None


@attrs.define
class SchoeningDomain(SamplingDomain[tuple[Assignment, WalkSteps]]):
    """Represents a sampling domain for the random choices in the Schoening algorithm."""

    n: int
    """number of variables in the 3-SAT instance"""

    n_assignment: int
    """number of random bits used to generate the starting assigment of the random walk"""

    n_walk_steps: int
    """number of random trits used to generate the walk steps (i.e. variable to flip in unsat clause)"""

    @staticmethod
    def generate_random_assignment(n, rng: np.random.Generator) -> Assignment:
        return rng.integers(2, size=n) * 2 - 1

    @staticmethod
    def generate_walk_steps(m, rng: np.random.Generator) -> WalkSteps:
        return rng.integers(3, size=m)

    def get_size(self) -> int:
        return 2**self.n_assignment * 3**self.n_walk_steps

    def get_probability_of_sampling_solution(self, key) -> float:
        return 0.75**self.n

    def get_random_sample(
        self, rng: np.random.Generator
    ) -> tuple[Assignment, WalkSteps]:
        return (
            self.generate_random_assignment(self.n_assignment, rng),
            self.generate_walk_steps(self.n_walk_steps, rng),
        )


def schoening_solve(
    inst: SatInstance,
    *,
    rng: np.random.Generator,
    error: float,
) -> Optional[Assignment]:
    """
    Find a satisfying assignment of a 3-SAT formula by using Schoening's algorithm,
    which incrementally flips assigned variables contained in unsatisfied clauses.

    Args:
        inst: The 3-SAT Instance for which to find a satisfying assignment.
        rng: Random number generator.
        error: Allowed failure probability.

    Returns:
        Satisfying assignment if found, None otherwise.
    """
    # schoening's algorithm is defined for 3-SAT instances.
    assert inst.k == 3

    # find a choice of randomness that makes Schoening's algorithm accept
    randomness = search_by_sampling_with_replacement(
        SchoeningDomain(inst.n, inst.n, 3 * inst.n),
        key=lambda r: schoening_random_walk(inst, r[0], r[1]) is not None,
        error=error,
        rng=rng,
    )

    # return satisfying assignment (if any was found)
    if randomness is not None:
        assignment, steps = randomness
        return schoening_random_walk(inst, assignment, steps)
    return None


def schoening_solve__bruteforce_over_starting_assigment(
    inst: SatInstance,
    *,
    rng: np.random.Generator,
    error: float,
) -> Optional[Assignment]:
    """
    Find a satisfying assignment of a 3-SAT formula by using a variant of Schöning's algorithm,
    by running quantum search over the walk steps, and bruteforcing over the starting assignment.

    Args:
        inst: The 3-SAT Instance for which to find a satisfying assignment.
        rng: Random number generator.
        error: Allowed failure probability.

    Returns:
        Satisfying assignment if found, None otherwise.
    """
    assert inst.k == 3

    # prepare search domain comprising all assignments
    n = inst.n
    domain = SchoeningDomain(n, 0, 3 * n)

    # define predicate, if steps lead to satisfying assignment return that
    def pred(steps) -> Optional[Assignment]:
        for _ in range(10**3):
            initial_assignment = SchoeningDomain.generate_random_assignment(n, rng)
            assignment = schoening_random_walk(inst, initial_assignment, steps)
            if assignment is not None:
                return assignment
        return None

    result = search_by_sampling_with_replacement(
        domain,
        key=lambda r: pred(r) is not None,
        rng=rng,
        error=error,
    )

    if result is not None:
        return pred(result)
    return None
