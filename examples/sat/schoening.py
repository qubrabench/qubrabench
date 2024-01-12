"""This module provides the Schöning example for solving SAT instances."""
from typing import Optional
import numpy as np
import attrs

from qubrabench.algorithms.search import (
    search_by_sampling_with_replacement,
    SamplingDomain,
)
from qubrabench.benchmark import oracle, named_oracle

from sat import SatInstance, Assignment

__all__ = ["schoening_solve", "schoening_bruteforce_steps"]


@oracle
def schoening_with_randomness(
    inst: SatInstance, initial_assignment: np.ndarray, walk_steps: np.ndarray
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
class SchoeningDomain(SamplingDomain[tuple[np.ndarray, np.ndarray]]):
    """The class SchoeningDomain implements the methods declared for a SamplingDomain. We use this class for search spaces that cannot be stored in memory."""

    n: int
    n_assignment: int
    n_walk_steps: int

    @staticmethod
    def generate_random_assignment(n, rng: np.random.Generator):
        return rng.integers(2, size=n) * 2 - 1

    @staticmethod
    def generate_walk_steps(m, rng: np.random.Generator):
        return rng.integers(3, size=m)

    def get_size(self) -> int:
        return 2**self.n_assignment * 3**self.n_walk_steps

    def get_probability_of_sampling_solution(self, key) -> float:
        return 0.75**self.n

    def get_random_sample(
        self, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
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
        key=named_oracle("inst.schoening")(
            lambda r: schoening_with_randomness(inst, r[0], r[1]) is not None
        ),
        error=error,
        rng=rng,
    )

    # return satisfying assignment (if any was found)
    if randomness is not None:
        assignment, steps = randomness
        return schoening_with_randomness(inst, assignment, steps)
    return None


def schoening_bruteforce_steps(
    inst: SatInstance,
    *,
    rng: np.random.Generator,
    error: float,
) -> Optional[Assignment]:
    """
    Find a satisfying assignment of a 3-SAT formula by using a variant of Schöning's algorithm,
    bruteforcing over all sequences of steps.

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
        for _ in range(10**5):
            initial_assignment = SchoeningDomain.generate_random_assignment(n, rng)
            assignment = schoening_with_randomness(inst, initial_assignment, steps)
            if assignment is not None:
                return assignment
        return None

    result = search_by_sampling_with_replacement(
        domain,
        key=named_oracle("inst.schoening")(lambda r: pred(r) is not None),
        rng=rng,
        error=error,
    )

    if result is not None:
        return pred(result)
    return None
