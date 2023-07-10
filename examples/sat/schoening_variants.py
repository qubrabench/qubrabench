"""This module provides the Schöning example for solving SAT instances."""
from typing import Optional
import numpy as np
import itertools
from sat import SatInstance, Assignment
from qubrabench.stats import QueryStats
from qubrabench.algorithms.search import search
from schoening import schoening_with_randomness

__all__ = [
    "schoening_bruteforce_assignment",
    "schoening_bruteforce_steps",
    "schoening_mixed_solve",
]


def schoening_bruteforce_assignment(
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


def schoening_bruteforce_steps(
    inst: SatInstance,
    *,
    rng: np.random.Generator,
    error: Optional[float] = None,
    stats: Optional[QueryStats] = None,
) -> Optional[Assignment]:
    """
    Find a satisfying assignment of a 3-SAT formula by using a variant of Schöning's algorithm,
    bruteforcing over all sequences of steps.

    Args:
        inst: The 3-SAT Instance for which to find a satisfying assignment.
        rng: Random number generator.
        error: Allowed failure probability.
        stats: Object that keeps track of statistics about evaluation queries to the SAT instance.

    Returns:
        Satisfying assignment if found, None otherwise.
    """
    assert inst.k == 3

    # prepare search domain comprising all assignments
    n = inst.n
    domain = [np.array(x, dtype=int) for x in itertools.product([-1, 1], repeat=n)]
    steps = [np.array(s, dtype=int) for s in itertools.product([0, 1, 2], repeat=3 * n)]

    # variable for the oracle to store sequence of steps that works
    sat_steps = None

    # find a choice of randomness that makes Schoening's algorithm accept by bruteforcing over steps
    def pred(assignments):
        for step in steps:
            # if current assignment is satisfying, sets variable in outer scope
            if schoening_with_randomness((assignments, step), inst) is not None:
                nonlocal sat_steps
                sat_steps = step
                return True
        return False

    randomness = search(domain, pred, error=error, stats=stats, rng=rng)
    sat_pair = (randomness, sat_steps)

    # return satisfying assignment (if any was found)
    if sat_steps is not None:
        return schoening_with_randomness(sat_pair, inst)
    return None


def schoening_mixed_solve(
    inst: SatInstance,
    *,
    rng: np.random.Generator,
    error: Optional[float] = None,
    stats: Optional[QueryStats] = None,
) -> Optional[Assignment]:
    """
    Find a satisfying assignment of a 3-SAT formula by using a variant of Schöning's algorithm,
    which bruteforces over a subset of both assignments and steps.

    Args:
        inst: The 3-SAT Instance for which to find a satisfying assignment.
        rng: Random number generator.
        error: Allowed failure probability.
        stats: Object that keeps track of statistics about evaluation queries to the SAT instance.

    Returns:
        Satisfying assignment if found, None otherwise.
    """
    assert inst.k == 3

    # FIXME: this defines the length of both the randomness and assignment to bruteforce over. What value should this have?
    bruteforce_length = 2

    # prepare search domain comprising partial assignments and partial steps
    n = inst.n
    assignments = [
        np.array(x, dtype=int)
        for x in itertools.product([-1, 1], repeat=n - bruteforce_length)
    ]
    steps = [
        np.array(s, dtype=int)
        for s in itertools.product([0, 1, 2], repeat=3 * (n - bruteforce_length))
    ]
    domain = itertools.product(assignments, steps)

    # the space over which to bruteforce
    bruteforce_assignments = [
        np.array(x, dtype=int)
        for x in itertools.product([-1, 1], repeat=bruteforce_length)
    ]
    bruteforce_steps = [
        np.array(s, dtype=int)
        for s in itertools.product([0, 1, 2], repeat=3 * bruteforce_length)
    ]
    bruteforce = itertools.product(bruteforce_assignments, bruteforce_steps)

    # variable for the oracle to store string that yields sat assignment
    sat_string = None

    # find a choice of randomness that makes Schoening's algorithm accept
    def pred(x):
        # split initial input up into assignment and steps
        initial_assignment, initial_steps = x
        for string in bruteforce:
            # join initial and bruteforce parts for search
            assignment, steps = string
            item = (
                np.concatenate((initial_assignment, assignment), dtype=None),
                np.concatenate((initial_steps, steps), dtype=None),
            )

            # if current assignment is satisfying, sets variable in outer scope
            if schoening_with_randomness(item, inst) is not None:
                nonlocal sat_string
                sat_string = string
                return True
        return False

    # join result of search with the result of bruteforce stored by oracle
    randomness = search(domain, pred, error=error, stats=stats, rng=rng)
    initial_assignment, initial_steps = randomness
    assignment, steps = sat_string
    sat_pair = (
        np.concatenate((initial_assignment, assignment), dtype=None),
        np.concatenate((initial_steps, steps), dtype=None),
    )

    # return satisfying assignment (if any was found)
    if sat_pair is not None:
        return schoening_with_randomness(sat_pair, inst)
    return None
