from typing import Optional
import numpy as np
import itertools
from functools import partial
from sat import SatInstance, Assignment
from qubrabench.stats import QueryStats
from qubrabench.algorithms.search import search


def schoning_solve(
    inst: SatInstance,
    *,
    rng: np.random.Generator,
    eps: Optional[float] = None,
    stats: Optional[QueryStats] = None,
) -> Optional[Assignment]:

    n = inst.n
    assignments = [np.array(x, dtype=int) for x in itertools.product([-1, 1], repeat=n)]
    seeds = [np.array(x, dtype=int) for x in itertools.product([0, 1, 2], repeat=3*n)]

    # elements of the form (assignment) X (Random bits)
    domain = itertools.product(assignments, seeds)

    # Supply schoning with SatInstance to check against
    schoning_with_sat = partial(schoning, inst=inst)

    # Predicate for the search function. Uses SatInstance and rng from context.
    return search(domain, schoning_with_sat, eps=eps, stats=stats, rng=rng)


def schoning(
    x: np.array,
    inst: SatInstance,
) -> bool:

    # Split input into assignment and seed
    (assignment, seed) = zip(x)
    assignment = np.copy(assignment[0])
    seed = seed[0]

    for i in range(0, len(seed)):
        if inst.evaluate(assignment):
            return True
        assignment[seed[i]] *= -1

    return False
