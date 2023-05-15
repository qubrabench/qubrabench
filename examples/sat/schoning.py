from typing import Optional
import numpy as np
import itertools
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

    domain = [np.array(x, dtype=int) for x in itertools.product([-1, 1], repeat=inst.n)]

    # Predicate for the search function. Uses SatInstance and rng from context.
    def schoning(
        x: np.array
    ) -> bool:

        n = inst.n
        for _ in range(0, 3 * n):
            if inst.evaluate(x):
                return True
            x = flip_random_variable(x, rng)

        return False

    return search(domain, schoning, eps=eps, stats=stats, rng=rng)

def flip_random_variable(
    x: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    x[rng.choice(np.arange(len(x)))] *= -1
    return x
