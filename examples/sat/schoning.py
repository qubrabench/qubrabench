from typing import Optional
import numpy as np
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
    # Setup random assignment.
    n = inst.n
    x = rng.choice([-1, 1], n)

    domain = []

    for _ in range(0, 3 * n):
        domain.append(x)
        x = flip_random_variable(x, rng)

    return search(domain, inst.evaluate, eps=eps, stats=stats, rng=rng)


def flip_random_variable(
    assignment: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    x = np.copy(assignment)
    x[rng.choice(np.arange(len(x)))] *= -1
    return x
