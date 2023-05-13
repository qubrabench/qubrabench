from typing import Optional
import itertools
import numpy as np
from sat import SatInstance, Assignment
from qubrabench.stats import QueryStats
from qubrabench.algorithms.search import search


def schoning(
    inst: SatInstance,
    *,
    rng: np.random.Generator,
    eps: Optional[float] = None,
    stats: Optional[QueryStats] = None,
) -> Optional[Assignment]:
    
    if rng is None:
        rng = np.random.default_rng()

    # Setup random assignment.
    n = inst.n
    x = rng.choice([-1, 1], n)

    for _ in 3*n:
        clauses = inst.evaluate(x)
        if clauses.size == 0:
            return x
        # Choose random falsified clause

        i = rng.choice(clauses.shape[1], 1, replace=False)
        clause = clauses[i]

        j = rng.choice(clause.shape, 1, replace=False)
        x[j] *= -1
        
        # Flip a variable in that clause and flip it

        




