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



