"""Module concerning solving SAT instance by brute forcing."""

import itertools
from typing import Optional

import numpy as np
from sat import Assignment, SatInstance

import qubrabench as qb


def bruteforce_solve(
    inst: SatInstance,
    *,
    rng: np.random.Generator,
    error: Optional[float] = None,
) -> Optional[Assignment]:
    """Find a satisfying solution to a SAT instance by randomly searching for solutions.

    Args:
        inst: The instance to be solved
        rng: Source of randomness
        error: Upper bound on the quantum failure rate. Defaults to None.

    Returns:
        The found assignment satisfying the instance.
    """
    # list of all bitstrings
    domain = [np.array(x, dtype=int) for x in itertools.product([-1, 1], repeat=inst.n)]

    # brute-force search
    return qb.algorithms.search.search(
        domain, inst.evaluate, max_fail_probability=error, rng=rng
    )
