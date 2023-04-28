from dataclasses import asdict, dataclass
import numpy as np
import random
import logging
import pandas as pd
import scipy

from qubrabench.bench.stats import QueryStats
from qubrabench.algorithms.search import search as search


@dataclass(frozen=True)
class SatInstance:
    """
    As in 4.3.2 in Cade et al, clauses are represented by vectors in {-1,0,1}^n and assignments by vectors in {-1,1}^n.
    """

    k: int  # number of literals per clause
    clauses: np.ndarray  # (clauses_no x literals_no) matrix

    @property
    def n(self):
        """Number of variables"""
        return self.clauses.shape[1]

    @property
    def m(self):
        """Number of clauses"""
        return self.clauses.shape[0]

    def evaluate(self, assignment):
        clauses = (self.clauses * assignment) > 0
        return np.all([np.any(c) for c in list(clauses)])

    @staticmethod
    def random(k, n, m, *, seed=None):
        """
        Generate a random k-SAT instance with n variables and m clauses.
        """
        # generate random clauses (m x n matrix)
        if seed is not None:
            np.random.seed(seed)
        clauses = np.zeros(shape=(m, n), dtype=int)
        for i in range(m):
            vs = np.random.choice(n, k, replace=False)
            clauses[i][vs] = np.random.choice([-1, 1], k)

        clauses = scipy.sparse.csr_matrix(clauses)

        return SatInstance(k, clauses)


# TODO: the algorithm (brute-force search for SAT solving) should have its own function
def run_specific_instance():
    inst = SatInstance(
        k=2, clauses=np.array([[1, 0, 0], [0, 1, 1], [0, -1, -1]], dtype=int)
    )

    # TODO: run multiple times to obtain proper statistics
    n = inst.n
    search_space = np.full((2**n, n), 1, dtype=int)
    for i in range(n):
        for start in range(0, 2**n, 2 ** (i + 1)):
            for j in range(2**i):
                search_space[start + j, i] = -1

    stats = QueryStats()
    search(search_space, lambda x: inst.evaluate(x), stats=stats, eps=10**-5)

    stats = asdict(stats)
    stats["n"] = n
    stats["k"] = inst.k
    stats["m"] = inst.m

    T = 0
    for x in list(search_space):
        if inst.evaluate(x):
            T += 1
    stats["T"] = T

    return stats
