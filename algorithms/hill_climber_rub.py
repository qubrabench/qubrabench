from dataclasses import dataclass, asdict
from typing import Callable, Iterable, TypeVar
import numpy as np
import random
import scipy
import pandas as pd

from bench.stats import QueryStats
import bench.qsearch as qsearch


@dataclass(frozen=True)
class MaxSatInstance:
    """
    As in 4.3.2 in Cade et al, clauses are represented by vectors in {-1,0,1}^n and assignments by vectors in {-1,1}^n.
    """

    k: int  # number of literals per clause
    clauses: np.ndarray
    weights: np.ndarray

    @property
    def n(self):
        """Number of variables"""
        return self.clauses.shape[1]

    def weight(self, assignment):
        sat_clauses = (self.clauses @ assignment.T) == self.k
        return self.weights @ sat_clauses

    @staticmethod
    def random(k, n, m):
        """
        Generate a random k-SAT instance with n variables and m clauses.
        """
        # generate random clauses (m x n matrix)
        clauses = np.zeros(shape=(m, n), dtype=int)
        for i in range(m):
            vs = np.random.choice(n, k, replace=False)
            clauses[i][vs] = np.random.choice([-1, 1], k)
        
        clauses = scipy.sparse.csr_matrix(clauses)

        # generate random weights in [0,1]
        weights = np.random.random(m)
        return MaxSatInstance(k, clauses, weights)

T = TypeVar("T")


def search(
    seq: Iterable[T],
    predicate: Callable[[T], bool],
    *,
    eps,
    K=130,
    stats: QueryStats = None,
):
    """
    Search a list by random sampling (and keep track of classical and quantum stats).

    TODO: Think about how to interpret eps for the classical algorithm.
    """
    seq = list(seq)

    # collect stats
    if stats:
        N = len(seq)
        T = sum(1 for x in seq if predicate(x))
        stats.classical_expected_queries += (N + 1) / (T + 1)
        stats.quantum_expected_classical_queries += qsearch.estimate_classical_queries(N, T, K)
        stats.quantum_expected_quantum_queries += qsearch.estimate_quantum_queries(N, T, eps, K)

    # run the classical sampling-without-replacement algorithms
    random.shuffle(seq)
    for x in seq:
        if stats:
            stats.classical_actual_queries += 1
        if predicate(x):
            return x


def simple_hill_climber(
    inst: MaxSatInstance, *, eps=10**-5, stats: QueryStats = None
):
    # precompute some matrices (see 4.3.2 in Cade et al)
    n = inst.n
    ones = np.ones(n, dtype=int)
    flip_mat = np.outer(ones, ones) - 2 * np.eye(n, dtype=int)

    # start with a random assignment
    x = np.random.choice([-1, 1], n)
    w = inst.weight(x)

    while True:
        # compute all Hamming neighbors (row by row) and their weights
        neighbors = flip_mat * np.outer(ones, x)
        weights = inst.weight(neighbors)
        # better = np.flatnonzero(weights > w)

        # find improved directions
        stats.classical_control_method_calls += 1
        result = search(  # TODO this is called too often!
            zip(neighbors, weights), lambda it: it[1] > w, eps=eps, stats=stats
        )
        if not result:
            return x
        x, w = result

def run(k, r, n, runs, dest, verbose):
    history = []
    for run in range(runs):
        if verbose:
            print(f"k={k}, r={r}, n={n}, #{run}")
        stats = QueryStats()
        inst = MaxSatInstance.random(k=k, n=n, m=r * n)
        simple_hill_climber(inst, stats=stats)
        stats = asdict(stats)
        stats["impl"] = "RUB"
        stats["n"] = n
        stats["k"] = k
        stats["r"] = r
        history.append(stats)

    history = pd.DataFrame(
        [list(row.values()) for row in history],
        columns=stats.keys(),
    )

    # print summary
    print(history.groupby(["k", "r", "n"]).mean())

    # save
    if dest is not None:
        print(f"saving to {dest}...")
        orig = pd.read_json(dest, orient="split") if dest.exists() else None
        history = pd.concat([orig, history])
        with dest.open("w") as f:
            f.write(history.to_json(orient="split"))

