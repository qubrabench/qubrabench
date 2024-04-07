import numpy as np

from qubrabench.benchmark import QueryStats, track_queries
from qubrabench.datastructures.qlist import QList


def test_qlist_iterate(rng):
    for _ in range(10):
        N = rng.integers(100, 1000)
        xs = QList(rng.random(size=N))

        with track_queries():
            _ = np.sum(xs)
            assert xs.stats == QueryStats.from_true_queries(N)
            _ = np.max(xs)
            assert xs.stats == QueryStats.from_true_queries(2 * N)
