import numpy as np

from qubrabench.benchmark import QueryStats, track_queries
from qubrabench.datastructures.qlist import QList


def test_qlist_iterate(rng):
    for _ in range(10):
        N = rng.integers(100, 1000)
        xs = QList(rng.random(size=N))

        with track_queries() as tracker:
            _ = np.sum(xs)
            assert tracker.get_stats(xs) == QueryStats(classical_actual_queries=N)
            _ = np.max(xs)
            assert tracker.get_stats(xs) == QueryStats(classical_actual_queries=2 * N)
