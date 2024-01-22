import numpy as np

from qubrabench.benchmark import QueryStats, track_queries
from qubrabench.datastructures.matrix import QMatrix


def test_qmatrix_iterate(rng):
    for _ in range(5):
        N, M = rng.integers(10, 50, size=2)
        mat = QMatrix(rng.random(size=(N, M)))

        with track_queries() as tracker:
            _ = np.sum(mat)
            assert tracker.get_stats(mat) == QueryStats(classical_actual_queries=N * M)
            _ = np.max(mat)
            assert tracker.get_stats(mat) == QueryStats(
                classical_actual_queries=2 * N * M
            )
