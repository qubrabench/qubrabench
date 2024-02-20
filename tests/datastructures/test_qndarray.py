import numpy as np

from qubrabench.benchmark import QueryStats, track_queries
from qubrabench.datastructures.qndarray import Qndarray


def test_qndarray_iterate(rng):
    for _ in range(5):
        N, M = rng.integers(10, 50, size=2)
        mat = rng.random(size=(N, M))
        qmat = Qndarray(mat)

        with track_queries() as tracker:
            mat_sum = 0
            for i in range(N):
                for j in range(M):
                    mat_sum += qmat[i, j]
            np.testing.assert_allclose(mat_sum, np.sum(mat))
            assert tracker.get_stats(qmat) == QueryStats(classical_actual_queries=N * M)

            mat_max = 0
            for i in range(N):
                for j in range(M):
                    mat_max = max(mat_max, qmat[i, j])

            np.testing.assert_allclose(mat_max, np.max(mat))
            assert tracker.get_stats(qmat) == QueryStats(
                classical_actual_queries=2 * N * M
            )


def test_qndarray_constructor_idempotent():
    a = Qndarray(np.eye(3))
    b = Qndarray(a)
    assert a == b


def test_qndarray_view():
    a = Qndarray(np.eye(3))
    row = a[:, 0]

    with track_queries() as tracker:
        _ = row[0]
        assert tracker.get_stats(row) == QueryStats(classical_actual_queries=1)
        assert tracker.get_stats(a) == QueryStats(classical_actual_queries=1)


def test_qndarray_nested_views():
    a = Qndarray(np.eye(3))
    b = a[:, :]
    row = b[:, 0]

    with track_queries() as tracker:
        _ = row[0]

        assert tracker.get_stats(row) == QueryStats(classical_actual_queries=1)
        assert tracker.get_stats(a) == QueryStats(classical_actual_queries=1)
        assert tracker.get_stats(b) == QueryStats(classical_actual_queries=1)