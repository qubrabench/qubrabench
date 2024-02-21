import numpy as np
from pytest import approx
from simplex import FindColumn

from qubrabench.benchmark import track_queries
from qubrabench.datastructures.qndarray import Qndarray


def test_find_column(rng):
    A = np.array([[1, 0, 1], [0, 1, 0]])
    B = [0, 1]
    c = np.array([1, 0, 0.5])

    np.testing.assert_allclose(np.linalg.norm(c[B]), 1)
    assert np.linalg.norm(A[:, B], ord=2) <= 1

    A = Qndarray(A)
    c = Qndarray(c)

    with track_queries() as tracker:
        k = FindColumn(A, B, c, epsilon=1e-3)
        assert k == 2

        queries = tracker.get_stats(A).quantum_expected_quantum_queries
        assert queries == approx(1015444.6104100969)
