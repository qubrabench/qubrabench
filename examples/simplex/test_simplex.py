import numpy as np
from pytest import approx
from simplex import FindColumn

from qubrabench.datastructures.qndarray import array


def test_find_column(rng):
    A = np.array([[1, 0, 1], [0, 1, 0]])
    B = [0, 1]
    c = np.array([1, 0, 0.5])

    np.testing.assert_allclose(np.linalg.norm(c[B]), 1)
    assert np.linalg.norm(A[:, B], ord=2) <= 1

    A = array(A)
    c = array(c)

    k = FindColumn(A, B, c, epsilon=1e-3)
    assert k == 2

    assert A.stats.quantum_expected_quantum_queries == approx(43645071.062891126)
