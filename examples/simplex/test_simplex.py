import numpy as np
from simplex import Simplex


def test_simplex_solution():
    n = 4

    A = np.eye(n)
    b = np.ones(n)
    c = np.ones(n)

    x = Simplex(A, b, c)
    np.testing.assert_allclose(x, np.ones(n))
