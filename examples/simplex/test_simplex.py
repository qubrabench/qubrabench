import numpy as np
import scipy as sp
from simplex import Simplex


def test_trivial_simplex_solution():
    n = 4

    A = np.eye(n)
    b = np.ones(n)
    c = np.ones(n)

    x = Simplex(A, b, c)
    np.testing.assert_allclose(x, np.ones(n))


def test_simplex_on_random_instances(rng):
    m, n = 4, 10

    for _ in range(5):
        A = rng.random((m, n))
        x = rng.random(n)
        b = A @ x
        c = rng.random(n)

        x_expected = sp.optimize.linprog(c, A_eq=A, b_eq=b).x
        x_actual = Simplex(A, b, c)
        assert x_actual is not None

        assert np.allclose(np.inner(c, x_expected), np.inner(c, x_actual))
        assert np.allclose(A @ x, b)
