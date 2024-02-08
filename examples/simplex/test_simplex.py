import numpy as np
from simplex import FindColumn


def test_find_column(rng):
    N, M = 4, 6

    B = [0, 1, 2, 3]

    A = rng.random((N, M))
    A /= np.linalg.norm(A[:, B])

    c = rng.random(M)
    c /= np.linalg.norm(c[B])

    k = FindColumn(A, B, c, epsilon=1e-3)
    print(k)
