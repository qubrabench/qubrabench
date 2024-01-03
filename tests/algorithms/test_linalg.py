import numpy as np
import pytest

from qubrabench.datastructures.blockencoding import BlockEncoding
from qubrabench.algorithms.linalg import solve


@pytest.mark.parametrize("N", [5, 10, 15, 20, 25, 30])
def test_solve(rng, N: int):
    A = rng.random(size=(N, N))
    b = rng.random(size=N)

    enc_A = BlockEncoding(A, alpha=np.linalg.norm(A), error=0)
    enc_b = BlockEncoding(b, alpha=np.linalg.norm(b), error=0)

    enc_y = solve(enc_A, enc_b)
    y = enc_y.matrix

    np.testing.assert_allclose(A @ y, b)
