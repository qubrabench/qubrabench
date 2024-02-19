import pytest
import numpy as np
from nesting import example


@pytest.mark.parametrize("N", [4, 10, 20])
def test_example_planted(rng, N: int):
    for _ in range(20):
        A = rng.random((N, N))
        x = rng.random(N)
        x /= np.linalg.norm(x)
        b = A @ x

        expected = np.any(np.abs(x) >= 0.5)
        actual = example(A, b)
        assert expected == actual
