from pprint import pprint

import numpy as np
import pytest
from nesting import example

from qubrabench.benchmark import track_queries


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


def test_example_planted_stats(rng):
    N = 20
    A = rng.random((N, N))
    x = rng.random(N)
    x /= np.linalg.norm(x)
    b = A @ x

    with track_queries() as tracker:
        example(A, b)
        print()
        pprint(tracker.stats)
