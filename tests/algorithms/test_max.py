import pytest
from pytest_check import check

from qubrabench.algorithms.max import max
from qubrabench.bench.stats import QueryStats


def test_max_return_value():
    N = 100
    assert max(range(N), eps=10**-5) == N - 1


def test_max_raises_on_empty_and_no_default():
    with pytest.raises(ValueError):
        max([], eps=10**-5)


def test_max_on_empty_with_default():
    assert max([], eps=10**-5, default=42) == 42
