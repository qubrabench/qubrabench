import pytest
import re

from qubrabench.algorithms.max import max
from qubrabench.bench.stats import QueryStats


def test_max_return_value():
    N = 100
    assert max(range(N), eps=10**-5) == N - 1


def test_max_raises_on_empty_and_no_default():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "max() arg is an empty sequence, and no default value provided"
        ),
    ):
        max([], eps=10**-5)


def test_max_raises_on_stats_requested_and_eps_missing():
    with pytest.raises(
        ValueError, match=re.escape("max() eps not provided, cannot compute stats")
    ):
        stats = QueryStats()
        max(range(100), stats=stats)


def test_max_on_empty_with_default():
    assert max([], eps=10**-5, default=42) == 42
