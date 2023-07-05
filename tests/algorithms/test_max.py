"""This module collects test functions for the qubrabench.max method."""

import pytest
import re

from qubrabench.algorithms.max import max
from qubrabench.stats import QueryStats


def test_max_return_value():
    """Tests that the max function returns the maximum value."""
    N = 100
    assert max(range(N), error=10**-5) == N - 1


def test_max_raises_on_empty_and_no_default():
    """Tests that a ValueError is raised by max if an empty sequence is passed"""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "max() arg is an empty sequence, and no default value provided"
        ),
    ):
        max([], error=10**-5)


def test_max_raises_on_stats_requested_and_eps_missing():
    """Tests that a Value error is raised by max if 'error' (failure rate) is not provided"""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "max() parameter 'error' not provided, cannot compute quantum query statistics"
        ),
    ):
        stats = QueryStats()
        max(range(100), stats=stats)


def test_max_on_empty_with_default():
    assert max([], error=10**-5, default=42) == 42
