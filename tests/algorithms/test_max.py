"""This module collects test functions for the qubrabench.max method."""

import re

import pytest

from qubrabench.algorithms.max import max
from qubrabench.benchmark import QueryStats, oracle, track_queries


def test_max_return_value():
    """Tests that the max function returns the maximum value."""
    N = 100
    assert max(range(N), max_failure_probability=10**-5) == N - 1


def test_max_raises_on_empty_and_no_default():
    """Tests that a ValueError is raised by max if an empty sequence is passed"""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "max() arg is an empty sequence, and no default value provided"
        ),
    ):
        max([], max_failure_probability=10**-5)


def test_max_raises_on_stats_requested_and_eps_missing():
    """Tests that a Value error is raised by max if 'error' (failure rate) is not provided"""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "max() parameter 'error' not provided, cannot compute quantum query statistics"
        ),
    ):
        with track_queries():
            max(range(100))


def test_max_on_empty_with_default():
    assert max([], max_failure_probability=10**-5, default=42) == 42


def test_max_stats():
    with track_queries() as tracker:

        @oracle
        def key(x):
            return -((x - 50) ** 2)

        result = max(range(100), key=key, max_failure_probability=10**-5)
        assert result == 50

        assert tracker.get_stats(key) == QueryStats(
            classical_actual_queries=100,
            classical_expected_queries=100,
            quantum_expected_classical_queries=0,
            quantum_expected_quantum_queries=pytest.approx(1405.3368),
        )
