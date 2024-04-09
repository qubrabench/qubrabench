"""This module collects test functions for the examples.sat.schoening module."""

import numpy as np
import pytest
from sat import SatInstance
from schoening import (
    schoening_solve,
    schoening_solve__bruteforce_over_starting_assigment,
)

from qubrabench.benchmark import QueryStats, track_queries


@pytest.fixture
def sample_sat_instance() -> SatInstance:
    return SatInstance(
        k=3,
        clauses=np.array([[1, 1, -1], [1, -1, 1], [-1, 1, 1], [1, -1, -1]], dtype=int),
    )


def test_solve(rng, sample_sat_instance) -> None:
    """Schoening's algorithm - simple variant"""
    with track_queries() as tracker:
        x = schoening_solve(sample_sat_instance, rng=rng, error=10**-5)

        # check stats
        assert tracker.get_stats(sample_sat_instance.evaluate) == QueryStats(
            classical_actual_queries=2,
            classical_expected_queries=7,
            quantum_expected_classical_queries=pytest.approx(15.22222222222222),
            quantum_expected_quantum_queries=pytest.approx(0),
        )

        # validate solution
        assert x is not None and sample_sat_instance.evaluate(x)


def test_bruteforce_steps(rng, sample_sat_instance) -> None:
    """Schöning's algorithm - variant: classical search over starting assignment"""
    with track_queries() as tracker:
        x = schoening_solve__bruteforce_over_starting_assigment(
            sample_sat_instance, rng=rng, error=10**-5
        )

        # check stats
        assert tracker.get_stats(sample_sat_instance.evaluate) == QueryStats(
            classical_actual_queries=2,
            classical_expected_queries=pytest.approx(11),
            quantum_expected_classical_queries=pytest.approx(24.703703703703702),
            quantum_expected_quantum_queries=pytest.approx(0),
        )

        # validate solution
        assert x is not None and sample_sat_instance.evaluate(x)
