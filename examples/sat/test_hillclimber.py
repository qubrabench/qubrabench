"""This module collects test functions surrounding the hillclimber example."""

import hillclimber
import pytest


def test_maxsat_values_100(rng) -> None:
    """Run the simple hillclimber"""

    def random_weights(size):
        return rng.integers(0, 100_000, size)

    history = hillclimber.run(
        3,
        3,
        100,
        n_runs=5,
        rng=rng,
        error=10**-5,
        random_weights=random_weights,
        steep=False,
    )
    history = history.groupby(["k", "r", "n"]).mean(numeric_only=True).reset_index()
    stats = history.loc[0].to_dict()

    assert stats == {
        "k": 3,
        "r": 3,
        "n": 100,
        "classical_control_method_calls": pytest.approx(36.2),
        "classical_actual_queries": pytest.approx(407.2),
        "classical_expected_queries": pytest.approx(406.76382374948906),
        "quantum_expected_classical_queries": pytest.approx(503.9227178372942),
        "quantum_expected_quantum_queries": pytest.approx(2826.2502856376645),
        "quantum_worst_case_classical_queries": pytest.approx(0.0),
        "quantum_worst_case_quantum_queries": pytest.approx(22746.627798239984),
    }


def test_maxsat_values_100_steep(rng) -> None:
    """Run the steep hillclimber"""

    def random_weights(size):
        return rng.integers(0, 100_000, size)

    history = hillclimber.run(
        3,
        3,
        100,
        n_runs=5,
        rng=rng,
        error=10**-5,
        random_weights=random_weights,
        steep=True,
    )
    history = history.groupby(["k", "r", "n"]).mean(numeric_only=True).reset_index()
    stats = history.loc[0].to_dict()

    assert stats == {
        "k": 3,
        "r": 3,
        "n": 100,
        "classical_control_method_calls": pytest.approx(22.0),
        "classical_actual_queries": pytest.approx(2200.0),
        "classical_expected_queries": pytest.approx(2200.0),
        "quantum_expected_classical_queries": pytest.approx(0.0),
        "quantum_expected_quantum_queries": pytest.approx(84320.20904223222),
    }
