"""This module collects test functions surrounding the hillclimber example."""

import pytest
import hillclimber


def test_maxsat_values_100(rng) -> None:
    """Perform an example run of a simple hillclimber and assure statistic collection.

    Args:
        rng (np.rng): Source of randomness provided by test fixtures
    """

    def random_weights(size):
        return rng.integers(0, 100_000, size)

    history = hillclimber.run(
        3, 3, 100, n_runs=5, rng=rng, error=10**-5, random_weights=random_weights
    )
    history = history.groupby(["k", "r", "n"]).mean(numeric_only=True).reset_index()
    stats = history.loc[0].to_dict()

    assert stats == {
        "k": 3,
        "r": 3,
        "n": 100,
        "classical_control_method_calls": pytest.approx(36.2),
        "classical_actual_queries": pytest.approx(407.2),
        "classical_expected_queries": pytest.approx(407.76382374948906),
        "quantum_expected_classical_queries": pytest.approx(503.9227178372942),
        "quantum_expected_quantum_queries": pytest.approx(1413.1251428188323),
    }
