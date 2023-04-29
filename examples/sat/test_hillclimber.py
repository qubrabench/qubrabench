from pytest_check import check

import numpy as np
import hillclimber


def random_weights(size):
    return np.random.randint(0, 100_000, size)


def test_maxsat_values_100():
    history = hillclimber.run(3, 3, 100, runs=5, seed=12, random_weights=random_weights)
    history = history.groupby(["k", "r", "n"]).mean(numeric_only=True).reset_index()
    check.equal(history.loc[0, "k"], 3)
    check.equal(history.loc[0, "r"], 3)
    check.equal(history.loc[0, "n"], 100)
    check.almost_equal(history.loc[0, "classical_control_method_calls"], 37.6)
    check.almost_equal(history.loc[0, "classical_actual_queries"], 508.4)
    check.almost_equal(history.loc[0, "classical_expected_queries"], 502.29809515885484)
    check.almost_equal(
        history.loc[0, "quantum_expected_classical_queries"], 627.4230153301583
    )
    check.almost_equal(
        history.loc[0, "quantum_expected_quantum_queries"], 1052.6117187935604
    )
