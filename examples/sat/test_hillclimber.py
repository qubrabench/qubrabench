from pytest_check import check

import numpy as np
import hillclimber


def random_weights(size):
    return np.random.randint(0, 100_000, size)


def test_maxsat_values_100():
    history = hillclimber.run(
        3, 3, 100, n_runs=5, seed=12, random_weights=random_weights
    )
    history = history.groupby(["k", "r", "n"]).mean(numeric_only=True).reset_index()

    check.equal(history.loc[0, "k"], 3)
    check.equal(history.loc[0, "r"], 3)
    check.equal(history.loc[0, "n"], 100)
    check.almost_equal(history.loc[0, "classical_control_method_calls"], 38)
    check.almost_equal(history.loc[0, "classical_actual_queries"], 418)
    check.almost_equal(history.loc[0, "classical_expected_queries"], 478.80899007437773)
    check.almost_equal(
        history.loc[0, "quantum_expected_classical_queries"], 591.7436280166625
    )
    check.almost_equal(
        history.loc[0, "quantum_expected_quantum_queries"], 1039.2107809631161
    )
