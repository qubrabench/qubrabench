from pytest_check import check

import numpy as np
import numpy.typing as npt
import hillclimber


def test_maxsat_values_100() -> None:
    rng = np.random.default_rng(seed=12)

    def random_weights(size: int) -> npt.NDArray[np.int_]:
        return rng.integers(0, 100_000, size)

    history = hillclimber.run(
        3,
        3,
        100,
        n_runs=5,
        rng=rng,
        eps=10**-5,
        random_weights=random_weights,
    )
    history = history.groupby(["k", "r", "n"]).mean(numeric_only=True).reset_index()

    check.equal(history.loc[0, "k"], 3)
    check.equal(history.loc[0, "r"], 3)
    check.equal(history.loc[0, "n"], 100)
    check.almost_equal(history.loc[0, "classical_control_method_calls"], 36.2)
    check.almost_equal(history.loc[0, "classical_actual_queries"], 407.2)
    check.almost_equal(history.loc[0, "classical_expected_queries"], 407.76382374948906)
    check.almost_equal(
        history.loc[0, "quantum_expected_classical_queries"], 503.9227178372942
    )
    check.almost_equal(
        history.loc[0, "quantum_expected_quantum_queries"], 1045.1251428188323
    )
