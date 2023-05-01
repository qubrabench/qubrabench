import numpy as np

from pytest_check import check

from sat import SatInstance
from bruteforce import run_specific_instance


def test_simple_sat_instance():
    inst = SatInstance(
        k=2, clauses=np.array([[1, 0, 0], [0, 1, 1], [0, -1, -1]], dtype=int)
    )

    history = run_specific_instance(
        inst, n_runs=200, eps=10**-5, rng=np.random.default_rng(seed=12)
    )
    history = history.groupby(["n", "k", "m"]).mean(numeric_only=True).reset_index()

    check.equal(history.loc[0, "n"], 3)
    check.equal(history.loc[0, "k"], 2)
    check.equal(history.loc[0, "m"], 3)
    check.equal(history.loc[0, "T"], 2)

    check.almost_equal(history.loc[0, "classical_control_method_calls"], 0)
    check.almost_equal(history.loc[0, "classical_actual_queries"], 3.285)
    check.almost_equal(history.loc[0, "classical_expected_queries"], 3)
    check.almost_equal(history.loc[0, "quantum_expected_classical_queries"], 4)
    check.almost_equal(history.loc[0, "quantum_expected_quantum_queries"], 0)
