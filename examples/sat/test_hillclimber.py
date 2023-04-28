import time

import pytest
from pytest_check import check

import numpy as np

from hillclimber_rub import run as rub_run
from hillclimber_kit import run as kit_run


def random_weights(size):
    return np.random.randint(0, 100_000, size)


@pytest.mark.parametrize("run", [kit_run, rub_run])
def test_maxsat_values_100(run):
    history = (
        kit_run(3, 3, 100, runs=5, seed=12, random_weights=random_weights)
        .groupby(["k", "r", "n"])
        .mean(numeric_only=True)
        .reset_index()
    )
    check.equal(history.loc[0, "k"], 3)
    check.equal(history.loc[0, "r"], 3)
    check.equal(history.loc[0, "n"], 100)
    check.almost_equal(history.loc[0, "classical_control_method_calls"], 33.6)
    check.almost_equal(history.loc[0, "classical_actual_queries"], 414.0)
    check.almost_equal(history.loc[0, "classical_expected_queries"], 420.54904671985)
    check.almost_equal(
        history.loc[0, "quantum_expected_classical_queries"], 520.7110812189618
    )
    check.almost_equal(
        history.loc[0, "quantum_expected_quantum_queries"], 1038.0306993360725
    )


# def test_maxsat_values_100():
#     history = (
#         rub_run(3, 3, 100, runs=5, seed=12, random_weights=random_weights)
#         .groupby(["k", "r", "n"])
#         .mean(numeric_only=True)
#         .reset_index()
#     )
#     check.equal(history.loc[0, "k"], 3)
#     check.equal(history.loc[0, "r"], 3)
#     check.equal(history.loc[0, "n"], 100)
#     check.almost_equal(history.loc[0, "classical_control_method_calls"], 37.6)
#     check.almost_equal(history.loc[0, "classical_actual_queries"], 508.4)
#     check.almost_equal(history.loc[0, "classical_expected_queries"], 502.29809515885484)
#     check.almost_equal(
#         history.loc[0, "quantum_expected_classical_queries"], 627.4230153301583
#     )
#     check.almost_equal(
#         history.loc[0, "quantum_expected_quantum_queries"], 1052.6117187935604
#     )
