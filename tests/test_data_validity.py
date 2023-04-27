import time

import pytest

import numpy as np

from qubrabench.examples.hillclimber_rub import run as rub_run
from qubrabench.examples.hillclimber_kit import run as kit_run


def random_weights(size):
    return np.random.randint(0, 100_000, size)


# TODO: andere zahlen?
@pytest.mark.kit
def test_kit_values_100():
    history = (
        kit_run(3, 3, 100, runs=5, seed=12, random_weights=random_weights)
        .groupby(["k", "r", "n"])
        .mean(numeric_only=True)
        .reset_index()
    )
    assert history.loc[0, "k"] == 3
    assert history.loc[0, "r"] == 3
    assert history.loc[0, "n"] == 100
    assert history.loc[0, "classical_control_method_calls"] == 39
    assert history.loc[0, "classical_actual_queries"] == 488.2
    assert history.loc[0, "classical_expected_queries"] == 455.0878159026828
    assert history.loc[0, "quantum_expected_classical_queries"] == 556.6182119870105
    assert history.loc[0, "quantum_expected_quantum_queries"] == 1039.134264879581


@pytest.mark.rub
def test_rub_values_100():
    history = (
        rub_run(3, 3, 100, runs=5, seed=12, random_weights=random_weights)
        .groupby(["k", "r", "n"])
        .mean(numeric_only=True)
        .reset_index()
    )
    assert history.loc[0, "k"] == 3
    assert history.loc[0, "r"] == 3
    assert history.loc[0, "n"] == 100
    assert history.loc[0, "classical_control_method_calls"] == 37.6
    assert history.loc[0, "classical_actual_queries"] == 508.4
    assert history.loc[0, "classical_expected_queries"] == 502.29809515885484
    assert history.loc[0, "quantum_expected_classical_queries"] == 627.4230153301583
    assert history.loc[0, "quantum_expected_quantum_queries"] == 1052.6117187935604
