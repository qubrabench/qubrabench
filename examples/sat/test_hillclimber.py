import time

import pytest
from pytest_check import check

import numpy as np

from hillclimber_rub import run as rub_run
from hillclimber_kit import run as kit_run


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
    with check:
        assert history.loc[0, "k"] == 3
    with check:
        assert history.loc[0, "r"] == 3
    with check:
        assert history.loc[0, "n"] == 100
    with check:
        assert history.loc[0, "classical_control_method_calls"] == 38.6
    with check:
        assert history.loc[0, "classical_actual_queries"] == 442.6
    with check:
        assert history.loc[0, "classical_expected_queries"] == 447.55989714369844
    with check:
        assert history.loc[0, "quantum_expected_classical_queries"] == 546.0817997883502
    with check:
        assert history.loc[0, "quantum_expected_quantum_queries"] == 1033.1587409686078


@pytest.mark.rub
def test_rub_values_100():
    history = (
        rub_run(3, 3, 100, runs=5, seed=12, random_weights=random_weights)
        .groupby(["k", "r", "n"])
        .mean(numeric_only=True)
        .reset_index()
    )
    with check:
        assert history.loc[0, "k"] == 3
    with check:
        assert history.loc[0, "r"] == 3
    with check:
        assert history.loc[0, "n"] == 100
    with check:
        assert history.loc[0, "classical_control_method_calls"] == 37.6
    with check:
        assert history.loc[0, "classical_actual_queries"] == 508.4
    with check:
        assert history.loc[0, "classical_expected_queries"] == 502.29809515885484
    with check:
        assert history.loc[0, "quantum_expected_classical_queries"] == 627.4230153301583
    with check:
        assert history.loc[0, "quantum_expected_quantum_queries"] == 1052.6117187935604
