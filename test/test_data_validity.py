import time

import pytest

import numpy as np

from algorithms.hillclimber_rub import run as rub_run
from algorithms.hillclimber_kit import run as kit_run


# TODO: andere zahlen?
@pytest.mark.kit
def test_kit_values_100():
    history = kit_run(3, 3, 100, 5, 12, None).groupby(["k", "r", "n"]).mean(numeric_only=True).reset_index()
    assert history.loc[0, 'k'] == 3
    assert history.loc[0, 'r'] == 3
    assert history.loc[0, 'n'] == 100
    assert history.loc[0, 'classical_control_method_calls'] == 35.8
    assert history.loc[0, 'classical_actual_queries'] == 470.0
    assert history.loc[0, 'classical_expected_queries'] == 0.0
    assert history.loc[0, 'quantum_expected_classical_queries'] == 548.924429142314
    assert history.loc[0, 'quantum_expected_quantum_queries'] == 1041.8182993248565


@pytest.mark.rub
def test_rub_values_100():
    history = rub_run(3, 3, 100, 5, 12, None).groupby(["k", "r", "n"]).mean(numeric_only=True).reset_index()
    assert history.loc[0, 'k'] == 3
    assert history.loc[0, 'r'] == 3
    assert history.loc[0, 'n'] == 100
    assert history.loc[0, 'classical_control_method_calls'] == 41.8
    assert history.loc[0, 'classical_actual_queries'] == 460.6
    assert history.loc[0, 'classical_expected_queries'] == 445.9318234031114
    assert history.loc[0, 'quantum_expected_classical_queries'] == 546.7700064951359
    assert history.loc[0, 'quantum_expected_quantum_queries'] == 1049.3654516618094
