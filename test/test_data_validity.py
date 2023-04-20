import time

import pytest

from algorithms.hillclimber_rub import run as rub_run
from algorithms.hillclimber_kit import run as kit_run


# TODO: andere zahlen?
@pytest.mark.kit
def test_kit_values_100():
    stats = kit_run(3, 3, 100, 5, 12, None)
    assert stats['k'] == 3
    assert stats['r'] == 3
    assert stats['n'] == 100
    assert stats['classical_control_method_calls'] == 39
    assert stats['classical_actual_queries'] == 507
    assert stats['classical_expected_queries'] == 0
    assert stats['quantum_expected_classical_queries'] == 609.6189241463908
    assert stats['quantum_expected_quantum_queries'] == 1037.1363201388592


@pytest.mark.rub
def test_rub_values_100():
    stats = rub_run(3, 3, 100, 5, 12, None)
    assert stats['k'] == 3
    assert stats['r'] == 3
    assert stats['n'] == 100
    assert stats['classical_control_method_calls'] == 40
    assert stats['classical_actual_queries'] == 454
    assert stats['classical_expected_queries'] == 458.859883653203
    assert stats['quantum_expected_classical_queries'] == 573.7982278376396
    assert stats['quantum_expected_quantum_queries'] == 1069.6659857019176
