import numpy as np

from qubrabench.algorithms.amplitude import estimate_amplitude
from qubrabench.benchmark import QueryStats
from qubrabench.datastructures.qndarray import array, state_preparation_unitary


def test_estimate_amplitude(rng):
    N = 20
    data = rng.random(N)
    vector = array(data)

    actual_a = estimate_amplitude(
        state_preparation_unitary(vector, eps=0),
        0,
        precision=1e-5,
        max_fail_probability=2 / 3,
    )
    expected_a = np.abs(data[0] / np.linalg.norm(data)) ** 2

    np.testing.assert_allclose(actual_a, expected_a)
    assert vector.stats == QueryStats(
        classical_actual_queries=0,
        classical_expected_queries=None,
        quantum_expected_classical_queries=0,
        quantum_expected_quantum_queries=138232,
    )
