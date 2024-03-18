import numpy as np

from qubrabench.algorithms.amplitude import estimate_amplitude
from qubrabench.datastructures.qndarray import array, state_preparation_unitary


def test_estimate_amplitude(rng):
    N = 20
    vector = rng.random(N)
    qvector = array(vector)

    actual_a = estimate_amplitude(
        state_preparation_unitary(qvector, eps=0),
        0,
        precision=1e-5,
        max_fail_probability=2 / 3,
    )
    expected_a = np.abs(vector[0] / np.linalg.norm(vector)) ** 2

    np.testing.assert_allclose(actual_a, expected_a)

    assert qvector.stats.quantum_expected_quantum_queries == 138232
