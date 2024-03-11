import numpy as np

from qubrabench.algorithms.amplitude import estimate_amplitude
from qubrabench.benchmark import default_tracker
from qubrabench.datastructures.qndarray import state_preparation_unitary


def test_estimate_amplitude(rng):
    N = 20
    vector = rng.random(N)
    enc_vector = state_preparation_unitary(vector, eps=0)

    actual_a = estimate_amplitude(
        enc_vector, 0, precision=1e-5, max_fail_probability=2 / 3
    )
    expected_a = np.abs(vector[0] / np.linalg.norm(vector)) ** 2

    np.testing.assert_allclose(actual_a, expected_a)

    stats = default_tracker().get_stats(enc_vector)
    assert stats.quantum_expected_quantum_queries == 69116
