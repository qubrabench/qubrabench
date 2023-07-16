"""This module collects test functions for the qubrabench.algorithms.statepreparation method."""

import numpy as np
from qubrabench.algorithms.statepreparation import (
    generate_amplitude_vect,
    phase_angle_dict,
    circuit,
)


def test_statepreparation():
    n_qubit = 4
    N = 2**n_qubit
    n_runs = 10

    for _ in range(n_runs):
        for d in range(1, N):
            vect = generate_amplitude_vect(n_qubit, d)
            angle_phase_dict = phase_angle_dict(vect)

            # Check if the angles are correctly computed (comment if you don't care about the checking)
            phi, c = circuit(angle_phase_dict)
            assert np.all(abs(vect - phi) <= 1e-7)
